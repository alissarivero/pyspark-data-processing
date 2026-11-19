"""Transformations, SQL, cache comparison, and a small classifier.

The fact table is partitioned by sale_month. Revenue work filters that column
before any join so Spark can skip entire 2022 directories. Dimension tables
are projected down to the columns this query uses and broadcast, so the fact
table is not shuffled just to look up category and customer attributes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from retail_pipeline.constants import UNUSED_FACT_COLUMNS
from retail_pipeline.session import explain_string, highlight_plan, job_count, save_text, snapshot_sql
from retail_pipeline.util import (
    CUSTOMERS,
    OUTPUT_DIR,
    PRODUCTS,
    REPORTS_DIR,
    TRANSACTIONS,
    fmt_bytes,
)

REVENUE_MONTH = "2023-01"
MEASURE_KEYS = [
    "sale_year",
    "sale_month",
    "store_region",
    "category",
    "subcategory",
    "channel",
    "price_band",
]


def normalize_status(value):
    """Python UDF body. Using this on the hot path blocks Parquet pushdown."""
    if value is None:
        return None
    return str(value).strip().upper()


def load_tables(spark):
    transactions = spark.read.parquet(str(TRANSACTIONS))
    products = spark.read.parquet(str(PRODUCTS)).withColumnRenamed("is_active", "product_active")
    customers = spark.read.parquet(str(CUSTOMERS)).withColumnRenamed("is_active", "customer_active")
    return transactions, products, customers


def add_measures(df):
    return (
        df.withColumn("gross_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
        .withColumn("net_revenue", F.round(F.col("gross_revenue") * (1 - F.col("discount_pct")), 2))
        .withColumn("sale_year", F.year("sale_date"))
        .withColumn(
            "price_band",
            F.when(F.col("unit_price") < 25, F.lit("budget"))
            .when(F.col("unit_price") < 75, F.lit("mid"))
            .otherwise(F.lit("premium")),
        )
        .drop("sale_date")
    )


def monthly_agg(df):
    return df.groupBy(*MEASURE_KEYS).agg(
        F.count("*").alias("orders"),
        F.sum("quantity").alias("units"),
        F.round(F.sum("gross_revenue"), 2).alias("gross_revenue"),
        F.round(F.sum("net_revenue"), 2).alias("net_revenue"),
        F.round(F.avg("discount_pct"), 4).alias("avg_discount"),
        F.countDistinct("customer_id").alias("customers"),
    )


def build_enriched(transactions, products, customers):
    """Narrow, filter, then broadcast. This is the pipeline the write-up times."""
    fact = transactions.select(
        "transaction_id",
        "customer_id",
        "product_id",
        "sale_date",
        "sale_month",
        "quantity",
        "unit_price",
        "discount_pct",
        "channel",
        "status",
        "store_region",
    )
    fact = (
        fact.filter(F.col("sale_month") >= F.lit(REVENUE_MONTH))
        .filter(F.col("status") == F.lit("COMPLETED"))
        .filter(F.col("discount_pct") <= F.lit(0.5))
        .filter(F.col("quantity") > F.lit(0))
    )
    fact = add_measures(fact)
    products_dim = products.filter(F.col("product_active")).select(
        "product_id", "category", "subcategory", "brand"
    )
    customers_dim = customers.filter(F.col("customer_active")).select(
        "customer_id", "segment", "loyalty_tier", "customer_region"
    )
    return fact.join(F.broadcast(products_dim), "product_id", "inner").join(
        F.broadcast(customers_dim), "customer_id", "inner"
    )


def build_naive(transactions, products, customers):
    """Same aggregation, but the joins are written first and broadcast is disabled.

    Catalyst still pushes the status and month filters into the scan. The plan
    difference that remains is the join: this version sort-merge joins the fact
    table, which shuffles it. The optimized version broadcasts the dimensions.
    """
    joined = transactions.join(products, "product_id", "inner").join(customers, "customer_id", "inner")
    filtered = (
        joined.filter(F.col("sale_month") >= F.lit(REVENUE_MONTH))
        .filter(F.col("status") == F.lit("COMPLETED"))
        .filter(F.col("discount_pct") <= F.lit(0.5))
        .filter(F.col("quantity") > F.lit(0))
        .filter(F.col("product_active") == F.lit(True))
        .filter(F.col("customer_active") == F.lit(True))
    )
    return monthly_agg(add_measures(filtered))


def _timed(spark, label: str, fn):
    spark.sparkContext.setJobDescription(label)
    print(f"\n=== {label} ===", flush=True)
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    print(f"{label} took {elapsed:.2f}s", flush=True)
    detail = snapshot_sql(spark, label)
    return value, elapsed, detail


def lazy_versus_actions(spark, transactions, month: str) -> dict:
    """Transformations only extend the plan. The count is what schedules a job."""
    before = job_count(spark)
    started = time.perf_counter()
    lazy_df = (
        transactions.filter(F.col("sale_month") == F.lit(month))
        .filter(F.col("status") == F.lit("COMPLETED"))
        .withColumn("gross_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
        .select("transaction_id", "sale_month", "status", "gross_revenue")
    )
    # Touch the plan so attribute access cannot be accused of skipping the builder.
    columns = lazy_df.columns
    build_seconds = time.perf_counter() - started
    after_transform = job_count(spark)
    plan = explain_string(lazy_df, "simple")
    after_explain = job_count(spark)
    row_count, action_seconds, _detail = _timed(spark, "lazy-demo-count", lazy_df.count)
    after_action = job_count(spark)
    report = {
        "month": month,
        "columns": columns,
        "build_seconds": round(build_seconds, 4),
        "action_seconds": round(action_seconds, 4),
        "jobs_before": before,
        "jobs_after_transform": after_transform,
        "jobs_after_explain": after_explain,
        "jobs_after_action": after_action,
        "rows": int(row_count),
        "plan": plan,
    }
    print(
        f"Transformations built in {build_seconds:.4f}s. "
        f"Jobs {before} -> {after_transform} after the transformations, "
        f"{after_explain} after explain, {after_action} after count()."
    )
    print(f"count() returned {row_count:,} rows in {action_seconds:.2f}s")
    save_text(REPORTS_DIR / "explain_lazy.txt", plan)
    return report


def pruning_experiment(spark, transactions) -> dict:
    by_month = transactions.filter(F.col("sale_month") >= F.lit(REVENUE_MONTH))
    by_date = transactions.filter(F.col("sale_date") >= F.lit(REVENUE_MONTH + "-01").cast("date"))
    month_plan = explain_string(by_month, "extended")
    date_plan = explain_string(by_date, "extended")
    save_text(REPORTS_DIR / "explain_pruning_month.txt", month_plan)
    save_text(REPORTS_DIR / "explain_pruning_date.txt", date_plan)
    month_rows, month_seconds, month_sql = _timed(spark, "pruning-by-month", by_month.count)
    date_rows, date_seconds, date_sql = _timed(spark, "pruning-by-date", by_date.count)
    return {
        "month_rows": int(month_rows),
        "date_rows": int(date_rows),
        "month_seconds": round(month_seconds, 3),
        "date_seconds": round(date_seconds, 3),
        "month_highlights": highlight_plan(month_plan),
        "date_highlights": highlight_plan(date_plan),
        "month_sql": _metric_lines(month_sql),
        "date_sql": _metric_lines(date_sql),
    }


def udf_pushdown_experiment(spark, transactions, month: str) -> dict:
    one_month = transactions.filter(F.col("sale_month") == F.lit(month))
    native = one_month.filter(F.col("status") == F.lit("COMPLETED"))
    normalize = F.udf(normalize_status, StringType())
    blocked = one_month.withColumn("status", normalize(F.col("status"))).filter(
        F.col("status") == F.lit("COMPLETED")
    )
    native_plan = explain_string(native, "extended")
    blocked_plan = explain_string(blocked, "extended")
    save_text(REPORTS_DIR / "explain_udf_native.txt", native_plan)
    save_text(REPORTS_DIR / "explain_udf_blocked.txt", blocked_plan)
    native_rows, native_seconds, _native_sql = _timed(spark, "udf-native-filter", native.count)
    blocked_rows, blocked_seconds, _blocked_sql = _timed(spark, "udf-blocked-filter", blocked.count)
    return {
        "month": month,
        "native_rows": int(native_rows),
        "blocked_rows": int(blocked_rows),
        "native_seconds": round(native_seconds, 3),
        "blocked_seconds": round(blocked_seconds, 3),
        "native_pushed": "EqualTo(status,COMPLETED)" in native_plan or "EqualTo(status, COMPLETED)" in native_plan,
        "blocked_mentions_python": "PythonUDF" in blocked_plan or "pythonUDF" in blocked_plan or "BatchEvalPython" in blocked_plan,
    }


def _with_conf(spark, updates: dict[str, str]):
    previous = {}
    for key, value in updates.items():
        try:
            previous[key] = spark.conf.get(key)
        except Exception:
            previous[key] = None
        spark.conf.set(key, value)
    return previous


def _restore_conf(spark, previous: dict) -> None:
    for key, value in previous.items():
        if value is None:
            continue
        spark.conf.set(key, value)


def run_naive(spark, transactions, products, customers, enabled: bool) -> dict:
    naive = build_naive(transactions, products, customers)
    # Explain under the shuffle-join settings so the text matches the job we time.
    previous = _with_conf(
        spark,
        {
            "spark.sql.autoBroadcastJoinThreshold": "-1",
            "spark.sql.adaptive.autoBroadcastJoinThreshold": "-1",
        },
    )
    try:
        plan = explain_string(naive, "extended")
        save_text(REPORTS_DIR / "explain_naive.txt", plan)
        save_text(REPORTS_DIR / "explain_naive_highlights.txt", highlight_plan(plan))
        if not enabled:
            return {"ran": False, "plan_saved": True, "highlights": highlight_plan(plan)}
        totals, elapsed, detail = _timed(
            spark,
            "naive-shuffle-join",
            lambda: naive.agg(
                F.sum("orders").alias("orders"),
                F.sum("net_revenue").alias("net_revenue"),
            ).collect()[0],
        )
        return {
            "ran": True,
            "seconds": round(elapsed, 3),
            "orders": int(totals["orders"]),
            "net_revenue": float(totals["net_revenue"]),
            "highlights": highlight_plan(plan),
            "sql_metrics": _metric_lines(detail),
            "sort_merge": "SortMergeJoin" in plan,
            "broadcast": "BroadcastHashJoin" in plan,
        }
    finally:
        _restore_conf(spark, previous)


def run_optimized(spark, enriched) -> dict:
    monthly = monthly_agg(enriched)
    plan = explain_string(monthly, "extended")
    formatted = explain_string(monthly, "formatted")
    save_text(REPORTS_DIR / "explain_optimized.txt", plan)
    save_text(REPORTS_DIR / "explain_optimized_formatted.txt", formatted)
    save_text(REPORTS_DIR / "explain_optimized_highlights.txt", highlight_plan(plan))
    output = OUTPUT_DIR / "monthly_sales"
    _rows, elapsed, detail = _timed(
        spark,
        "optimized-monthly-write",
        lambda: monthly.write.mode("overwrite").partitionBy("sale_year").parquet(str(output)),
    )
    stored = spark.read.parquet(str(output))
    spark.sparkContext.setJobDescription("optimized-monthly-totals")
    totals = stored.agg(F.sum("orders").alias("orders"), F.sum("net_revenue").alias("net_revenue")).collect()[0]
    return {
        "seconds": round(elapsed, 3),
        "orders": int(totals["orders"]),
        "net_revenue": float(totals["net_revenue"]),
        "output": str(output),
        "highlights": highlight_plan(plan),
        "formatted": formatted,
        "sql_metrics": _metric_lines(detail),
        "broadcast": "BroadcastHashJoin" in plan,
        "partition_filters": "PartitionFilters" in plan,
        "pushed_filters": "PushedFilters" in plan,
        "omits_unused_columns": all(column not in plan for column in UNUSED_FACT_COLUMNS),
    }


def sql_on_monthly(spark) -> dict:
    monthly = spark.read.parquet(str(OUTPUT_DIR / "monthly_sales"))
    monthly.createOrReplaceTempView("monthly_sales")
    plan_df = spark.sql(
        """
        SELECT category,
               store_region,
               SUM(net_revenue) AS net_revenue,
               SUM(orders) AS orders
        FROM monthly_sales
        WHERE sale_month >= '2024-01'
        GROUP BY category, store_region
        ORDER BY net_revenue DESC
        LIMIT 15
        """
    )
    save_text(REPORTS_DIR / "explain_sql_monthly.txt", explain_string(plan_df, "extended"))
    rows, elapsed, _detail = _timed(spark, "sql-top-category-region", plan_df.collect)
    payload = [row.asDict() for row in rows]
    _write_dicts(REPORTS_DIR / "top_category_region.csv", payload)
    return {"seconds": round(elapsed, 3), "rows": payload}


def sql_status_mix(spark, transactions, customers) -> dict:
    recent = transactions.filter(F.col("sale_month") >= F.lit(REVENUE_MONTH)).select(
        "customer_id", "channel", "status", "store_region", "sale_month"
    )
    customers.select("customer_id", "customer_region", "customer_active").createOrReplaceTempView("customer_dim")
    recent.createOrReplaceTempView("recent_fact")
    query = spark.sql(
        """
        SELECT /*+ BROADCAST(c) */
               f.channel AS channel,
               CASE
                   WHEN f.store_region = c.customer_region THEN 'same_region'
                   ELSE 'cross_region'
               END AS region_match,
               f.status AS status,
               COUNT(*) AS transactions
        FROM recent_fact f
        JOIN customer_dim c
          ON f.customer_id = c.customer_id
        WHERE c.customer_active = true
        GROUP BY f.channel,
                 CASE
                     WHEN f.store_region = c.customer_region THEN 'same_region'
                     ELSE 'cross_region'
                 END,
                 f.status
        ORDER BY channel, region_match, status
        """
    )
    save_text(REPORTS_DIR / "explain_sql_status.txt", explain_string(query, "extended"))
    rows, elapsed, detail = _timed(spark, "sql-status-by-channel", query.collect)
    payload = [row.asDict() for row in rows]
    _write_dicts(REPORTS_DIR / "status_by_channel.csv", payload)
    return {
        "seconds": round(elapsed, 3),
        "rows": payload,
        "sql_metrics": _metric_lines(detail),
        "broadcast": "BroadcastHashJoin" in explain_string(query, "extended") or "BROADCAST" in explain_string(query, "simple"),
    }


def cache_demo(spark, enriched) -> dict:
    """Two actions on one grouped lineage, then the same two actions after cache.

    The grouped customer table is the reused result. Caching the raw fact table
    would pin about a gigabyte of rows for little benefit, because Parquet is
    already a cheap scan. The expensive part here is the shuffle, and that is
    what the second uncached action repeats.
    """
    enriched.createOrReplaceTempView("enriched_sales")
    customer_value = spark.sql(
        """
        SELECT customer_id,
               segment,
               loyalty_tier,
               customer_region,
               COUNT(*) AS orders,
               ROUND(SUM(net_revenue), 2) AS lifetime_net,
               ROUND(AVG(discount_pct), 4) AS avg_discount
        FROM enriched_sales
        GROUP BY customer_id, segment, loyalty_tier, customer_region
        """
    )
    before = explain_string(customer_value, "extended")
    save_text(REPORTS_DIR / "explain_customer_before_cache.txt", before)

    def _average():
        return customer_value.agg(F.avg("lifetime_net").alias("avg_lifetime_net")).collect()[0]["avg_lifetime_net"]

    _count1, uncached_count_s, _d1 = _timed(spark, "cache-uncached-count", customer_value.count)
    avg1, uncached_avg_s, _d2 = _timed(spark, "cache-uncached-avg", _average)
    customer_value.persist()
    _count2, materialize_s, _d3 = _timed(spark, "cache-materialize", customer_value.count)
    _count3, cached_count_s, _d4 = _timed(spark, "cache-cached-count", customer_value.count)
    avg2, cached_avg_s, _d5 = _timed(spark, "cache-cached-avg", _average)
    customer_value.createOrReplaceTempView("customer_value")
    after = explain_string(
        spark.sql("SELECT segment, COUNT(*) AS customers FROM customer_value GROUP BY segment"),
        "extended",
    )
    save_text(REPORTS_DIR / "explain_customer_after_cache.txt", after)
    segment_query = spark.sql(
        """
        SELECT segment,
               loyalty_tier,
               COUNT(*) AS customers,
               ROUND(AVG(lifetime_net), 2) AS avg_lifetime_net,
               ROUND(AVG(orders), 2) AS avg_orders
        FROM customer_value
        GROUP BY segment, loyalty_tier
        ORDER BY avg_lifetime_net DESC
        """
    )
    segment_rows, segment_s, _detail = _timed(spark, "sql-segment-loyalty", segment_query.collect)
    payload = [row.asDict() for row in segment_rows]
    _write_dicts(REPORTS_DIR / "segment_loyalty.csv", payload)
    customer_value.unpersist()
    return {
        "uncached_count_seconds": round(uncached_count_s, 3),
        "uncached_avg_seconds": round(uncached_avg_s, 3),
        "materialize_seconds": round(materialize_s, 3),
        "cached_count_seconds": round(cached_count_s, 3),
        "cached_avg_seconds": round(cached_avg_s, 3),
        "segment_sql_seconds": round(segment_s, 3),
        "avg_lifetime_uncached": float(avg1),
        "avg_lifetime_cached": float(avg2),
        "customers": int(_count1),
        "after_plan_uses_cache": "InMemory" in after,
        "segments": payload,
    }


def run_ml(spark, transactions, products, customers) -> dict:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

    sample = (
        transactions.filter(F.col("sale_month") >= F.lit(REVENUE_MONTH))
        .filter(F.col("status").isin("COMPLETED", "CANCELLED"))
        .filter(F.pmod(F.col("transaction_id"), F.lit(40)) == F.lit(0))
        .select(
            "transaction_id",
            "customer_id",
            "product_id",
            "quantity",
            "unit_price",
            "discount_pct",
            "channel",
            "status",
        )
    )
    labeled = (
        sample.join(F.broadcast(products.filter(F.col("product_active")).select("product_id", "category")), "product_id")
        .join(
            F.broadcast(
                customers.filter(F.col("customer_active")).select("customer_id", "segment", "loyalty_tier")
            ),
            "customer_id",
        )
        .withColumn("label", (F.col("status") == F.lit("CANCELLED")).cast("double"))
        .select("quantity", "unit_price", "discount_pct", "channel", "category", "segment", "loyalty_tier", "label")
    )
    labeled.persist()
    spark.sparkContext.setJobDescription("ml-materialize-sample")
    sample_rows = labeled.count()
    categorical = ["channel", "category", "segment", "loyalty_tier"]
    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep") for column in categorical
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{column}_idx", outputCol=f"{column}_oh", dropLast=True) for column in categorical
    ]
    assembler = VectorAssembler(
        inputCols=["quantity", "unit_price", "discount_pct", *[f"{column}_oh" for column in categorical]],
        outputCol="features",
    )
    learner = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30, regParam=0.05, elasticNetParam=0.0)
    train, test = labeled.randomSplit([0.8, 0.2], seed=42)
    pipeline = Pipeline(stages=[*indexers, *encoders, assembler, learner])
    spark.sparkContext.setJobDescription("ml-fit")
    started = time.perf_counter()
    model = pipeline.fit(train)
    fit_seconds = time.perf_counter() - started
    spark.sparkContext.setJobDescription("ml-transform")
    predictions = model.transform(test)
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)
    aupr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(predictions)
    confusion = [row.asDict() for row in predictions.groupBy("label", "prediction").count().collect()]
    label_counts = [row.asDict() for row in labeled.groupBy("label").count().collect()]
    positives = sum(int(row["count"]) for row in label_counts if row["label"] == 1.0)
    baseline = max(positives, sample_rows - positives) / sample_rows if sample_rows else None
    coefficients = _coefficients(model, predictions)
    preview = [
        row.asDict()
        for row in predictions.select(
            "channel", "category", "segment", "quantity", "discount_pct", "label", "prediction"
        ).limit(12).collect()
    ]
    labeled.unpersist()
    result = {
        "sample_rows": int(sample_rows),
        "positive_rows": int(positives),
        "positive_rate": (positives / sample_rows) if sample_rows else None,
        "majority_class_accuracy": baseline,
        "fit_seconds": round(fit_seconds, 3),
        "area_under_roc": float(auc),
        "area_under_pr": float(aupr),
        "confusion": confusion,
        "coefficients": coefficients,
        "preview": preview,
    }
    _write_dicts(REPORTS_DIR / "ml_preview.csv", preview)
    print(
        f"Logistic regression AUC {auc:.3f} on {sample_rows:,} sampled rows "
        f"(positive rate {result['positive_rate']:.3f})."
    )
    return result


def _coefficients(model, predictions) -> list[dict]:
    logistic = model.stages[-1]
    names = {}
    metadata = predictions.schema["features"].metadata.get("ml_attr", {}).get("attrs", {})
    for group in metadata.values():
        for attr in group:
            names[int(attr["idx"])] = attr["name"]
    weights = list(logistic.coefficients.toArray())
    ranked = sorted(
        (
            {"feature": names.get(index, f"f{index}"), "weight": float(weight)}
            for index, weight in enumerate(weights)
        ),
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )
    return ranked[:12]


def _metric_lines(detail) -> list[str]:
    if not isinstance(detail, dict):
        return []
    lines = []
    nodes = detail.get("nodes") or []
    for node in nodes:
        node_name = node.get("nodeName") or node.get("name") or "node"
        for metric in node.get("metrics") or []:
            name = metric.get("name")
            value = metric.get("value")
            if name:
                lines.append(f"{node_name}: {name} = {value}")
    return lines[:40]


def _write_dicts(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_month(metadata: dict) -> str:
    months = metadata.get("months") or []
    if "2024-06" in months:
        return "2024-06"
    if months:
        return months[len(months) // 2]
    return "2024-06"


def compare_totals(optimized: dict, naive: dict) -> dict:
    if not naive.get("ran"):
        return {"compared": False}
    order_gap = abs(optimized["orders"] - naive["orders"])
    net_gap = abs(optimized["net_revenue"] - naive["net_revenue"])
    net_base = max(abs(optimized["net_revenue"]), 1.0)
    return {
        "compared": True,
        "order_gap": order_gap,
        "net_gap": round(net_gap, 2),
        "net_relative_gap": net_gap / net_base,
        "match": order_gap == 0 and (net_gap / net_base) < 1e-6,
    }
