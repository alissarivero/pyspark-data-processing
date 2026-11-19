#!/usr/bin/env python3
"""Retail PySpark pipeline: lazy plans, joins, SQL, cache, and a classifier.

Run from the project root, after bootstrap.sh:

    .venv/bin/python pyspark_pipeline.py
    .venv/bin/python pyspark_pipeline.py --max-rows 250000   # logic check
    .venv/bin/python pyspark_pipeline.py --hold-ui           # leave the UI up
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retail_pipeline.generate import generate_dataset  # noqa: E402
from retail_pipeline.pipeline import (  # noqa: E402
    build_enriched,
    cache_demo,
    choose_month,
    compare_totals,
    lazy_versus_actions,
    load_tables,
    pruning_experiment,
    run_ml,
    run_naive,
    run_optimized,
    sql_on_monthly,
    sql_status_mix,
    udf_pushdown_experiment,
)
from retail_pipeline.publish import publish  # noqa: E402
from retail_pipeline.session import build_spark, spill_summary  # noqa: E402
from retail_pipeline.util import (  # noqa: E402
    OUTPUT_DIR,
    REPORTS_DIR,
    TARGET_BYTES,
    disk_free,
    ensure_dirs,
    fmt_bytes,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate retail Parquet and run the Spark pipeline.")
    parser.add_argument("--target-gb", type=float, default=TARGET_BYTES / 1024**3)
    parser.add_argument("--max-rows", type=int, default=None, help="Skip the 1 GB scale and write this many fact rows.")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--skip-naive", action="store_true")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--hold-ui", action="store_true", help="Keep the Spark UI up until reports/SPARK_UI_STOP appears.")
    return parser.parse_args()


def hold_ui(spark) -> None:
    ready = REPORTS_DIR / "SPARK_UI_READY"
    stop = REPORTS_DIR / "SPARK_UI_STOP"
    stop.unlink(missing_ok=True)
    ready.write_text((spark.sparkContext.uiWebUrl or "") + "\n")
    print(f"SPARK_UI_READY {spark.sparkContext.uiWebUrl}", flush=True)
    print("Waiting for reports/SPARK_UI_STOP (25 minutes max).", flush=True)
    deadline = time.time() + 25 * 60
    while time.time() < deadline and not stop.exists():
        time.sleep(1)
    ready.unlink(missing_ok=True)


def evaluate(metrics: dict, max_rows: int | None) -> list[str]:
    failures = []
    lazy = metrics["lazy"]
    if lazy["jobs_before"] is not None and lazy["jobs_after_transform"] != lazy["jobs_before"]:
        failures.append("transformations scheduled a Spark job")
    if (
        lazy["jobs_before"] is not None
        and lazy["jobs_after_action"] is not None
        and lazy["jobs_after_action"] <= lazy["jobs_before"]
    ):
        failures.append("count() did not schedule a Spark job")
    if metrics["pruning"]["month_rows"] != metrics["pruning"]["date_rows"]:
        failures.append("month partition filter and date filter returned different row counts")
    optimized = metrics["optimized"]
    if not optimized["pushed_filters"]:
        failures.append("optimized plan has no PushedFilters")
    if not optimized["partition_filters"]:
        failures.append("optimized plan has no PartitionFilters")
    if not optimized["broadcast"]:
        failures.append("optimized plan has no BroadcastHashJoin")
    if not (OUTPUT_DIR / "monthly_sales").exists():
        failures.append("monthly_sales Parquet was not written")
    comparison = metrics["comparison"]
    if comparison.get("compared") and not comparison.get("match"):
        failures.append(
            f"naive and optimized totals differ (orders {comparison['order_gap']}, net {comparison['net_gap']})"
        )
    if max_rows is None:
        size = metrics["dataset"]["transactions_bytes"]
        if size < 1_000_000_000:
            failures.append(f"transactions are {size} bytes, under 1 GB")
    if not metrics["cache"]["after_plan_uses_cache"]:
        failures.append("cached customer query did not read an in-memory plan")
    return failures


def main() -> int:
    args = parse_args()
    ensure_dirs()
    spark = build_spark()
    try:
        dataset = generate_dataset(
            spark,
            target_bytes=int(args.target_gb * 1024**3),
            max_rows=args.max_rows,
            force=args.force_generate,
        )
        transactions, products, customers = load_tables(spark)
        month = choose_month(dataset)
        print(f"\nDataset transactions {fmt_bytes(dataset['transactions_bytes'])}")
        lazy = lazy_versus_actions(spark, transactions, month)
        pruning = pruning_experiment(spark, transactions)
        udf = udf_pushdown_experiment(spark, transactions, month)
        free = disk_free()
        run_shuffle_join = not args.skip_naive and free > int(0.7 * 1024**3)
        if not run_shuffle_join:
            print(f"Skipping the timed shuffle join (free disk {fmt_bytes(free)}). The naive plan is still captured.")
        naive = run_naive(spark, transactions, products, customers, enabled=run_shuffle_join)
        enriched = build_enriched(transactions, products, customers)
        optimized = run_optimized(spark, enriched)
        monthly_sql = sql_on_monthly(spark)
        status_sql = sql_status_mix(spark, transactions, customers)
        cache = cache_demo(spark, enriched)
        ml = {"skipped": True} if args.skip_ml else run_ml(spark, transactions, products, customers)
        comparison = compare_totals(optimized, naive)
        metrics = {
            "spark_version": spark.version,
            "java_version": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
            "master": spark.sparkContext.master,
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "application_id": spark.sparkContext.applicationId,
            "ui": spark.sparkContext.uiWebUrl,
            "dataset": dataset,
            "lazy": {key: value for key, value in lazy.items() if key != "plan"},
            "pruning": pruning,
            "udf": udf,
            "naive": {key: value for key, value in naive.items() if key != "highlights"},
            "optimized": {key: value for key, value in optimized.items() if key != "formatted"},
            "monthly_sql": {"seconds": monthly_sql["seconds"], "top": monthly_sql["rows"][:8]},
            "status_sql": {"seconds": status_sql["seconds"], "rows": status_sql["rows"], "broadcast": status_sql["broadcast"]},
            "cache": {key: value for key, value in cache.items() if key != "segments"},
            "segments": cache["segments"],
            "ml": ml,
            "comparison": comparison,
            "spill": spill_summary(spark),
        }
        metrics["checks_failed"] = evaluate(metrics, args.max_rows)
        (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        publish(metrics)
        print("\nChecklist")
        for line in checklist(metrics):
            print(line)
        if metrics["checks_failed"]:
            print("FAILURES:")
            for item in metrics["checks_failed"]:
                print(f"  - {item}")
        else:
            print("All checks passed.")
            (REPORTS_DIR / "PIPELINE_OK").write_text("ok\n")
        if args.hold_ui:
            hold_ui(spark)
        return 1 if metrics["checks_failed"] else 0
    finally:
        spark.stop()


def checklist(metrics: dict) -> list[str]:
    dataset = metrics["dataset"]
    cache = metrics["cache"]
    lines = [
        f"Parquet transactions {fmt_bytes(dataset['transactions_bytes'])} partitioned by sale_month",
        f"Lazy build {metrics['lazy']['build_seconds']}s, count() action {metrics['lazy']['action_seconds']}s",
        f"Partition pruning count {metrics['pruning']['month_seconds']}s vs date filter {metrics['pruning']['date_seconds']}s",
        f"Optimized monthly write {metrics['optimized']['seconds']}s, broadcast={metrics['optimized']['broadcast']}",
        f"Naive shuffle join ran={metrics['naive'].get('ran')} seconds={metrics['naive'].get('seconds')}",
        f"SQL status mix {metrics['status_sql']['seconds']}s, monthly SQL {metrics['monthly_sql']['seconds']}s",
        f"Cache avg() {cache['uncached_avg_seconds']}s uncached vs {cache['cached_avg_seconds']}s cached",
        f"MLlib AUC {metrics['ml'].get('area_under_roc', 'skipped')}",
        f"Output {OUTPUT_DIR / 'monthly_sales'}",
    ]
    return lines


if __name__ == "__main__":
    sys.exit(main())
