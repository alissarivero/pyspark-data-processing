"""Build a deterministic retail star schema and write it as Parquet.

Transactions are written in chunks. partitionBy("sale_month") shuffles each
chunk, so a single 1 GB write would hold the output and the shuffle at the
same time. Chunks keep that spike near 200 MB, which matters on a nearly full disk.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from pyspark.sql import functions as F

from retail_pipeline.constants import (
    AGE_BANDS,
    BRANDS,
    CATEGORIES,
    CHANNELS,
    DATE_ORIGIN,
    DATE_SPAN_DAYS,
    DEVICES,
    LOYALTY,
    MARKETPLACE_CHANNEL_INDEX,
    N_CUSTOMERS,
    N_PRODUCTS,
    N_STORES,
    PAYMENTS,
    PLANTED_SIGNALS,
    REGIONS,
    SEGMENTS,
    STORE_CHANNEL_INDEX,
    SUBCATEGORIES,
)
from retail_pipeline.util import (
    CUSTOMERS,
    DATA_DIR,
    DISK_FLOOR_BYTES,
    METADATA,
    PRODUCTS,
    TARGET_BYTES,
    TRANSACTIONS,
    dir_size,
    disk_free,
    fmt_bytes,
)

CHUNK_TARGET_BYTES = 200 * 1024 * 1024
MAX_ROWS = 40_000_000
PROBE_ROWS = 200_000


def pick(index_col, values: list[str]):
    return F.element_at(F.array(*[F.lit(value) for value in values]), index_col.cast("int") + 1)


def hash_hex(id_col, salt: int):
    raw = F.lower(F.hex(F.xxhash64(id_col, F.lit(int(salt)))))
    return F.substring(F.lpad(raw, 16, "0"), 1, 16)


def list_price(product_id):
    return F.round(F.lit(4.99) + F.pmod(product_id, F.lit(180)) * F.lit(1.25), 2)


def products_df(spark):
    frame = spark.range(0, N_PRODUCTS).withColumnRenamed("id", "product_id")
    product_id = F.col("product_id")
    return frame.select(
        product_id,
        pick(F.pmod(product_id, F.lit(len(CATEGORIES))), CATEGORIES).alias("category"),
        pick(F.pmod(product_id, F.lit(len(SUBCATEGORIES))), SUBCATEGORIES).alias("subcategory"),
        pick(F.pmod(product_id, F.lit(len(BRANDS))), BRANDS).alias("brand"),
        list_price(product_id).alias("list_price"),
        (F.pmod(product_id, F.lit(25)) != F.lit(0)).alias("is_active"),
    )


def customers_df(spark):
    frame = spark.range(0, N_CUSTOMERS).withColumnRenamed("id", "customer_id")
    customer_id = F.col("customer_id")
    return frame.select(
        customer_id,
        F.date_add(
            F.lit("2019-01-01").cast("date"),
            F.pmod(customer_id, F.lit(2000)).cast("int"),
        ).alias("signup_date"),
        pick(F.pmod(customer_id, F.lit(len(REGIONS))), REGIONS).alias("customer_region"),
        pick(F.pmod(F.xxhash64(customer_id, F.lit(3)), F.lit(len(SEGMENTS))), SEGMENTS).alias("segment"),
        pick(F.pmod(F.xxhash64(customer_id, F.lit(4)), F.lit(len(LOYALTY))), LOYALTY).alias("loyalty_tier"),
        pick(F.pmod(F.xxhash64(customer_id, F.lit(5)), F.lit(len(AGE_BANDS))), AGE_BANDS).alias("age_band"),
        (F.pmod(customer_id, F.lit(40)) != F.lit(0)).alias("is_active"),
    )


def transactions_df(spark, start: int, end: int):
    """Deterministic fact rows for transaction ids in [start, end)."""
    frame = spark.range(start, end).withColumnRenamed("id", "transaction_id")
    tid = F.col("transaction_id")
    customer_id = F.pmod(F.xxhash64(tid, F.lit(1)), F.lit(N_CUSTOMERS))
    product_id = F.pmod(F.xxhash64(tid, F.lit(2)), F.lit(N_PRODUCTS))
    channel_idx = F.pmod(F.xxhash64(tid, F.lit(3)), F.lit(len(CHANNELS)))
    disc_bucket = F.pmod(F.xxhash64(tid, F.lit(4)), F.lit(100))
    quantity = (F.pmod(F.xxhash64(tid, F.lit(5)), F.lit(5)) + F.lit(1)).cast("int")
    day_offset = F.pmod(F.xxhash64(tid, F.lit(7)), F.lit(DATE_SPAN_DAYS)).cast("int")
    roll = F.pmod(F.xxhash64(tid, F.lit(9)), F.lit(100)).cast("int")
    customer_region_idx = F.pmod(customer_id, F.lit(len(REGIONS)))
    fulfillment_region_idx = F.pmod(F.xxhash64(tid, F.lit(21)), F.lit(len(REGIONS)))
    store_region_idx = F.when(channel_idx == F.lit(STORE_CHANNEL_INDEX), customer_region_idx).otherwise(
        fulfillment_region_idx
    )
    risk = (
        F.when(channel_idx == F.lit(MARKETPLACE_CHANNEL_INDEX), F.lit(18)).otherwise(F.lit(4))
        + F.when(disc_bucket >= 98, F.lit(14)).when(disc_bucket >= 93, F.lit(7)).otherwise(F.lit(0))
        + F.when(quantity >= 4, F.lit(8)).otherwise(F.lit(0))
    ).cast("int")
    discount = (
        F.when(disc_bucket < 55, F.lit(0.0))
        .when(disc_bucket < 80, F.lit(0.05))
        .when(disc_bucket < 93, F.lit(0.10))
        .when(disc_bucket < 98, F.lit(0.20))
        .otherwise(F.lit(0.60))
    )
    status = (
        F.when(roll < risk, F.lit("CANCELLED"))
        .when(roll < risk + 5, F.lit("RETURNED"))
        .when(roll < risk + 7, F.lit("PENDING"))
        .otherwise(F.lit("COMPLETED"))
    )
    sale_date = F.date_add(F.lit(DATE_ORIGIN).cast("date"), day_offset)
    promo_n = F.pmod(F.xxhash64(tid, F.lit(16)), F.lit(20))
    return frame.select(
        tid.alias("transaction_id"),
        customer_id.alias("customer_id"),
        product_id.alias("product_id"),
        F.pmod(F.xxhash64(tid, F.lit(8)), F.lit(N_STORES)).cast("int").alias("store_id"),
        sale_date.alias("sale_date"),
        F.date_format(sale_date, "yyyy-MM").alias("sale_month"),
        quantity.alias("quantity"),
        list_price(product_id).alias("unit_price"),
        discount.alias("discount_pct"),
        pick(channel_idx, CHANNELS).alias("channel"),
        pick(F.pmod(F.xxhash64(tid, F.lit(14)), F.lit(len(PAYMENTS))), PAYMENTS).alias("payment_method"),
        status.alias("status"),
        pick(store_region_idx, REGIONS).alias("store_region"),
        pick(F.pmod(F.xxhash64(tid, F.lit(13)), F.lit(len(DEVICES))), DEVICES).alias("device_type"),
        F.when(
            F.pmod(F.xxhash64(tid, F.lit(15)), F.lit(10)) == F.lit(0),
            F.concat(F.lit("PROMO"), promo_n.cast("string")),
        ).alias("promo_code"),
        F.concat(hash_hex(tid, 11), hash_hex(tid, 12)).alias("receipt_token"),
        F.concat(hash_hex(tid, 17), hash_hex(tid, 18)).alias("payment_ref"),
    )


def write_transactions(df, path, overwrite: bool) -> None:
    (
        df.repartition("sale_month")
        .write.mode("overwrite" if overwrite else "append")
        .partitionBy("sale_month")
        .parquet(str(path))
    )


def _months(path) -> list[str]:
    if not path.exists():
        return []
    months = []
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("sale_month="):
            months.append(child.name.split("=", 1)[1])
    return sorted(months)


def generate_dataset(spark, target_bytes: int | None = None, max_rows: int | None = None, force: bool = False) -> dict:
    target_bytes = TARGET_BYTES if target_bytes is None else int(target_bytes)
    spark.sparkContext.setJobDescription("generate-dimensions")
    previous_shuffle = spark.conf.get("spark.sql.shuffle.partitions")
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    try:
        return _generate(spark, target_bytes, max_rows, force)
    finally:
        spark.conf.set("spark.sql.shuffle.partitions", previous_shuffle)


def _existing_is_enough(target_bytes: int, max_rows: int | None, force: bool) -> dict | None:
    if force or max_rows is not None or not METADATA.exists() or not TRANSACTIONS.exists():
        return None
    meta = json.loads(METADATA.read_text())
    have = int(meta.get("transactions_bytes") or 0)
    if have >= int(target_bytes * 0.98) and have == dir_size(TRANSACTIONS):
        print(f"Reusing transactions at {fmt_bytes(have)}")
        return meta
    return None


def _generate(spark, target_bytes: int, max_rows: int | None, force: bool) -> dict:
    cached = _existing_is_enough(target_bytes, max_rows, force)
    if cached is not None:
        return cached

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    free = disk_free()
    print(f"Free disk before generate: {fmt_bytes(free)}")
    limited = False
    if max_rows is None:
        # Last chunk still needs room for its own shuffle beside the files already written.
        usable = free - DISK_FLOOR_BYTES - CHUNK_TARGET_BYTES
        if usable < target_bytes:
            print(
                f"Lowering the dataset target from {fmt_bytes(target_bytes)} to {fmt_bytes(max(usable, 0))} "
                "so the last write cannot fill the disk."
            )
            target_bytes = max(usable, 64 * 1024 * 1024)
            limited = True

    if max_rows is None:
        probe_path = DATA_DIR / "_probe"
        if probe_path.exists():
            shutil.rmtree(probe_path)
        spark.sparkContext.setJobDescription("generate-size-probe")
        write_transactions(transactions_df(spark, 0, PROBE_ROWS), probe_path, overwrite=True)
        probe_bytes = dir_size(probe_path)
        shutil.rmtree(probe_path)
        bytes_per_row = probe_bytes / PROBE_ROWS
        rows = min(MAX_ROWS, max(PROBE_ROWS, int(target_bytes / bytes_per_row * 1.08)))
        print(f"Probe {PROBE_ROWS:,} rows -> {fmt_bytes(probe_bytes)} ({bytes_per_row:.1f} bytes/row)")
        print(f"Planning {rows:,} transaction rows for about {fmt_bytes(target_bytes)}")
    else:
        rows = int(max_rows)
        bytes_per_row = None
        print(f"Smoke generation: {rows:,} transaction rows")

    if TRANSACTIONS.exists():
        shutil.rmtree(TRANSACTIONS)
    rows_per_chunk = rows if max_rows is not None else max(1, int(rows * CHUNK_TARGET_BYTES / max(target_bytes, 1)))
    if max_rows is not None:
        rows_per_chunk = rows

    written = 0
    chunk_index = 0
    while written < rows:
        if disk_free() < DISK_FLOOR_BYTES + CHUNK_TARGET_BYTES:
            print(f"Stopping generate with {written:,} rows; free disk is {fmt_bytes(disk_free())}")
            limited = True
            break
        end = min(rows, written + rows_per_chunk)
        spark.sparkContext.setJobDescription(f"generate-transactions-{chunk_index}")
        write_transactions(transactions_df(spark, written, end), TRANSACTIONS, overwrite=(written == 0))
        written = end
        chunk_index += 1
        print(
            f"Chunk {chunk_index}: rows {written:,}/{rows:,}  "
            f"on disk {fmt_bytes(dir_size(TRANSACTIONS))}  free {fmt_bytes(disk_free())}"
        )

    size = dir_size(TRANSACTIONS)
    if max_rows is None and not limited and size < target_bytes * 0.98 and written < MAX_ROWS:
        actual_bpr = size / written if written else 1
        extra = min(MAX_ROWS - written, int((target_bytes - size) / actual_bpr * 1.05) + 1)
        if extra > 0 and disk_free() > DISK_FLOOR_BYTES + CHUNK_TARGET_BYTES:
            print(f"Appending {extra:,} rows to close a {fmt_bytes(target_bytes - size)} gap")
            spark.sparkContext.setJobDescription("generate-transactions-topup")
            write_transactions(transactions_df(spark, written, written + extra), TRANSACTIONS, overwrite=False)
            written += extra
            size = dir_size(TRANSACTIONS)
            bytes_per_row = size / written

    spark.sparkContext.setJobDescription("generate-dimensions")
    if CUSTOMERS.exists():
        shutil.rmtree(CUSTOMERS)
    if PRODUCTS.exists():
        shutil.rmtree(PRODUCTS)
    customers_df(spark).write.mode("overwrite").parquet(str(CUSTOMERS))
    products_df(spark).write.mode("overwrite").parquet(str(PRODUCTS))

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transactions_rows": written,
        "transactions_bytes": dir_size(TRANSACTIONS),
        "customers_rows": N_CUSTOMERS,
        "customers_bytes": dir_size(CUSTOMERS),
        "products_rows": N_PRODUCTS,
        "products_bytes": dir_size(PRODUCTS),
        "bytes_per_row": (dir_size(TRANSACTIONS) / written) if written else None,
        "probe_bytes_per_row": bytes_per_row,
        "months": _months(TRANSACTIONS),
        "target_bytes": target_bytes,
        "limited_by_disk": limited,
        "partitioned_by": "sale_month",
        "planted_signals": PLANTED_SIGNALS,
        "total_bytes": dir_size(TRANSACTIONS) + dir_size(CUSTOMERS) + dir_size(PRODUCTS),
    }
    METADATA.write_text(json.dumps(meta, indent=2))
    print(f"Transactions {fmt_bytes(meta['transactions_bytes'])} across {len(meta['months'])} months")
    print(f"Total dataset {fmt_bytes(meta['total_bytes'])}")
    return meta
