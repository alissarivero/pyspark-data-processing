"""SparkSession construction and plan/UI helpers.

Driver memory is set through the environment before the JVM starts. Setting
spark.driver.memory on the builder is too late: in local mode the JVM is
already up by the time that config is read.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from retail_pipeline.util import REPORTS_DIR, ROOT, ensure_dirs, runtime_root

JAVA_CANDIDATES = [
    ROOT / ".tools" / "jdk",
    Path("/Applications/PyCharm.app/Contents/jbr/Contents/Home"),
]


def configure_environment() -> str:
    ensure_dirs()
    java_home = os.environ.get("JAVA_HOME", "")
    if not java_home or not Path(java_home, "bin/java").exists():
        for candidate in JAVA_CANDIDATES:
            if (candidate / "bin" / "java").exists():
                java_home = str(candidate)
                break
    if not java_home or not Path(java_home, "bin/java").exists():
        raise SystemExit("JDK 17+ not found. Set JAVA_HOME and re-run.")

    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["SPARK_DRIVER_MEMORY"] = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
    if "PYSPARK_SUBMIT_ARGS" not in os.environ:
        memory = os.environ["SPARK_DRIVER_MEMORY"]
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--driver-memory {memory} pyspark-shell"
    return java_home


def build_spark(app_name: str = "retail-pipeline"):
    from pyspark.sql import SparkSession

    java_home = configure_environment()
    runtime = runtime_root()
    event_dir = runtime / "spark-events"
    local_dir = runtime / "spark-local"
    event_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    # The launcher ignores spark.local.dir unless SPARK_LOCAL_DIRS is set first.
    os.environ["SPARK_LOCAL_DIRS"] = str(local_dir)

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.ui.port", "4040")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "32")
        # 10 MB is Spark's default. Customer rows can exceed it, so the
        # optimized pipeline broadcasts explicitly instead of hoping the
        # threshold catches the dimension table.
        .config("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.debug.maxToStringFields", "200")
        .config("spark.sql.files.maxPartitionBytes", str(16 * 1024 * 1024))
        .config("spark.sql.files.openCostInBytes", str(4 * 1024 * 1024))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.local.dir", str(local_dir))
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.compress", "true")
        .config("spark.eventLog.dir", event_dir.as_uri())
        .config("spark.ui.retainedJobs", "200")
        .config("spark.ui.retainedStages", "200")
        .config("spark.sql.ui.retainedExecutions", "80")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.execution.pythonUDF.arrow.enabled", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    java_version = spark.sparkContext._jvm.java.lang.System.getProperty("java.version")
    print(f"Spark {spark.version}  Java {java_version}")
    print(f"JAVA_HOME {java_home}")
    print(f"UI {spark.sparkContext.uiWebUrl}")
    print(f"master {spark.sparkContext.master}  defaultParallelism {spark.sparkContext.defaultParallelism}")
    return spark


def explain_string(df, mode: str = "extended") -> str:
    jvm = df.sparkSession.sparkContext._jvm
    query = df._jdf.queryExecution()
    try:
        return jvm.PythonSQLUtils.explainString(query, mode)
    except Exception as first_error:
        try:
            mode_enum = jvm.org.apache.spark.sql.execution.ExplainMode.fromString(mode)
            return query.explainString(mode_enum)
        except Exception:
            return f"(explain {mode} failed: {first_error})\n{query}"


def rest(spark, path: str, retries: int = 6):
    ui = spark.sparkContext.uiWebUrl
    if not ui:
        return None
    app_id = spark.sparkContext.applicationId
    url = f"{ui.rstrip('/')}/api/v1/applications/{app_id}{path}"
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                return json.load(response)
        except Exception as exc:  # UI can lag the first call
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    print(f"Spark REST {path} unavailable: {last_error}")
    return None


def job_count(spark) -> int | None:
    payload = rest(spark, "/jobs")
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        jobs = payload.get("jobs")
        if isinstance(jobs, list):
            return len(jobs)
    return None


def snapshot_sql(spark, label: str) -> dict | None:
    executions = rest(spark, "/sql")
    if isinstance(executions, dict):
        executions = executions.get("sql") or executions.get("executions") or []
    if not isinstance(executions, list) or not executions:
        return None
    latest = max(executions, key=lambda item: item.get("id", -1))
    detail = rest(spark, f"/sql/{latest.get('id')}") or latest
    path = REPORTS_DIR / "sql" / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detail, indent=2, default=str))
    return detail if isinstance(detail, dict) else {"raw": detail}


def spill_summary(spark) -> dict:
    stages = rest(spark, "/stages") or []
    if isinstance(stages, dict):
        stages = stages.get("stages") or []
    if not isinstance(stages, list):
        return {}
    disk = sum(int(stage.get("diskBytesSpilled") or 0) for stage in stages)
    memory = sum(int(stage.get("memoryBytesSpilled") or 0) for stage in stages)
    return {
        "stages": len(stages),
        "disk_bytes_spilled": disk,
        "memory_bytes_spilled": memory,
    }


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def highlight_plan(plan: str, limit: int = 80) -> str:
    keys = (
        "PushedFilters",
        "PartitionFilters",
        "ReadSchema",
        "BroadcastHashJoin",
        "SortMergeJoin",
        "BroadcastExchange",
        "Exchange",
        "FileScan",
        "HashAggregate",
        "AdaptiveSparkPlan",
        "InMemory",
        "Filter",
        "Project",
        "ColumnarToRow",
    )
    lines = [line.rstrip() for line in plan.splitlines() if any(key in line for key in keys)]
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more matching lines"]
    return "\n".join(lines)
