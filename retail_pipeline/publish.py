"""Write an HTML report and a notebook that embed the run's actual numbers."""

from __future__ import annotations

import html
import json
from pathlib import Path

from retail_pipeline.util import REPORTS_DIR, ROOT, fmt_bytes


def _pre(text: str) -> str:
    return f"<pre>{html.escape(text or '')}</pre>"


def _table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    headers = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def publish(metrics: dict) -> None:
    analysis_path = REPORTS_DIR / "performance_analysis.md"
    analysis = analysis_path.read_text() if analysis_path.exists() else ""
    highlights = (REPORTS_DIR / "explain_optimized_highlights.txt").read_text() if (
        REPORTS_DIR / "explain_optimized_highlights.txt"
    ).exists() else ""
    naive = (REPORTS_DIR / "explain_naive_highlights.txt").read_text() if (
        REPORTS_DIR / "explain_naive_highlights.txt"
    ).exists() else ""
    dataset = metrics["dataset"]
    shots = [
        ("Spark jobs", "reports/screenshots/01_jobs_success.png"),
        ("SQL details", "reports/screenshots/02_sql_query_details.png"),
        ("Physical plan", "reports/screenshots/03_explain_plan.png"),
    ]
    images = []
    for title, relative in shots:
        path = ROOT / relative
        if path.exists():
            images.append(
                f"<figure><img src='{relative}' alt='{html.escape(title)}'><figcaption>{html.escape(title)}</figcaption></figure>"
            )
        else:
            images.append(f"<p>Screenshot pending: {html.escape(relative)}</p>")
    paragraphs = [part.strip() for part in analysis.split("\n\n") if part.strip()]
    analysis_html = (
        "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)
        or "<p>Analysis is added after the run.</p>"
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Retail PySpark pipeline</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 980px; line-height: 1.45; color: #1c1c1c; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
pre {{ background: #f4f1ea; padding: 1rem; overflow: auto; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
h1, h2 {{ font-weight: 600; }}
</style>
</head>
<body>
<h1>Retail PySpark pipeline</h1>
<p>Spark {html.escape(str(metrics.get('spark_version')))} on {html.escape(str(metrics.get('master')))},
Java {html.escape(str(metrics.get('java_version')))}, application {html.escape(str(metrics.get('application_id')))}.</p>
<h2>Dataset</h2>
<ul>
<li>Transactions: {html.escape(fmt_bytes(dataset['transactions_bytes']))}, {dataset['transactions_rows']:,} rows, partitioned by sale_month</li>
<li>Customers: {dataset['customers_rows']:,} rows ({html.escape(fmt_bytes(dataset['customers_bytes']))})</li>
<li>Products: {dataset['products_rows']:,} rows ({html.escape(fmt_bytes(dataset['products_bytes']))})</li>
<li>Total: {html.escape(fmt_bytes(dataset['total_bytes']))}</li>
</ul>
<h2>Timings</h2>
{_table([
    {"step": "lazy transformations", "seconds": metrics["lazy"]["build_seconds"]},
    {"step": "lazy count action", "seconds": metrics["lazy"]["action_seconds"]},
    {"step": "count filtered by sale_month", "seconds": metrics["pruning"]["month_seconds"]},
    {"step": "count filtered by sale_date", "seconds": metrics["pruning"]["date_seconds"]},
    {"step": "native status filter, one month", "seconds": metrics["udf"]["native_seconds"]},
    {"step": "Python UDF status filter, one month", "seconds": metrics["udf"]["blocked_seconds"]},
    {"step": "naive shuffle join", "seconds": metrics["naive"].get("seconds")},
    {"step": "optimized monthly write", "seconds": metrics["optimized"]["seconds"]},
    {"step": "SQL status by channel", "seconds": metrics["status_sql"]["seconds"]},
    {"step": "customer avg, uncached", "seconds": metrics["cache"]["uncached_avg_seconds"]},
    {"step": "customer avg, cached", "seconds": metrics["cache"]["cached_avg_seconds"]},
    {"step": "MLlib fit", "seconds": metrics["ml"].get("fit_seconds")},
])}
<h2>Performance analysis</h2>
{analysis_html}
<h2>Optimized plan highlights</h2>
{_pre(highlights)}
<h2>Naive plan highlights</h2>
{_pre(naive)}
<h2>2024 category and region leaders</h2>
{_table(metrics["monthly_sql"]["top"])}
<h2>Segment summary</h2>
{_table(metrics.get("segments") or [])}
<h2>Model</h2>
<p>AUC {html.escape(str(metrics["ml"].get("area_under_roc")))},
PR AUC {html.escape(str(metrics["ml"].get("area_under_pr")))},
sample {html.escape(str(metrics["ml"].get("sample_rows")))}.</p>
{_table(metrics["ml"].get("coefficients") or [])}
<h2>Screenshots</h2>
{''.join(images)}
</body>
</html>
"""
    (ROOT / "pyspark_pipeline.html").write_text(page)
    _write_notebook(metrics, highlights, analysis)
    print(f"Wrote {ROOT / 'pyspark_pipeline.html'} and pyspark_pipeline.ipynb")


def _write_notebook(metrics: dict, highlights: str, analysis: str) -> None:
    def md(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}

    def code(source: str, output: str, execution_count: int) -> dict:
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": execution_count,
            "source": _lines(source),
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": _lines(output),
                }
            ],
        }

    dataset = metrics["dataset"]
    lazy = metrics["lazy"]
    cells = [
        md(
            "# Retail PySpark pipeline\n\n"
            "Star schema in Parquet: transactions partitioned by `sale_month`, plus customer and product dimensions. "
            "The executable source is `pyspark_pipeline.py`. This notebook records the run.\n"
        ),
        code(
            "from retail_pipeline.session import build_spark\nspark = build_spark()\nprint(spark.version)",
            f"Spark {metrics.get('spark_version')}  Java {metrics.get('java_version')}\nUI {metrics.get('ui')}\n",
            1,
        ),
        md(
            "## Dataset\n\n"
            f"Transactions: **{fmt_bytes(dataset['transactions_bytes'])}**, {dataset['transactions_rows']:,} rows, "
            f"{len(dataset.get('months') or [])} month partitions.\n\n"
            "High-entropy `receipt_token` and `payment_ref` columns make the Parquet files large. "
            "The revenue pipeline never reads them.\n"
        ),
        code(
            "lazy = (\n"
            "    transactions.filter(sale_month == month)\n"
            "        .filter(status == 'COMPLETED')\n"
            "        .withColumn('gross_revenue', round(quantity * unit_price, 2))\n"
            ")\nlazy.count()  # this is the action\n",
            (
                f"transformations {lazy['build_seconds']}s, jobs {lazy['jobs_before']} -> {lazy['jobs_after_transform']}\n"
                f"explain jobs {lazy['jobs_after_explain']}\n"
                f"count() {lazy['action_seconds']}s, rows {lazy['rows']}, jobs {lazy['jobs_after_action']}\n"
            ),
            2,
        ),
        md("## Optimized physical plan\n\nFilters on `sale_month` are partition filters. Status, discount, and quantity are pushed into the Parquet scan. Products and customers are broadcast.\n"),
        code("monthly.explain(mode='extended')", highlights + "\n", 3),
        md("## Cache\n\nThe customer rollup shuffles. The second action repeats that shuffle unless the rollup is cached.\n"),
        code(
            "customer_value.persist()\ncustomer_value.agg(avg('lifetime_net')).collect()",
            (
                f"uncached avg {metrics['cache']['uncached_avg_seconds']}s\n"
                f"materialize {metrics['cache']['materialize_seconds']}s\n"
                f"cached avg {metrics['cache']['cached_avg_seconds']}s\n"
                f"in-memory plan after cache: {metrics['cache']['after_plan_uses_cache']}\n"
            ),
            4,
        ),
        md("## Performance analysis\n\n" + (analysis or "Filled in after the measured run.") + "\n"),
        md("## Model\n\nLogistic regression on a 2.5% sample predicts cancellation from channel, category, segment, loyalty, quantity, price, and discount.\n"),
        code(
            "BinaryClassificationEvaluator(metricName='areaUnderROC').evaluate(predictions)",
            f"AUC {metrics['ml'].get('area_under_roc')}\nPR AUC {metrics['ml'].get('area_under_pr')}\n",
            5,
        ),
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    (ROOT / "pyspark_pipeline.ipynb").write_text(json.dumps(notebook, indent=2))


def _lines(text: str) -> list[str]:
    if text == "":
        return []
    if not text.endswith("\n"):
        text += "\n"
    return [line + "\n" for line in text.split("\n")[:-1]]
