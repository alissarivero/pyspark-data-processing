#!/usr/bin/env bash
# Create the local virtualenv and confirm a JDK 17+ is available.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --no-cache-dir -r requirements.txt

if [[ -z "${JAVA_HOME:-}" ]]; then
  if [[ -x "/Applications/PyCharm.app/Contents/jbr/Contents/Home/bin/java" ]]; then
    export JAVA_HOME="/Applications/PyCharm.app/Contents/jbr/Contents/Home"
  elif [[ -x ".tools/jdk/bin/java" ]]; then
    export JAVA_HOME="$(cd .tools/jdk && pwd)"
  else
    echo "Set JAVA_HOME to a JDK 17 or newer, then re-run." >&2
    exit 1
  fi
fi

echo "JAVA_HOME=$JAVA_HOME"
"$JAVA_HOME/bin/java" -version
ln -sfn "$PWD" /tmp/retail-pipeline
echo "Ready: JAVA_HOME=\"$JAVA_HOME\" .venv/bin/python pyspark_pipeline.py"
