#!/bin/sh
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run python install.py first."; exit 1; }
exec .venv/bin/python -m ftstudio
