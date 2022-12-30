#!/bin/bash
# man:gunicorn
set -eu

exec gunicorn \
    --bind "0.0.0.0:${PORT:-8080}" \
    "index:server"
