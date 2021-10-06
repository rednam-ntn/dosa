#!/bin/sh

set -a
. ./.env
set +a

exec uvicorn --reload --host $HOST --port $PORT "$APP_MODULE"
