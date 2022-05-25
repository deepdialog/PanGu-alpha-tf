#!/bin/bash

# uvicorn main:app --host 0.0.0.0 --port 8000

gunicorn \
    -k uvicorn.workers.UvicornWorker  \
    --bind "0.0.0.0:8000" \
    --workers 2 \
    --timeout 240 \
    --graceful-timeout 240 \
    --max-requests 0 \
    --log-level debug \
    main:app


