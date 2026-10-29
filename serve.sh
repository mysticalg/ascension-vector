#!/bin/sh
# Serve this folder and open http://localhost:8000
cd "$(dirname "$0")"
echo "Open http://localhost:8000 in Chrome/Firefox (Ctrl+C to stop)"
python3 -m http.server 8000
