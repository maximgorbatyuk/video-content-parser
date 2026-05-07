#!/usr/bin/env bash
# Installs the Python client deps for analyze_videos.py.
# The model itself is downloaded separately and served by llama-server -- see README.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYBIN="${PYBIN:-python3}"
PYVER=$("$PYBIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PYVER"
if [[ "$PYVER" < "3.10" ]]; then
    echo "ERROR: need Python 3.10+. Try 'brew install python@3.11'." >&2
    exit 1
fi

if [[ ! -d venv ]]; then
    "$PYBIN" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip wheel
pip install --upgrade "av>=12" "pillow" "requests" "python-dotenv" "rich"

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo
    echo "Created .env from .env.example — edit it to set SOURCE_FOLDER, OUTPUT_FOLDER, LLM_URL."
fi

echo
echo "Done. Next steps:"
echo "  1. Edit $SCRIPT_DIR/.env"
echo "  2. Start your local LLM (llama-server / LM Studio)."
echo "  3. source $SCRIPT_DIR/venv/bin/activate"
echo "  4. python $SCRIPT_DIR/analyze_videos.py"
