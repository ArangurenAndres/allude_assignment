#!/usr/bin/env bash
# setup.sh — setup environment and run analytical web app

set -e  # Stop on first error

PYTHON_BIN="python3"
VENV_DIR=".venv"

echo ">>> Creating virtual environment in ${VENV_DIR} ..."

if [ ! -d "${VENV_DIR}" ]; then
    ${PYTHON_BIN} -m venv ${VENV_DIR}
else
    echo ">>> Virtual environment already exists."
fi

echo ">>> Activating environment ..."
# shellcheck disable=SC1091
source ${VENV_DIR}/bin/activate

echo ">>> Upgrading pip ..."
pip install --upgrade pip

echo ">>> Installing dependencies ..."
pip install -r requirements.txt

echo ">>> Disabling LLM (Analytical mode only) ..."
export USE_LLM=0

echo ">>> Running web application ..."
python main.py web
