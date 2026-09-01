#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FDDI_ASSET_ROOT="${FDDI_ASSET_ROOT:-${PROJECT_ROOT}/../FDDI_assets}"

if [[ ! -d "${FDDI_ASSET_ROOT}" ]]; then
    echo "FDDI assets not found: ${FDDI_ASSET_ROOT}" >&2
    echo "Set FDDI_ASSET_ROOT to the directory containing models/, datasets/, and results/." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
exec python app.py "$@"
