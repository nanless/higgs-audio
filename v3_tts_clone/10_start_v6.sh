#!/bin/bash
# Start the independent Higgs Audio V3 V6 half-hour top-up pipeline.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_ENV="${HERE}/10_v6_topup_pipeline.env"

exec bash "${HERE}/07_topup_pipeline.sh"
