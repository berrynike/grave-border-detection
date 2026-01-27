#!/bin/bash
# Run HPO study with caffeinate to prevent sleep during long runs
#
# Usage:
#   ./scripts/run_hpo.sh              # Run full HPO (50 trials)
#   ./scripts/run_hpo.sh +hpo=pilot   # Run pilot HPO (20 trials)
#
# To resume an interrupted study, just run the same command again.
# Studies are stored in SQLite and will pick up where they left off.

set -e

cd "$(dirname "$0")/.."

echo "Starting HPO study..."
echo "To view progress: ./scripts/view_hpo.sh"
echo ""

# Use caffeinate to prevent sleep during training
caffeinate -i uv run python -m grave_border_detection.hpo "$@"
