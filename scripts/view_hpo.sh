#!/bin/bash
# Launch Optuna Dashboard to view HPO results
#
# Usage:
#   ./scripts/view_hpo.sh              # View default study (hpo_studies.db)
#   ./scripts/view_hpo.sh pilot        # View pilot study (hpo_pilot.db)
#
# Dashboard will be available at http://localhost:8080

set -e

cd "$(dirname "$0")/.."

if [ "$1" = "pilot" ]; then
    DB_PATH="hpo_pilot.db"
else
    DB_PATH="hpo_studies.db"
fi

if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database file '$DB_PATH' not found."
    echo "Run an HPO study first: ./scripts/run_hpo.sh"
    exit 1
fi

echo "Opening Optuna Dashboard for $DB_PATH..."
echo "Dashboard: http://localhost:8080"
echo ""

uv run optuna-dashboard "sqlite:///$DB_PATH"
