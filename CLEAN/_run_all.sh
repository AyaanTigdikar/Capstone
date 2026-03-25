#!/bin/bash
set -e
cd /Users/leoss/Desktop/GitHub/Capstone/CLEAN

JUPYTER="/Users/leoss/Library/Python/3.10/bin/jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=3600"

run_nb() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "▶  $1"
    echo "══════════════════════════════════════════"
    $JUPYTER "$1" 2>&1
    echo "✓  Done: $1"
}

echo "Started: $(date)"

# NB4-NB6 completed — resuming from NB7
run_nb "7_Viz_Descriptive_Clustering.ipynb"
run_nb "8_Viz_ML.ipynb"
run_nb "9_Viz_Regression.ipynb"
run_nb "10_Robustness_Regression.ipynb"
run_nb "11_Robustness_ML.ipynb"
run_nb "13_Appendix_B_Charts.ipynb"

echo ""
echo "══════════════════════════════════════════"
echo "ALL DONE: $(date)"
echo "Charts saved to: outputs/charts/"
ls outputs/charts/*.png 2>/dev/null | wc -l | xargs echo "PNG count:"
