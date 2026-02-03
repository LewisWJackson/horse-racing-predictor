#!/bin/bash
# Horse Racing Results Checker
# Runs at 21:30 daily to check race results and update bet history
# Then pushes to GitHub so Streamlit Cloud auto-redeploys

cd /Users/lewisjackson/projects/horse-racing-predictor

# Log output
LOG_FILE="data/cron_log.txt"
echo "=== Results check started: $(date) ===" >> "$LOG_FILE"

# Check results for today and update bet_history.csv
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/check_results.py >> "$LOG_FILE" 2>&1

echo "=== Results check finished: $(date) ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
