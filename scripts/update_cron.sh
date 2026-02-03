#!/bin/bash
# Horse Racing Race Card Auto-Updater
# Runs every 6 hours to fetch fresh race cards from Racing Post
# and pushes to GitHub so Streamlit Cloud auto-redeploys

cd /Users/lewisjackson/projects/horse-racing-predictor

# Log output
LOG_FILE="data/cron_log.txt"
echo "=== Update started: $(date) ===" >> "$LOG_FILE"

# Run the updater (fetch today + tomorrow, auto-push to GitHub)
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/update_races.py --days 2 >> "$LOG_FILE" 2>&1

echo "=== Update finished: $(date) ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
