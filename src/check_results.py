#!/usr/bin/env python3
"""
Check Results & Update Bet History
===================================
Fetches race results from Racing Post and updates bet_history.csv with outcomes.
Designed to run as a cron job after races have finished.

Usage:
    python3 check_results.py                # Check today's results
    python3 check_results.py --date 2026-02-03  # Check specific date
"""

import requests
import os
import sys
import argparse
import time
import pandas as pd
from datetime import datetime, timedelta
from lxml import html

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def fetch_results_for_date(date_str: str) -> dict:
    """
    Fetch race results from Racing Post for a given date.
    Returns dict of {(course, race_id): [(position, horse_name), ...]}
    """
    import re as _re

    url = f'https://www.racingpost.com/results/{date_str}'
    results = {}

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  Failed to fetch results for {date_str} (status {resp.status_code})")
            return results

        # Extract all race result URLs from the page
        pattern = r'/results/(\d+)/([^/]+)/' + date_str.replace('-', '-') + r'/(\d+)'
        matches = _re.findall(pattern, resp.text)

        seen = set()
        race_urls = []
        for course_id, course_slug, race_id in matches:
            if race_id not in seen:
                seen.add(race_id)
                course = course_slug.replace('-', ' ').title()
                full_url = f'https://www.racingpost.com/results/{course_id}/{course_slug}/{date_str}/{race_id}'
                race_urls.append((course, race_id, full_url))

        print(f"  Found {len(race_urls)} races with results")

        for course, race_id, race_url in race_urls:
            race_results = fetch_race_result_html(race_url)
            if race_results:
                results[(course, race_id)] = race_results
            time.sleep(0.3)

    except Exception as e:
        print(f"  Error fetching results page: {e}")

    return results


def fetch_race_result_html(url: str) -> list:
    """
    Fetch result for a specific race by parsing the result page HTML.
    Returns [(position, horse_name), ...]
    """
    import re as _re

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        html_text = resp.text

        # Extract positions from data-test-selector="text-horsePosition"
        pos_pattern = r'data-test-selector="text-horsePosition"\s*>\s*(\d+|PU|F|UR|RR|BD|RO|SU)\s*<'
        positions = _re.findall(pos_pattern, html_text)

        # Extract horse names from data-test-selector="link-horseName"
        name_pattern = r'data-test-selector="link-horseName"\s*>\s*([^<]+?)\s*<'
        names = _re.findall(name_pattern, html_text)

        runners = []
        for i, name in enumerate(names):
            pos = positions[i] if i < len(positions) else '?'
            runners.append((pos, name.strip()))

        return runners

    except Exception as e:
        print(f"    Error fetching race result: {e}")
        return []


def update_bet_history(date_str: str = None):
    """Check results and update bet_history.csv."""
    history_path = os.path.join(DATA_DIR, 'bet_history.csv')

    if not os.path.exists(history_path):
        print("No bet history found.")
        return

    df = pd.read_csv(history_path)
    pending = df[df['result'] == 'Pending']

    if len(pending) == 0:
        print("No pending bets to check.")
        return

    # Filter by date if specified
    if date_str:
        pending = pending[pending['date'] == date_str]

    if len(pending) == 0:
        print(f"No pending bets for {date_str}.")
        return

    dates_to_check = pending['date'].unique()

    print(f"Checking results for {len(pending)} pending bets...")

    for check_date in dates_to_check:
        print(f"\n  Fetching results for {check_date}...")
        results = fetch_results_for_date(check_date)

        if not results:
            print(f"  No results available yet for {check_date}")
            continue

        date_bets = df[(df['date'] == check_date) & (df['result'] == 'Pending')]

        for idx, bet in date_bets.iterrows():
            horse = bet['horse']
            course = bet['course']
            bet_time = str(bet['time'])
            bet_type = bet['bet_type']
            stake = float(bet['stake'])
            decimal_odds = float(bet['decimal_odds'])

            # Skip doubles/accumulators - handle separately
            if bet_type in ('Double', 'Treble', 'Accumulator'):
                continue

            # Clean horse name for matching (remove region suffix like "(FR)")
            horse_clean = horse.split(' (')[0].strip().lower()

            # Find matching result
            found = False
            for (res_course, res_race_id), race_result in results.items():
                if not res_course:
                    continue

                # Match course (fuzzy)
                course_lower = course.lower().split('(')[0].strip()
                if course_lower not in res_course.lower() and res_course.lower() not in course_lower:
                    continue

                for pos, res_horse in race_result:
                    res_horse_clean = res_horse.split(' (')[0].strip().lower()

                    if horse_clean == res_horse_clean or horse_clean in res_horse_clean or res_horse_clean in horse_clean:
                        position = int(pos) if str(pos).isdigit() else 99
                        found = True

                        if bet_type == 'Win':
                            if position == 1:
                                returns = stake * decimal_odds
                                profit = returns - stake
                                df.loc[idx, 'result'] = 'Won'
                                df.loc[idx, 'position'] = position
                                df.loc[idx, 'returns'] = round(returns, 2)
                                df.loc[idx, 'profit'] = round(profit, 2)
                                print(f"    ✅ {horse} - WON! Position {position} - Returns £{returns:.2f}")
                            else:
                                df.loc[idx, 'result'] = 'Lost'
                                df.loc[idx, 'position'] = position
                                df.loc[idx, 'returns'] = 0
                                df.loc[idx, 'profit'] = -stake
                                print(f"    ❌ {horse} - Lost (Position {position})")

                        elif bet_type == 'Each Way':
                            # Each way: half stake on win, half on place
                            # Typical place terms: 1/4 odds for top 3 (or top 4 in handicaps)
                            win_stake = stake / 2
                            place_stake = stake / 2
                            place_odds = 1 + (decimal_odds - 1) / 4  # 1/4 odds place

                            if position == 1:
                                returns = (win_stake * decimal_odds) + (place_stake * place_odds)
                                profit = returns - stake
                                df.loc[idx, 'result'] = 'Won'
                                df.loc[idx, 'position'] = position
                                df.loc[idx, 'returns'] = round(returns, 2)
                                df.loc[idx, 'profit'] = round(profit, 2)
                                print(f"    ✅ {horse} - WON! E/W returns £{returns:.2f}")
                            elif position <= 4:  # Assume 4 places for now
                                returns = place_stake * place_odds
                                profit = returns - stake
                                df.loc[idx, 'result'] = 'Placed'
                                df.loc[idx, 'position'] = position
                                df.loc[idx, 'returns'] = round(returns, 2)
                                df.loc[idx, 'profit'] = round(profit, 2)
                                print(f"    🥉 {horse} - Placed ({position}th) - Place returns £{returns:.2f}")
                            else:
                                df.loc[idx, 'result'] = 'Lost'
                                df.loc[idx, 'position'] = position
                                df.loc[idx, 'returns'] = 0
                                df.loc[idx, 'profit'] = -stake
                                print(f"    ❌ {horse} - Lost (Position {position})")

                        break
                if found:
                    break

            if not found:
                print(f"    ⏳ {horse} - No result found yet")

        # Handle Doubles / Accumulators
        multi_bets = df[(df['date'] == check_date) & (df['result'] == 'Pending') &
                        (df['bet_type'].isin(['Double', 'Treble', 'Accumulator']))]

        for idx, bet in multi_bets.iterrows():
            horse_str = bet['horse']
            stake = float(bet['stake'])
            # Split legs: "Horse A + Horse B"
            legs = [h.strip() for h in horse_str.split('+')]

            all_won = True
            any_lost = False
            all_checked = True

            for leg in legs:
                leg_clean = leg.split('(')[0].strip().lower()
                leg_found = False
                leg_won = False

                for (res_course, _), race_result in results.items():
                    for pos, res_horse in race_result:
                        res_clean = res_horse.split('(')[0].strip().lower()
                        if leg_clean == res_clean or leg_clean in res_clean or res_clean in leg_clean:
                            leg_found = True
                            position = int(pos) if str(pos).isdigit() else 99
                            if position == 1:
                                leg_won = True
                            else:
                                any_lost = True
                                all_won = False
                            break
                    if leg_found:
                        break

                if not leg_found:
                    all_checked = False
                    all_won = False

            if any_lost:
                df.loc[idx, 'result'] = 'Lost'
                df.loc[idx, 'returns'] = 0
                df.loc[idx, 'profit'] = -stake
                print(f"    ❌ {horse_str} (Multi) - LOST")
            elif all_won and all_checked:
                returns = float(bet['potential_returns'])
                profit = returns - stake
                df.loc[idx, 'result'] = 'Won'
                df.loc[idx, 'returns'] = round(returns, 2)
                df.loc[idx, 'profit'] = round(profit, 2)
                print(f"    ✅ {horse_str} (Multi) - WON! Returns £{returns:.2f}")
            else:
                print(f"    ⏳ {horse_str} (Multi) - Waiting for remaining legs")

    # Save updated history
    df.to_csv(history_path, index=False)
    print(f"\nBet history updated: {history_path}")

    # Print summary
    settled = df[df['result'] != 'Pending']
    if len(settled) > 0:
        total_staked = settled['stake'].sum()
        total_returns = settled['returns'].sum()
        total_profit = settled['profit'].sum()
        wins = len(settled[settled['result'] == 'Won'])
        places = len(settled[settled['result'] == 'Placed'])
        losses = len(settled[settled['result'] == 'Lost'])

        print(f"\n{'='*40}")
        print(f"OVERALL PERFORMANCE")
        print(f"{'='*40}")
        print(f"  Bets settled: {len(settled)}")
        print(f"  Won: {wins} | Placed: {places} | Lost: {losses}")
        print(f"  Total staked:  £{total_staked:.2f}")
        print(f"  Total returns: £{total_returns:.2f}")
        print(f"  Profit/Loss:   £{total_profit:+.2f}")
        if total_staked > 0:
            roi = (total_profit / total_staked) * 100
            print(f"  ROI: {roi:+.1f}%")

    # Auto-push to GitHub
    print("\nPushing updated bet history to GitHub...")
    os.chdir(BASE_DIR)
    os.system('git add data/bet_history.csv')
    os.system(f'git commit -m "Auto-update bet results {datetime.now().strftime("%Y-%m-%d %H:%M")}"')
    os.system('git push')


def main():
    parser = argparse.ArgumentParser(description='Check race results and update bet history')
    parser.add_argument('--date', type=str, help='Check specific date (YYYY-MM-DD)')
    parser.add_argument('--no-push', action='store_true', help='Skip auto-push to GitHub')
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime('%Y-%m-%d')

    print(f"Race Results Checker")
    print(f"Checking results for: {date_str}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    update_bet_history(date_str)


if __name__ == '__main__':
    main()
