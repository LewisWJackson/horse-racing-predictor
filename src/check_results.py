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
    Returns dict of {(course, time): [(position, horse_name), ...]}
    """
    url = f'https://www.racingpost.com/results/{date_str}'
    results = {}

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  Failed to fetch results for {date_str} (status {resp.status_code})")
            return results

        doc = html.fromstring(resp.content)

        for meeting in doc.xpath('//section[@data-accordion-row]'):
            course_el = meeting.xpath(".//span[contains(@class, 'RC-accordion__courseName')]")
            if not course_el:
                continue
            course = ' '.join(course_el[0].text_content().strip().split())

            for race_link in meeting.xpath(".//a[contains(@class, 'RC-meetingItem__link')]"):
                race_id = race_link.attrib.get('data-race-id', '')
                time_el = race_link.xpath(".//span[contains(@class, 'RC-meetingItem__timeText')]")
                race_time = time_el[0].text_content().strip() if time_el else ''

                if race_id:
                    # Fetch individual race result
                    race_results = fetch_race_result(race_id)
                    if race_results:
                        results[(course, race_time)] = race_results

    except Exception as e:
        print(f"  Error fetching results page: {e}")

    return results


def fetch_race_result(race_id: str) -> list:
    """
    Fetch result for a specific race.
    Returns [(position, horse_name), ...]
    """
    url = f'https://www.racingpost.com/results/data/{race_id}'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            runners = []
            for runner in data.get('runners', []):
                pos = runner.get('position', '')
                name = (runner.get('horseName') or '').strip()
                if name:
                    runners.append((pos, name))
            return runners
    except Exception:
        pass

    # Fallback: try the card runners endpoint
    url2 = f'https://www.racingpost.com/profile/horse/data/cardrunners/{race_id}.json'
    try:
        resp = requests.get(url2, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            runners_map = data.get('runners', {})
            runners = []
            for r in runners_map.values():
                pos = r.get('raceOutcomePosition', r.get('position', ''))
                name = (r.get('horseName') or '').strip()
                if name and pos:
                    runners.append((pos, name))
            return sorted(runners, key=lambda x: int(x[0]) if str(x[0]).isdigit() else 99)
    except Exception:
        pass

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

            # Clean horse name for matching (remove region suffix)
            horse_clean = horse.split(' (')[0].strip().lower()

            # Find matching result
            found = False
            for (res_course, res_time), race_result in results.items():
                if not res_course or not res_time:
                    continue

                # Match course (fuzzy)
                if course.lower() not in res_course.lower() and res_course.lower() not in course.lower():
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
