#!/usr/bin/env python3
"""
Auto-Update Races
=================
Fetches real upcoming race cards from Racing Post for the next N days.
Designed to run as a cron job so the dashboard always has fresh data.

Usage:
    python3 update_races.py           # Fetch today + tomorrow
    python3 update_races.py --days 3  # Fetch next 3 days
"""

import requests
import os
import sys
import pickle
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from lxml import html

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.9',
}

RACE_TYPE = {'F': 'Flat', 'X': 'Flat', 'C': 'Chase', 'U': 'Chase', 'H': 'Hurdle', 'B': 'NH Flat', 'W': 'NH Flat'}


def load_elo_ratings():
    elo_path = os.path.join(MODELS_DIR, 'elo_ratings.pkl')
    try:
        with open(elo_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {'horse_elo': {}, 'jockey_elo': {}, 'trainer_elo': {}}


def fetch_race_ids_for_date(date_str: str) -> list:
    """Fetch all race IDs and course names from Racing Post for a given date."""
    url = f'https://www.racingpost.com/racecards/{date_str}'
    races = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  Failed to fetch {url} (status {resp.status_code})")
            return races

        doc = html.fromstring(resp.content)

        for meeting in doc.xpath('//section[@data-accordion-row]'):
            course_el = meeting.xpath(".//span[contains(@class, 'RC-accordion__courseName')]")
            if not course_el:
                continue
            course = ' '.join(course_el[0].text_content().strip().split())

            for race in meeting.xpath(".//a[contains(@class, 'RC-meetingItem__link')]"):
                race_id = race.attrib.get('data-race-id', '')
                if race_id:
                    races.append({'race_id': race_id, 'course': course})

    except Exception as e:
        print(f"  Error fetching race IDs for {date_str}: {e}")

    return races


def fetch_runners_json(race_id: str) -> dict:
    """Fetch runners JSON for a specific race."""
    url = f'https://www.racingpost.com/profile/horse/data/cardrunners/{race_id}.json'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"    Error fetching runners for race {race_id}: {e}")
    return {}


def furlongs_to_distance_str(furlongs: float) -> str:
    """Convert furlongs to human-readable distance."""
    if not furlongs:
        return ''
    miles = int(furlongs // 8)
    remaining_f = furlongs % 8
    if miles == 0:
        return f"{furlongs:.0f}f"
    elif remaining_f == 0:
        return f"{miles}m"
    else:
        return f"{miles}m{remaining_f:.0f}f"


def fetch_all_racecards(num_days: int = 2) -> list:
    """Fetch all race cards for the next N days."""
    elo_data = load_elo_ratings()
    horse_elo = elo_data.get('horse_elo', {})
    jockey_elo = elo_data.get('jockey_elo', {})
    trainer_elo = elo_data.get('trainer_elo', {})

    all_rows = []
    today = datetime.now()

    for day_offset in range(num_days):
        date = today + timedelta(days=day_offset)
        date_str = date.strftime('%Y-%m-%d')
        day_name = date.strftime('%A')

        print(f"\n{'='*60}")
        print(f"Fetching races for {day_name} {date_str}")
        print(f"{'='*60}")

        race_list = fetch_race_ids_for_date(date_str)
        print(f"  Found {len(race_list)} races")

        for race_info in race_list:
            race_id = race_info['race_id']
            course = race_info['course']

            data = fetch_runners_json(race_id)
            runners_map = data.get('runners', {})
            if not runners_map:
                print(f"    {course} (ID: {race_id}) - no runners data")
                continue

            # Extract race-level info from first runner
            first_runner = list(runners_map.values())[0]
            race_dt_str = first_runner.get('raceDatetime', '')
            race_time = ''
            if race_dt_str:
                try:
                    parsed_dt = datetime.fromisoformat(race_dt_str.replace('Z', '+00:00'))
                    race_time = parsed_dt.strftime('%H:%M')
                except Exception:
                    pass

            furlongs = first_runner.get('distanceFurlongRounded', 0)
            distance = furlongs_to_distance_str(furlongs)
            race_type_code = first_runner.get('raceTypeCode', '')
            race_type = RACE_TYPE.get(race_type_code, 'Flat')

            # Try to get going from the racecard page header
            going = ''

            race_name = f"{course} {race_type} ({distance})" if distance else f"{course} {race_type}"

            runner_count = sum(1 for r in runners_map.values() if not r.get('nonRunner', False))
            print(f"  {race_time} {course} - {race_name} - {runner_count} runners")

            for runner_id, runner in runners_map.items():
                horse_name = (runner.get('horseName') or '').strip()
                if not horse_name or runner.get('nonRunner', False):
                    continue

                jockey = (runner.get('jockeyName') or '').strip()
                trainer = (runner.get('trainerStylename') or '').strip()
                region = (runner.get('countryOriginCode') or '').strip()
                form_figs = runner.get('figuresCalculated', [])
                form = ''.join(f.get('formFigure', '') for f in (form_figs or []))[::-1]

                horse_display = f"{horse_name} ({region})" if region else horse_name

                # ELO lookup (try with and without region suffix)
                h_elo = horse_elo.get(horse_display, horse_elo.get(horse_name, 1500))
                j_elo = jockey_elo.get(jockey, 1500)
                t_elo = trainer_elo.get(trainer, 1500)

                all_rows.append({
                    'date': date_str,
                    'day': day_name,
                    'course': course,
                    'time': race_time,
                    'race_name': race_name,
                    'race_id': race_id,
                    'distance': distance,
                    'going': going,
                    'race_type': race_type,
                    'runners': runner_count,
                    'number': runner.get('startNumber') or '',
                    'draw': runner.get('draw') or '',
                    'horse': horse_display,
                    'jockey': jockey,
                    'trainer': trainer,
                    'age': runner.get('horseAge') or '',
                    'sex': runner.get('horseSexCode') or '',
                    'weight_lbs': runner.get('weightCarriedLbs') or '',
                    'form': form,
                    'ofr': runner.get('officialRatingToday') or '',
                    'rpr': runner.get('rpPostmark') or '',
                    'last_run': runner.get('daysSinceLastRun') or '',
                    'horse_elo': h_elo,
                    'jockey_elo': j_elo,
                    'trainer_elo': t_elo,
                    'composite_elo': h_elo * 0.5 + j_elo * 0.3 + t_elo * 0.2,
                    'odds': 0.0,
                })

            time.sleep(0.3)

    return all_rows


def estimate_odds_from_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate odds from ELO ratings within each race using softmax."""
    for (date, course, race_time), race in df.groupby(['date', 'course', 'time']):
        indices = race.index
        elos = race['composite_elo'].values.astype(float)

        elo_centered = elos - np.mean(elos)
        exp_elos = np.exp(elo_centered / 100)
        probs = exp_elos / exp_elos.sum()

        overround = 1.15
        odds = overround / probs
        odds = np.clip(odds, 1.5, 200)

        df.loc[indices, 'odds'] = np.round(odds, 1)

    return df


def save_and_push(races: list, output_path: str, auto_push: bool = True):
    """Save races to CSV and optionally push to GitHub."""
    df = pd.DataFrame(races)

    if len(df) == 0:
        print("No races found!")
        return

    # Ensure odds column is float
    df['odds'] = df['odds'].astype(float)
    df = estimate_odds_from_elo(df)

    df.to_csv(output_path, index=False)

    # Summary
    total_races = len(df.groupby(['date', 'course', 'time']))
    total_runners = len(df)
    print(f"\nSaved {total_runners} runners across {total_races} races to {output_path}")

    for date_val in sorted(df['date'].unique()):
        day_df = df[df['date'] == date_val]
        day_name = day_df['day'].iloc[0]
        courses = day_df['course'].nunique()
        races_count = len(day_df.groupby(['course', 'time']))
        runners = len(day_df)
        print(f"  {day_name} {date_val}: {courses} courses, {races_count} races, {runners} runners")

    # Auto-push to GitHub
    if auto_push:
        print("\nPushing to GitHub...")
        os.chdir(BASE_DIR)
        os.system('git add data/upcoming_races.csv data/last_update.txt')
        os.system(f'git commit -m "Auto-update race cards {datetime.now().strftime("%Y-%m-%d %H:%M")}"')
        result = os.system('git push')
        if result == 0:
            print("Pushed to GitHub - Streamlit Cloud will auto-redeploy.")
        else:
            print("WARNING: Git push failed. Check credentials.")


def main():
    parser = argparse.ArgumentParser(description='Fetch upcoming horse race cards')
    parser.add_argument('--days', type=int, default=2, help='Number of days to fetch (default: 2)')
    parser.add_argument('--no-push', action='store_true', help='Skip auto-push to GitHub')
    args = parser.parse_args()

    print(f"Horse Racing Race Card Updater")
    print(f"Fetching next {args.days} day(s) of race cards...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    races = fetch_all_racecards(num_days=args.days)

    output_path = os.path.join(DATA_DIR, 'upcoming_races.csv')

    # Log update time
    log_path = os.path.join(DATA_DIR, 'last_update.txt')
    with open(log_path, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    save_and_push(races, output_path, auto_push=not args.no_push)

    print(f"\nDone! {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
