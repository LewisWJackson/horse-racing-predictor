#!/usr/bin/env python3
"""
Auto-Update Races
=================
Fetches real upcoming race cards from Racing Post for today + tomorrow.
Designed to run as a cron job so the dashboard always has fresh data.

Usage:
    python3 update_races.py           # Fetch today + tomorrow
    python3 update_races.py --days 7  # Fetch next 7 days (may be limited)
"""

import requests
import json
import csv
import os
import sys
import pickle
import argparse
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


def load_elo_ratings():
    """Load ELO ratings."""
    elo_path = os.path.join(MODELS_DIR, 'elo_ratings.pkl')
    try:
        with open(elo_path, 'rb') as f:
            return pickle.load(f)
    except:
        return {'horse_elo': {}, 'jockey_elo': {}, 'trainer_elo': {}}


def fetch_rp_racecard_urls(date_str: str) -> list:
    """
    Fetch race URLs from Racing Post for a given date.
    Returns list of (race_id, course, time, href) tuples.
    """
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
            course = course_el[0].text_content().strip()

            for race in meeting.xpath(".//a[contains(@class, 'RC-meetingItem__link')]"):
                race_id = race.attrib.get('data-race-id', '')
                href = race.attrib.get('href', '')

                # Try multiple methods to get the time
                race_time = ''
                time_el = race.xpath(".//span[contains(@class, 'RC-meetingItem__timeText')]")
                if time_el:
                    race_time = time_el[0].text_content().strip()
                else:
                    time_el = race.xpath(".//time")
                    if time_el:
                        race_time = time_el[0].text_content().strip()

                if not race_time and href:
                    # Extract from href if possible
                    parts = href.split('/')
                    if len(parts) > 3:
                        race_time = parts[-1] if ':' in parts[-1] else ''

                races.append({
                    'race_id': race_id,
                    'course': course,
                    'time': race_time,
                    'href': href
                })

    except Exception as e:
        print(f"  Error fetching race URLs for {date_str}: {e}")

    return races


def fetch_race_runners(race_id: str) -> dict:
    """
    Fetch runners JSON for a specific race from Racing Post.
    """
    url = f'https://www.racingpost.com/profile/horse/data/cardrunners/{race_id}.json'

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"    Error fetching runners for race {race_id}: {e}")

    return {}


def fetch_race_details(href: str) -> dict:
    """
    Fetch race details (distance, going, race name, etc.) from the racecard page.
    """
    url = f'https://www.racingpost.com{href}'
    details = {}

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return details

        doc = html.fromstring(resp.content)

        # Race name
        race_name_el = doc.xpath("//span[contains(@class, 'RC-header__raceInstanceTitle')]")
        details['race_name'] = race_name_el[0].text_content().strip() if race_name_el else ''

        # Distance
        dist_el = doc.xpath("//strong[contains(@class, 'RC-header__raceDistanceRound')]")
        details['distance'] = dist_el[0].text_content().strip() if dist_el else ''

        if not details['distance']:
            dist_el2 = doc.xpath("//span[contains(@class, 'RC-header__raceDistance')]")
            details['distance'] = dist_el2[0].text_content().strip().strip('()') if dist_el2 else ''

        # Going
        going_el = doc.xpath("//div[contains(@class, 'RC-headerBox__going')]")
        if going_el:
            going_text = going_el[0].text_content().strip()
            if 'Going:' in going_text:
                details['going'] = going_text.split('Going:')[1].strip()
            else:
                details['going'] = going_text

        # Class
        class_el = doc.xpath("//span[contains(@class, 'RC-header__raceClass')]")
        details['race_class'] = class_el[0].text_content().strip().strip('()') if class_el else ''

        # Runner count
        runners_el = doc.xpath("//div[contains(@class, 'RC-headerBox__runners')]")
        if runners_el:
            runners_text = runners_el[0].text_content().strip()
            if 'Runners:' in runners_text:
                try:
                    details['num_runners'] = int(runners_text.split('Runners:')[1].strip().split('(')[0].strip())
                except:
                    pass

    except Exception as e:
        print(f"    Error fetching race details: {e}")

    return details


def fetch_all_racecards(num_days: int = 2) -> list:
    """
    Fetch all race cards for the next N days.
    Returns list of race dicts with runners.
    """
    elo_data = load_elo_ratings()
    horse_elo = elo_data.get('horse_elo', {})
    jockey_elo = elo_data.get('jockey_elo', {})
    trainer_elo = elo_data.get('trainer_elo', {})

    all_races = []
    today = datetime.now()

    for day_offset in range(num_days):
        date = today + timedelta(days=day_offset)
        date_str = date.strftime('%Y-%m-%d')
        day_name = date.strftime('%A')

        print(f"\n{'='*60}")
        print(f"Fetching races for {day_name} {date_str}")
        print(f"{'='*60}")

        # Get race URLs for this date
        race_urls = fetch_rp_racecard_urls(date_str)
        print(f"  Found {len(race_urls)} races")

        for race_info in race_urls:
            race_id = race_info['race_id']
            course = race_info['course']
            race_time = race_info['time']

            # Get race details (including off time)
            details = fetch_race_details(race_info['href'])
            race_name = details.get('race_name', f'{course} Race')
            distance = details.get('distance', '')
            going = details.get('going', '')

            # Get runners
            runners_data = fetch_race_runners(race_id)
            runners_map = runners_data.get('runners', {})

            # Get off time from first runner's raceDatetime
            if runners_map and not race_time:
                first_runner = list(runners_map.values())[0]
                race_dt = first_runner.get('raceDatetime', '')
                if race_dt:
                    try:
                        from datetime import datetime as dt_parser
                        parsed_dt = dt_parser.fromisoformat(race_dt.replace('Z', '+00:00'))
                        race_time = parsed_dt.strftime('%H:%M')
                    except:
                        pass

            # Clean up course name (remove excess whitespace)
            course = ' '.join(course.split())

            print(f"  {race_time} {course} (ID: {race_id}) - {len(runners_map)} runners")

            # Get race name from details or runners
            if not race_name or race_name == f'{course} Race':
                first_r = list(runners_map.values())[0] if runners_map else {}
                race_name = details.get('race_name', '') or first_r.get('raceTitle', '') or f'{course} Race'

            for runner_id, runner in runners_map.items():
                horse_name = (runner.get('horseName') or '').strip()
                if not horse_name:
                    continue

                # Skip non-runners
                if runner.get('nonRunner', False):
                    continue

                jockey = (runner.get('jockeyName') or '').strip()
                trainer = (runner.get('trainerStylename') or '').strip()
                draw = runner.get('draw') or ''
                number = runner.get('startNumber') or ''
                age = runner.get('horseAge') or ''
                weight_lbs = runner.get('weightCarriedLbs') or ''
                form = ''.join(f['formFigure'] for f in runner.get('figuresCalculated', []))[::-1] if runner.get('figuresCalculated') else ''
                ofr = runner.get('officialRatingToday') or ''
                rpr = runner.get('rpPostmark') or ''
                last_run = runner.get('daysSinceLastRun') or ''
                sex = runner.get('horseSexCode') or ''
                region = runner.get('countryOriginCode') or ''

                # Look up ELO ratings
                h_elo = horse_elo.get(horse_name, horse_elo.get(f"{horse_name} ({region})", 1500))
                j_elo = jockey_elo.get(jockey, 1500)
                t_elo = trainer_elo.get(trainer, 1500)

                # Generate odds estimate from RPR/OFR if available
                # (Real odds would come from a betting API)
                odds = 0  # Will be populated from betting markets

                all_races.append({
                    'date': date_str,
                    'day': day_name,
                    'course': course,
                    'time': race_time,
                    'race_name': race_name,
                    'race_id': race_id,
                    'distance': distance,
                    'going': going,
                    'race_class': details.get('race_class', ''),
                    'num_runners': details.get('num_runners', ''),
                    'number': number,
                    'draw': draw,
                    'horse': f"{horse_name} ({region})" if region else horse_name,
                    'horse_clean': horse_name,
                    'jockey': jockey,
                    'trainer': trainer,
                    'age': age,
                    'sex': sex,
                    'weight_lbs': weight_lbs,
                    'form': form,
                    'ofr': ofr,
                    'rpr': rpr,
                    'last_run': last_run,
                    'horse_elo': h_elo,
                    'jockey_elo': j_elo,
                    'trainer_elo': t_elo,
                    'composite_elo': h_elo * 0.5 + j_elo * 0.3 + t_elo * 0.2,
                    'odds': odds
                })

            # Small delay to be polite
            import time
            time.sleep(0.5)

    return all_races


def estimate_odds_from_elo(df):
    """
    When real odds aren't available, estimate from ELO ratings within each race.
    Uses softmax-style probability distribution.
    """
    import pandas as pd
    import numpy as np

    for (date, course, time), race in df.groupby(['date', 'course', 'time']):
        indices = race.index
        elos = race['composite_elo'].values

        # Softmax to get probabilities
        elo_centered = elos - np.mean(elos)
        exp_elos = np.exp(elo_centered / 100)  # Temperature scaling
        probs = exp_elos / exp_elos.sum()

        # Convert probabilities to decimal odds (with ~15% overround)
        overround = 1.15
        odds = overround / probs
        odds = np.clip(odds, 1.5, 200)

        df.loc[indices, 'odds'] = np.round(odds, 1)

    return df


def save_races(races: list, output_path: str):
    """Save races to CSV."""
    import pandas as pd

    df = pd.DataFrame(races)

    if len(df) == 0:
        print("No races found!")
        return

    # Estimate odds from ELO if not available
    if (df['odds'] == 0).any():
        df = estimate_odds_from_elo(df)

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} runners across {len(df.groupby(['date', 'course', 'time']))} races to {output_path}")

    # Summary
    for date in sorted(df['date'].unique()):
        day_df = df[df['date'] == date]
        day_name = day_df['day'].iloc[0]
        courses = day_df['course'].nunique()
        races = len(day_df.groupby(['course', 'time']))
        runners = len(day_df)
        print(f"  {day_name} {date}: {courses} courses, {races} races, {runners} runners")


def main():
    parser = argparse.ArgumentParser(description='Fetch upcoming horse race cards')
    parser.add_argument('--days', type=int, default=2, help='Number of days to fetch (default: 2)')
    args = parser.parse_args()

    print(f"Horse Racing Race Card Updater")
    print(f"Fetching next {args.days} day(s) of race cards...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    races = fetch_all_racecards(num_days=args.days)

    output_path = os.path.join(DATA_DIR, 'upcoming_races.csv')
    save_races(races, output_path)

    # Log update time
    log_path = os.path.join(DATA_DIR, 'last_update.txt')
    with open(log_path, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print(f"\nDone! Dashboard data updated at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
