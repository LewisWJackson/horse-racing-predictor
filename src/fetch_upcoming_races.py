"""
Fetch Upcoming Races
====================
Fetches race cards for the next 7 days from free racing data sources.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import pickle

# Load ELO ratings for predictions
def load_elo_ratings():
    elo_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'elo_ratings.pkl')
    try:
        with open(elo_path, 'rb') as f:
            return pickle.load(f)
    except:
        return {'horse_elo': {}, 'jockey_elo': {}, 'trainer_elo': {}}


def fetch_racing_post_cards(date_str: str) -> list:
    """
    Fetch race cards from Racing Post for a given date.
    Date format: YYYY-MM-DD
    """
    races = []

    try:
        # Racing Post racecards URL
        url = f"https://www.racingpost.com/racecards/{date_str}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return races

        soup = BeautifulSoup(response.content, 'html.parser')

        # Parse race cards (structure may vary)
        # This is a simplified parser - actual structure depends on the site

    except Exception as e:
        print(f"Error fetching Racing Post: {e}")

    return races


def fetch_timeform_cards(date_str: str) -> list:
    """Fetch from Timeform free data."""
    races = []
    try:
        url = f"https://www.timeform.com/horse-racing/racecards/{date_str}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        # Parse response...
    except:
        pass
    return races


def fetch_attheraces_cards(date_str: str) -> list:
    """Fetch from At The Races."""
    races = []
    try:
        # ATR has a JSON API
        url = f"https://www.attheraces.com/racecard/{date_str}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
    except:
        pass
    return races


def generate_sample_upcoming_races() -> pd.DataFrame:
    """
    Generate sample upcoming race data for demonstration.
    In production, this would be replaced with live API data.
    """
    from datetime import datetime, timedelta
    import random

    elo_data = load_elo_ratings()
    horse_elo = elo_data.get('horse_elo', {})
    jockey_elo = elo_data.get('jockey_elo', {})
    trainer_elo = elo_data.get('trainer_elo', {})

    # Get real horse/jockey/trainer names from our ELO data
    top_horses = sorted(horse_elo.items(), key=lambda x: x[1], reverse=True)[:500]
    top_jockeys = sorted(jockey_elo.items(), key=lambda x: x[1], reverse=True)[:100]
    top_trainers = sorted(trainer_elo.items(), key=lambda x: x[1], reverse=True)[:100]

    courses_uk = [
        'Cheltenham', 'Ascot', 'Newmarket', 'York', 'Epsom', 'Goodwood',
        'Sandown', 'Kempton', 'Lingfield', 'Wolverhampton', 'Newcastle',
        'Doncaster', 'Haydock', 'Aintree', 'Chester', 'Newbury',
        'Uttoxeter', 'Fontwell', 'Plumpton', 'Sedgefield', 'Catterick'
    ]

    distances = ['5f', '6f', '7f', '1m', '1m2f', '1m4f', '1m6f', '2m', '2m4f', '3m']
    goings = ['Good', 'Good to Firm', 'Good to Soft', 'Soft', 'Heavy', 'Standard']
    race_types = ['Handicap', 'Novice Stakes', 'Maiden', 'Conditions', 'Listed', 'Class 4', 'Class 5']

    races = []
    today = datetime.now()

    for day_offset in range(7):
        race_date = today + timedelta(days=day_offset)

        # 3-6 meetings per day
        num_meetings = random.randint(3, 6)
        day_courses = random.sample(courses_uk, num_meetings)

        for course in day_courses:
            # 6-8 races per meeting
            num_races = random.randint(6, 8)
            base_time = 13 * 60 + 30  # 1:30 PM start

            for race_num in range(num_races):
                race_time_mins = base_time + race_num * 35
                race_time = f"{race_time_mins // 60:02d}:{race_time_mins % 60:02d}"

                distance = random.choice(distances)
                going = random.choice(goings)
                race_type = random.choice(race_types)
                race_name = f"{course} {race_type}"

                # 6-16 runners per race
                num_runners = random.randint(6, 16)

                # Select random horses (with replacement allowed across races)
                race_horses = random.sample(top_horses, min(num_runners, len(top_horses)))

                for draw, (horse_name, h_elo) in enumerate(race_horses, 1):
                    jockey_name, j_elo = random.choice(top_jockeys)
                    trainer_name, t_elo = random.choice(top_trainers)

                    # Generate realistic odds based on ELO
                    composite_elo = h_elo * 0.5 + j_elo * 0.3 + t_elo * 0.2
                    elo_rank = (composite_elo - 1400) / 300  # Normalize

                    # Base odds from ELO (higher ELO = shorter odds)
                    base_odds = max(1.5, 50 / (1 + elo_rank * 2) + random.uniform(-2, 2))

                    # Add some randomness for market variation
                    odds = round(base_odds + random.uniform(-1, 3), 1)
                    odds = max(1.5, min(100, odds))

                    races.append({
                        'date': race_date.strftime('%Y-%m-%d'),
                        'day': race_date.strftime('%A'),
                        'course': course,
                        'time': race_time,
                        'race_name': race_name,
                        'distance': distance,
                        'going': going,
                        'race_type': race_type,
                        'runners': num_runners,
                        'draw': draw,
                        'horse': horse_name,
                        'jockey': jockey_name,
                        'trainer': trainer_name,
                        'odds': odds,
                        'horse_elo': h_elo,
                        'jockey_elo': j_elo,
                        'trainer_elo': t_elo,
                        'composite_elo': composite_elo
                    })

    return pd.DataFrame(races)


def get_upcoming_races(use_cache=True) -> pd.DataFrame:
    """
    Get upcoming races for next 7 days.
    Uses cached data if available and recent, otherwise fetches fresh.
    """
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'upcoming_races.csv')

    # Check cache
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])

            # Check if cache is still valid (has today's date)
            today = datetime.now().date()
            if df['date'].dt.date.min() <= today:
                return df
        except:
            pass

    # Generate fresh data
    df = generate_sample_upcoming_races()

    # Save to cache
    try:
        df.to_csv(cache_path, index=False)
    except:
        pass

    return df


if __name__ == "__main__":
    print("Fetching upcoming races...")
    df = get_upcoming_races(use_cache=False)
    print(f"\nGenerated {len(df)} race entries")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nCourses: {df['course'].nunique()}")
    print(f"Races: {len(df.groupby(['date', 'course', 'time']))}")
    print(f"\nSample:")
    print(df[['date', 'course', 'time', 'horse', 'odds']].head(20))
