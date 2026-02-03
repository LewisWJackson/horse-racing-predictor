"""
Data Preprocessing for Horse Racing
====================================
Standardizes data from various sources into the format expected by the predictor.

Supports:
1. Kaggle Hong Kong Racing dataset
2. UK Racing Data (CSV exports)
3. Custom CSV formats

Expected output columns:
- race_id, date, track, race_name, race_class, distance_furlongs, going
- horse_id, horse_name, position, jockey, trainer
- Optional: weight, draw, age, odds
"""

import pandas as pd
import numpy as np
from typing import Optional
import re
import warnings
warnings.filterwarnings('ignore')


def parse_distance_to_furlongs(distance_str: str) -> float:
    """
    Convert distance string to furlongs.

    Examples:
        "5f" -> 5.0
        "1m" -> 8.0
        "1m 2f" -> 10.0
        "1m2f" -> 10.0
        "2400m" (meters) -> 12.0
        "1 mile" -> 8.0
        "6 furlongs" -> 6.0
    """
    if pd.isna(distance_str):
        return 8.0  # Default to 1 mile

    dist_str = str(distance_str).lower().strip()

    # Already numeric (assume furlongs)
    try:
        return float(dist_str)
    except ValueError:
        pass

    # Meters to furlongs (1 furlong = 201.168 meters)
    if 'm' in dist_str and any(c.isdigit() for c in dist_str):
        # Check if it's meters (like "2400m" or "2400 meters")
        meters_match = re.search(r'(\d+)\s*(?:meters?|metre?s?|m)$', dist_str)
        if meters_match and int(meters_match.group(1)) > 100:  # Likely meters not miles
            meters = int(meters_match.group(1))
            return meters / 201.168

    # Parse miles and furlongs
    miles = 0
    furlongs = 0

    # Match patterns like "1m 2f", "1m2f", "1 mile 2 furlongs"
    mile_match = re.search(r'(\d+)\s*(?:m|mile|miles)', dist_str)
    furlong_match = re.search(r'(\d+)\s*(?:f|furlong|furlongs)', dist_str)

    if mile_match:
        miles = int(mile_match.group(1))
    if furlong_match:
        furlongs = int(furlong_match.group(1))

    if miles > 0 or furlongs > 0:
        return miles * 8 + furlongs

    # Handle yards (110 yards ≈ 0.5 furlongs)
    yards_match = re.search(r'(\d+)\s*(?:y|yards?)', dist_str)
    if yards_match:
        yards = int(yards_match.group(1))
        return yards / 220  # 220 yards = 1 furlong

    # Default
    return 8.0


def standardize_going(going_str: str) -> str:
    """
    Standardize going/track condition descriptions.

    Returns: Good, Good_to_Firm, Firm, Good_to_Soft, Soft, Heavy, All_Weather
    """
    if pd.isna(going_str):
        return 'Good'

    going = str(going_str).lower().strip()

    # All-weather surfaces
    if any(x in going for x in ['polytrack', 'tapeta', 'fibresand', 'all weather', 'aw']):
        return 'All_Weather'

    # Heavy/Soft
    if any(x in going for x in ['heavy', 'very soft']):
        return 'Heavy'

    if any(x in going for x in ['soft', 'yielding', 'holding']):
        if 'good' in going:
            return 'Good_to_Soft'
        return 'Soft'

    # Firm
    if any(x in going for x in ['hard', 'very firm']):
        return 'Firm'

    if 'firm' in going:
        if 'good' in going:
            return 'Good_to_Firm'
        return 'Firm'

    # Good
    if 'standard' in going or 'good' in going:
        return 'Good'

    # Fast (common in US)
    if 'fast' in going:
        return 'Firm'

    return 'Good'


def standardize_race_class(class_str: str) -> str:
    """
    Standardize race class descriptions.

    Returns categories: Group_1, Group_2, Group_3, Listed, Stakes,
                        Class_1, Class_2, Class_3, Class_4, Class_5,
                        Handicap, Maiden, Novice, Claiming, Other
    """
    if pd.isna(class_str):
        return 'Other'

    cls = str(class_str).lower().strip()

    # Group/Grade races
    if any(x in cls for x in ['group 1', 'grade 1', 'g1', 'grp 1']):
        return 'Group_1'
    if any(x in cls for x in ['group 2', 'grade 2', 'g2', 'grp 2']):
        return 'Group_2'
    if any(x in cls for x in ['group 3', 'grade 3', 'g3', 'grp 3']):
        return 'Group_3'

    if 'listed' in cls:
        return 'Listed'

    if 'stakes' in cls and not 'claiming' in cls:
        return 'Stakes'

    # UK Class system
    if 'class 1' in cls or 'class1' in cls:
        return 'Class_1'
    if 'class 2' in cls or 'class2' in cls:
        return 'Class_2'
    if 'class 3' in cls or 'class3' in cls:
        return 'Class_3'
    if 'class 4' in cls or 'class4' in cls:
        return 'Class_4'
    if 'class 5' in cls or 'class5' in cls or 'class 6' in cls:
        return 'Class_5'

    # Race types
    if 'handicap' in cls or 'hcap' in cls:
        return 'Handicap'
    if 'maiden' in cls:
        return 'Maiden'
    if 'novice' in cls:
        return 'Novice'
    if 'claiming' in cls:
        return 'Claiming'

    return 'Other'


def clean_name(name: str) -> str:
    """Clean horse/jockey/trainer name."""
    if pd.isna(name):
        return 'Unknown'

    name = str(name).strip()

    # Remove country codes (IRE), (USA), etc.
    name = re.sub(r'\s*\([A-Z]{2,3}\)\s*', '', name)

    # Remove trailing numbers or codes
    name = re.sub(r'\s+\d+$', '', name)

    return name.strip()


def create_unique_horse_id(horse_name: str, trainer: str = None, year_of_birth: int = None) -> str:
    """
    Create a unique horse ID.

    This handles cases where multiple horses have the same name.
    """
    parts = [clean_name(horse_name)]

    if trainer and not pd.isna(trainer):
        parts.append(clean_name(trainer)[:10])

    if year_of_birth and not pd.isna(year_of_birth):
        parts.append(str(int(year_of_birth)))

    return "_".join(parts).replace(" ", "_").lower()


def preprocess_kaggle_hk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess Kaggle Hong Kong Racing dataset.

    Expected columns vary by dataset version.
    """
    print("Preprocessing Hong Kong Racing data...")

    # Create a copy
    df = df.copy()

    # Map columns (adjust based on actual dataset)
    column_mapping = {
        'race_id': 'race_id',
        'date': 'date',
        'venue': 'track',
        'race_name': 'race_name',
        'race_class': 'race_class',
        'distance': 'distance_raw',
        'going': 'going_raw',
        'horse_name': 'horse_name',
        'horse_id': 'horse_id_raw',
        'finish_position': 'position',
        'jockey': 'jockey',
        'trainer': 'trainer',
        'actual_weight': 'weight',
        'draw': 'draw',
        'horse_age': 'age',
        'win_odds': 'odds',
    }

    # Rename columns that exist
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Create race_id if not exists
    if 'race_id' not in df.columns:
        if 'date' in df.columns and 'track' in df.columns:
            df['race_id'] = df['date'].astype(str) + '_' + df['track'].astype(str) + '_' + df.groupby(['date', 'track']).cumcount().astype(str)
        else:
            df['race_id'] = range(len(df))

    # Process distance
    if 'distance_raw' in df.columns:
        df['distance_furlongs'] = df['distance_raw'].apply(parse_distance_to_furlongs)
    elif 'distance' in df.columns:
        df['distance_furlongs'] = df['distance'].apply(parse_distance_to_furlongs)
    else:
        df['distance_furlongs'] = 8.0

    # Process going
    if 'going_raw' in df.columns:
        df['going'] = df['going_raw'].apply(standardize_going)
    elif 'going' not in df.columns:
        df['going'] = 'Good'

    # Process race class
    if 'race_class' in df.columns:
        df['race_class'] = df['race_class'].apply(standardize_race_class)
    else:
        df['race_class'] = 'Other'

    # Create unique horse ID
    if 'horse_id' not in df.columns:
        df['horse_id'] = df.apply(
            lambda row: create_unique_horse_id(
                row.get('horse_name', 'Unknown'),
                row.get('trainer', None),
                row.get('age', None)
            ),
            axis=1
        )

    # Clean names
    df['horse_name'] = df['horse_name'].apply(clean_name)
    df['jockey'] = df['jockey'].apply(clean_name) if 'jockey' in df.columns else 'Unknown'
    df['trainer'] = df['trainer'].apply(clean_name) if 'trainer' in df.columns else 'Unknown'
    df['track'] = df['track'].apply(clean_name) if 'track' in df.columns else 'Unknown'

    # Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Ensure position is numeric
    if 'position' in df.columns:
        df['position'] = pd.to_numeric(df['position'], errors='coerce')

    # Fill missing race names
    if 'race_name' not in df.columns:
        df['race_name'] = 'Race ' + df['race_id'].astype(str)

    # Select final columns
    final_cols = [
        'race_id', 'date', 'track', 'race_name', 'race_class',
        'distance_furlongs', 'going', 'horse_id', 'horse_name',
        'position', 'jockey', 'trainer'
    ]

    optional_cols = ['weight', 'draw', 'age', 'odds']
    for col in optional_cols:
        if col in df.columns:
            final_cols.append(col)

    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in df.columns]

    result = df[final_cols].copy()

    print(f"Processed {len(result)} rows, {result['race_id'].nunique()} unique races")
    return result


def preprocess_generic(df: pd.DataFrame, column_mapping: dict = None) -> pd.DataFrame:
    """
    Preprocess a generic racing dataset with custom column mapping.

    Args:
        df: Raw DataFrame
        column_mapping: Dict mapping source columns to target columns

    Returns:
        Preprocessed DataFrame
    """
    print("Preprocessing generic racing data...")

    df = df.copy()

    # Apply column mapping if provided
    if column_mapping:
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

    # Process distance
    if 'distance' in df.columns and 'distance_furlongs' not in df.columns:
        df['distance_furlongs'] = df['distance'].apply(parse_distance_to_furlongs)
    elif 'distance_furlongs' not in df.columns:
        df['distance_furlongs'] = 8.0

    # Process going
    if 'going' in df.columns:
        df['going'] = df['going'].apply(standardize_going)
    else:
        df['going'] = 'Good'

    # Process race class
    if 'race_class' in df.columns:
        df['race_class'] = df['race_class'].apply(standardize_race_class)
    else:
        df['race_class'] = 'Other'

    # Create IDs if needed
    if 'race_id' not in df.columns:
        df['race_id'] = range(len(df))

    if 'horse_id' not in df.columns:
        df['horse_id'] = df.apply(
            lambda row: create_unique_horse_id(
                row.get('horse_name', 'Unknown'),
                row.get('trainer', None)
            ),
            axis=1
        )

    # Clean and standardize
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['position'] = pd.to_numeric(df.get('position'), errors='coerce')

    for col in ['horse_name', 'jockey', 'trainer', 'track']:
        if col in df.columns:
            df[col] = df[col].apply(clean_name)
        else:
            df[col] = 'Unknown'

    print(f"Processed {len(df)} rows")
    return df


def load_and_preprocess(filepath: str, source_type: str = 'auto',
                        column_mapping: dict = None) -> pd.DataFrame:
    """
    Load and preprocess racing data from file.

    Args:
        filepath: Path to CSV file
        source_type: 'hk_kaggle', 'generic', or 'auto'
        column_mapping: Optional column mapping for generic sources

    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading data from {filepath}...")

    # Load data
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Auto-detect source type
    if source_type == 'auto':
        if any('hong kong' in str(col).lower() for col in df.columns):
            source_type = 'hk_kaggle'
        elif 'race_id' in df.columns and 'horse_id' in df.columns:
            source_type = 'generic'
        else:
            source_type = 'generic'

    # Preprocess based on source
    if source_type == 'hk_kaggle':
        return preprocess_kaggle_hk(df)
    else:
        return preprocess_generic(df, column_mapping)


def create_sample_data(n_races: int = 100, n_horses_per_race: int = 12) -> pd.DataFrame:
    """
    Create sample horse racing data for testing.

    This generates realistic-looking synthetic data.
    """
    print(f"Creating sample data: {n_races} races, ~{n_horses_per_race} horses each...")

    np.random.seed(42)

    tracks = ['Ascot', 'Newmarket', 'Epsom', 'Cheltenham', 'Aintree', 'York', 'Goodwood']
    goings = ['Good', 'Good_to_Firm', 'Good_to_Soft', 'Soft', 'Firm']
    classes = ['Group_1', 'Group_2', 'Group_3', 'Listed', 'Handicap', 'Maiden', 'Class_3', 'Class_4']

    # Generate horse pool
    n_horses = 500
    horse_pool = [f"Horse_{i}" for i in range(n_horses)]
    jockey_pool = [f"J.{['Smith', 'Johnson', 'Brown', 'Williams', 'Jones', 'Taylor'][i % 6]}" for i in range(50)]
    trainer_pool = [f"T.{['Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin'][i % 6]}" for i in range(30)]

    records = []
    start_date = pd.Timestamp('2020-01-01')

    for race_num in range(n_races):
        race_date = start_date + pd.Timedelta(days=race_num * 3)
        track = np.random.choice(tracks)
        going = np.random.choice(goings)
        race_class = np.random.choice(classes)
        distance = np.random.choice([5, 6, 7, 8, 10, 12, 14, 16])
        race_id = f"R{race_num:04d}"
        race_name = f"{track} {race_class} ({distance}f)"

        # Select horses for this race
        n_runners = np.random.randint(6, n_horses_per_race + 1)
        horses = np.random.choice(horse_pool, size=n_runners, replace=False)

        # Assign positions (1 to n_runners)
        positions = list(range(1, n_runners + 1))
        np.random.shuffle(positions)

        for i, horse in enumerate(horses):
            records.append({
                'race_id': race_id,
                'date': race_date,
                'track': track,
                'race_name': race_name,
                'race_class': race_class,
                'distance_furlongs': distance,
                'going': going,
                'horse_id': horse.lower().replace(' ', '_'),
                'horse_name': horse,
                'position': positions[i],
                'jockey': np.random.choice(jockey_pool),
                'trainer': np.random.choice(trainer_pool),
                'weight': np.random.randint(110, 140),
                'draw': i + 1,
                'age': np.random.randint(3, 8),
                'odds': round(np.random.exponential(10) + 2, 1),
            })

    df = pd.DataFrame(records)
    print(f"Created sample data: {len(df)} rows, {df['race_id'].nunique()} races")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("HORSE RACING DATA PREPROCESSING")
    print("=" * 60)

    print("\nThis module provides data preprocessing utilities.")
    print("\nUsage:")
    print("  from data_preprocessing import load_and_preprocess")
    print("  df = load_and_preprocess('path/to/data.csv')")

    print("\n" + "=" * 60)
    print("CREATING SAMPLE DATA FOR TESTING")
    print("=" * 60)

    sample_df = create_sample_data(n_races=200)
    sample_df.to_csv('../data/raw/sample_racing_data.csv', index=False)
    print(f"\nSample data saved to ../data/raw/sample_racing_data.csv")

    print("\nSample data preview:")
    print(sample_df.head(10))
