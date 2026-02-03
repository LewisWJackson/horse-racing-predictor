"""
Horse Racing Predictor - Real Data Training
============================================
Trains on the UK/Ireland 2015-2025 dataset (1.75M rows)
Optimized for maximum accuracy with XGBoost hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
import pickle

np.random.seed(42)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def parse_distance(dist_str: str) -> float:
    """Convert distance string like '2m3½f' to furlongs."""
    if pd.isna(dist_str):
        return 8.0

    dist = str(dist_str).lower().strip()
    miles = 0
    furlongs = 0

    # Extract miles
    mile_match = re.search(r'(\d+)m', dist)
    if mile_match:
        miles = int(mile_match.group(1))

    # Extract furlongs (handle fractions like ½)
    furlong_match = re.search(r'(\d+)(?:½)?f', dist)
    if furlong_match:
        furlongs = int(furlong_match.group(1))
        if '½' in dist or '1/2' in dist:
            furlongs += 0.5

    total = miles * 8 + furlongs
    return total if total > 0 else 8.0


def parse_position(pos) -> Optional[int]:
    """Parse position - handle PU, F, UR, etc."""
    if pd.isna(pos):
        return None
    pos_str = str(pos).strip().upper()

    # Non-finishers
    if pos_str in ['PU', 'F', 'UR', 'RO', 'BD', 'CO', 'RR', 'LFT', 'DSQ', 'VOI', 'REF', 'SU']:
        return None

    try:
        return int(float(pos_str))
    except:
        return None


def parse_odds(sp: str) -> float:
    """Parse starting price odds like '1/3F', '25/1', 'evens'."""
    if pd.isna(sp):
        return 20.0

    sp_str = str(sp).strip().upper().replace('F', '').replace('J', '').replace('C', '')

    if 'EVENS' in sp_str or 'EVS' in sp_str:
        return 2.0

    try:
        if '/' in sp_str:
            parts = sp_str.split('/')
            num = float(parts[0])
            denom = float(parts[1])
            return (num / denom) + 1  # Convert fractional to decimal
        else:
            return float(sp_str)
    except:
        return 20.0


def parse_weight(wgt: str) -> float:
    """Parse weight like '11-6' (stones-pounds) to total pounds."""
    if pd.isna(wgt):
        return 140.0  # Default

    wgt_str = str(wgt).strip()
    try:
        if '-' in wgt_str:
            parts = wgt_str.split('-')
            stones = int(parts[0])
            pounds = int(parts[1])
            return stones * 14 + pounds
        else:
            return float(wgt_str)
    except:
        return 140.0


def standardize_going(going: str) -> str:
    """Standardize going descriptions."""
    if pd.isna(going):
        return 'Good'

    going_lower = str(going).lower()

    if any(x in going_lower for x in ['heavy']):
        return 'Heavy'
    elif any(x in going_lower for x in ['soft', 'yielding']):
        if 'good' in going_lower:
            return 'Good_to_Soft'
        return 'Soft'
    elif any(x in going_lower for x in ['firm', 'hard', 'fast']):
        if 'good' in going_lower:
            return 'Good_to_Firm'
        return 'Firm'
    elif 'standard' in going_lower:
        return 'Standard'  # All-weather
    else:
        return 'Good'


def categorize_distance(furlongs: float) -> str:
    """Categorize distance."""
    if furlongs < 7:
        return 'Sprint'
    elif furlongs < 10:
        return 'Mile'
    elif furlongs < 14:
        return 'Middle'
    elif furlongs < 18:
        return 'Long'
    else:
        return 'Marathon'


def load_and_clean_data(filepath: str, sample_frac: float = None) -> pd.DataFrame:
    """Load and clean the raceform.csv data."""
    print(f"Loading data from {filepath}...")

    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} rows")

    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"Sampled to {len(df):,} rows")

    # Parse columns
    print("Parsing data...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['position'] = df['pos'].apply(parse_position)
    df['distance_furlongs'] = df['dist'].apply(parse_distance)
    df['odds_decimal'] = df['sp'].apply(parse_odds)
    df['weight_lbs'] = df['wgt'].apply(parse_weight)
    df['going_std'] = df['going'].apply(standardize_going)
    df['distance_cat'] = df['distance_furlongs'].apply(categorize_distance)

    # Clean horse names (remove country codes)
    df['horse_clean'] = df['horse'].str.replace(r'\s*\([A-Z]{2,3}\)\s*', '', regex=True).str.strip()
    df['horse_id'] = df['horse_clean'].str.lower().str.replace(' ', '_')

    # Clean jockey/trainer names
    df['jockey_clean'] = df['jockey'].fillna('Unknown')
    df['trainer_clean'] = df['trainer'].fillna('Unknown')
    df['course_clean'] = df['course'].str.replace(r'\s*\([A-Z]{2,3}\)\s*', '', regex=True).str.strip()

    # Filter to valid finishers only
    valid = df[df['position'].notna() & (df['position'] > 0)].copy()
    print(f"Valid finishers: {len(valid):,} rows ({len(valid)/len(df)*100:.1f}%)")

    # Sort by date
    valid = valid.sort_values('date').reset_index(drop=True)

    return valid


# ============================================================================
# ELO SYSTEM (Optimized for speed)
# ============================================================================

class FastELO:
    """Optimized ELO system for large datasets."""

    def __init__(self, k_factor: int = 32, initial_elo: int = 1500):
        self.k = k_factor
        self.initial = initial_elo

        # Horse ratings
        self.horse_elo: Dict[str, float] = {}
        self.horse_track_elo: Dict[Tuple[str, str], float] = {}
        self.horse_dist_elo: Dict[Tuple[str, str], float] = {}
        self.horse_going_elo: Dict[Tuple[str, str], float] = {}
        self.horse_races: Dict[str, int] = defaultdict(int)

        # Jockey ratings
        self.jockey_elo: Dict[str, float] = {}
        self.jockey_races: Dict[str, int] = defaultdict(int)

        # Trainer ratings
        self.trainer_elo: Dict[str, float] = {}
        self.trainer_races: Dict[str, int] = defaultdict(int)

        # Jockey-Trainer combos
        self.jt_combo_wins: Dict[Tuple[str, str], int] = defaultdict(int)
        self.jt_combo_races: Dict[Tuple[str, str], int] = defaultdict(int)

        # Horse form tracking
        self.horse_recent_pos: Dict[str, List[int]] = defaultdict(list)
        self.horse_last_date: Dict[str, pd.Timestamp] = {}
        self.horse_wins: Dict[str, int] = defaultdict(int)
        self.horse_places: Dict[str, int] = defaultdict(int)

        # Track stats
        self.horse_track_wins: Dict[Tuple[str, str], int] = defaultdict(int)
        self.horse_track_races: Dict[Tuple[str, str], int] = defaultdict(int)

        # Distance stats
        self.horse_dist_wins: Dict[Tuple[str, str], int] = defaultdict(int)
        self.horse_dist_races: Dict[Tuple[str, str], int] = defaultdict(int)

        # Going stats
        self.horse_going_wins: Dict[Tuple[str, str], int] = defaultdict(int)
        self.horse_going_races: Dict[Tuple[str, str], int] = defaultdict(int)

    def get(self, d: dict, key, default=None):
        """Safe dict get with default."""
        if default is None:
            default = self.initial
        return d.get(key, default)

    def expected(self, elo_a: float, elo_b: float) -> float:
        """Expected score."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

    def process_race(self, race_df: pd.DataFrame, track: str, dist_cat: str, going: str):
        """Process a single race and update all ELOs."""
        runners = race_df[['horse_id', 'position', 'jockey_clean', 'trainer_clean']].values.tolist()
        n = len(runners)

        if n < 2:
            return

        # Calculate ELO changes using virtual matchups
        horse_changes = defaultdict(float)
        horse_matchups = defaultdict(int)
        jockey_changes = defaultdict(float)
        jockey_matchups = defaultdict(int)
        trainer_changes = defaultdict(float)
        trainer_matchups = defaultdict(int)

        for i in range(n):
            hid_a, pos_a, jock_a, train_a = runners[i]
            elo_a = self.get(self.horse_elo, hid_a)
            jelo_a = self.get(self.jockey_elo, jock_a)
            telo_a = self.get(self.trainer_elo, train_a)

            for j in range(i + 1, n):
                hid_b, pos_b, jock_b, train_b = runners[j]
                elo_b = self.get(self.horse_elo, hid_b)
                jelo_b = self.get(self.jockey_elo, jock_b)
                telo_b = self.get(self.trainer_elo, train_b)

                # Expected scores
                exp_a = self.expected(elo_a, elo_b)
                exp_b = 1 - exp_a
                jexp_a = self.expected(jelo_a, jelo_b)
                jexp_b = 1 - jexp_a
                texp_a = self.expected(telo_a, telo_b)
                texp_b = 1 - texp_a

                # Actual scores
                if pos_a < pos_b:
                    actual_a, actual_b = 1, 0
                elif pos_a > pos_b:
                    actual_a, actual_b = 0, 1
                else:
                    actual_a, actual_b = 0.5, 0.5

                # K-factor adjustments
                k_a = self.k * (1.3 if self.horse_races[hid_a] < 5 else 1.0)
                k_b = self.k * (1.3 if self.horse_races[hid_b] < 5 else 1.0)

                # Accumulate changes
                horse_changes[hid_a] += k_a * (actual_a - exp_a)
                horse_changes[hid_b] += k_b * (actual_b - exp_b)
                horse_matchups[hid_a] += 1
                horse_matchups[hid_b] += 1

                jockey_changes[jock_a] += self.k * 0.5 * (actual_a - jexp_a)
                jockey_changes[jock_b] += self.k * 0.5 * (actual_b - jexp_b)
                jockey_matchups[jock_a] += 1
                jockey_matchups[jock_b] += 1

                trainer_changes[train_a] += self.k * 0.5 * (actual_a - texp_a)
                trainer_changes[train_b] += self.k * 0.5 * (actual_b - texp_b)
                trainer_matchups[train_a] += 1
                trainer_matchups[train_b] += 1

        # Apply changes (scaled by sqrt of matchups like golf ELO)
        for hid in horse_changes:
            if horse_matchups[hid] > 0:
                change = horse_changes[hid] / np.sqrt(horse_matchups[hid])
                self.horse_elo[hid] = self.get(self.horse_elo, hid) + change

                # Track-specific
                key = (hid, track)
                self.horse_track_elo[key] = self.get(self.horse_track_elo, key) + change * 0.5

                # Distance-specific
                key = (hid, dist_cat)
                self.horse_dist_elo[key] = self.get(self.horse_dist_elo, key) + change * 0.5

                # Going-specific
                key = (hid, going)
                self.horse_going_elo[key] = self.get(self.horse_going_elo, key) + change * 0.5

        for jock in jockey_changes:
            if jockey_matchups[jock] > 0:
                change = jockey_changes[jock] / np.sqrt(jockey_matchups[jock])
                self.jockey_elo[jock] = self.get(self.jockey_elo, jock) + change

        for train in trainer_changes:
            if trainer_matchups[train] > 0:
                change = trainer_changes[train] / np.sqrt(trainer_matchups[train])
                self.trainer_elo[train] = self.get(self.trainer_elo, train) + change

        # Update stats
        for hid, pos, jock, train in runners:
            self.horse_races[hid] += 1
            self.jockey_races[jock] += 1
            self.trainer_races[train] += 1

            self.horse_recent_pos[hid].append(int(pos))
            if len(self.horse_recent_pos[hid]) > 10:
                self.horse_recent_pos[hid] = self.horse_recent_pos[hid][-10:]

            if pos == 1:
                self.horse_wins[hid] += 1
            if pos <= 3:
                self.horse_places[hid] += 1

            # Track stats
            self.horse_track_races[(hid, track)] += 1
            if pos == 1:
                self.horse_track_wins[(hid, track)] += 1

            # Distance stats
            self.horse_dist_races[(hid, dist_cat)] += 1
            if pos == 1:
                self.horse_dist_wins[(hid, dist_cat)] += 1

            # Going stats
            self.horse_going_races[(hid, going)] += 1
            if pos == 1:
                self.horse_going_wins[(hid, going)] += 1

            # JT combo
            self.jt_combo_races[(jock, train)] += 1
            if pos == 1:
                self.jt_combo_wins[(jock, train)] += 1


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_features(df: pd.DataFrame, elo: FastELO) -> pd.DataFrame:
    """Calculate all features for the dataset."""
    print("Calculating features...")

    features = []

    # Group by race
    races = df.groupby('race_id')
    total_races = len(races)

    for idx, (race_id, race_df) in enumerate(races):
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1:,}/{total_races:,} races ({(idx+1)/total_races*100:.1f}%)")

        race_df = race_df.copy()
        track = race_df['course_clean'].iloc[0]
        dist_cat = race_df['distance_cat'].iloc[0]
        going = race_df['going_std'].iloc[0]
        distance = race_df['distance_furlongs'].iloc[0]
        race_date = race_df['date'].iloc[0]
        field_size = len(race_df)

        # Get features for each horse BEFORE the race
        for _, row in race_df.iterrows():
            hid = row['horse_id']
            jock = row['jockey_clean']
            train = row['trainer_clean']

            # Horse ELO
            h_elo = elo.get(elo.horse_elo, hid)
            h_track_elo = elo.get(elo.horse_track_elo, (hid, track))
            h_dist_elo = elo.get(elo.horse_dist_elo, (hid, dist_cat))
            h_going_elo = elo.get(elo.horse_going_elo, (hid, going))

            # Jockey/Trainer ELO
            j_elo = elo.get(elo.jockey_elo, jock)
            t_elo = elo.get(elo.trainer_elo, train)

            # Composite ELO
            composite = (0.45 * h_elo + 0.10 * h_track_elo + 0.10 * h_dist_elo +
                        0.05 * h_going_elo + 0.15 * j_elo + 0.15 * t_elo)

            # Experience
            h_races = elo.horse_races[hid]
            j_races = elo.jockey_races[jock]
            t_races = elo.trainer_races[train]

            # Form (last 5 races)
            recent = elo.horse_recent_pos[hid][-5:] if elo.horse_recent_pos[hid] else []
            form_avg = np.mean(recent) if recent else 10
            form_wins = sum(1 for p in recent if p == 1)
            form_places = sum(1 for p in recent if p <= 3)
            form_score = sum((10 - min(p, 10)) for p in recent) / max(len(recent), 1) / 10

            # Win/place rates
            h_win_rate = elo.horse_wins[hid] / max(h_races, 1)
            h_place_rate = elo.horse_places[hid] / max(h_races, 1)

            # Track-specific performance
            track_races = elo.horse_track_races[(hid, track)]
            track_win_rate = elo.horse_track_wins[(hid, track)] / max(track_races, 1)

            # Distance-specific performance
            dist_races = elo.horse_dist_races[(hid, dist_cat)]
            dist_win_rate = elo.horse_dist_wins[(hid, dist_cat)] / max(dist_races, 1)

            # Going-specific performance
            going_races = elo.horse_going_races[(hid, going)]
            going_win_rate = elo.horse_going_wins[(hid, going)] / max(going_races, 1)

            # JT combo
            jt_races = elo.jt_combo_races[(jock, train)]
            jt_win_rate = elo.jt_combo_wins[(jock, train)] / max(jt_races, 1)

            # Days since last race
            last_date = elo.horse_last_date.get(hid)
            if last_date and race_date:
                days_since = (race_date - last_date).days
            else:
                days_since = 365

            features.append({
                'race_id': race_id,
                'horse_id': hid,
                'horse_name': row['horse_clean'],
                'jockey': jock,
                'trainer': train,
                'position': row['position'],
                'odds': row['odds_decimal'],
                'draw': row['draw'] if pd.notna(row['draw']) else field_size / 2,
                'weight': row['weight_lbs'],
                'age': row['age'] if pd.notna(row['age']) else 5,
                'field_size': field_size,
                'distance': distance,
                'track': track,
                'going': going,
                'date': race_date,

                # ELO features
                'horse_elo': h_elo,
                'horse_track_elo': h_track_elo,
                'horse_dist_elo': h_dist_elo,
                'horse_going_elo': h_going_elo,
                'jockey_elo': j_elo,
                'trainer_elo': t_elo,
                'composite_elo': composite,

                # Experience
                'horse_races': h_races,
                'jockey_races': j_races,
                'trainer_races': t_races,

                # Form
                'form_avg_pos': form_avg,
                'form_wins': form_wins,
                'form_places': form_places,
                'form_score': form_score,

                # Win rates
                'horse_win_rate': h_win_rate,
                'horse_place_rate': h_place_rate,
                'track_win_rate': track_win_rate,
                'dist_win_rate': dist_win_rate,
                'going_win_rate': going_win_rate,
                'jt_combo_win_rate': jt_win_rate,

                # Proven at track/dist/going
                'track_races': track_races,
                'dist_races': dist_races,
                'going_races': going_races,
                'jt_combo_races': jt_races,

                # Fitness
                'days_since_last': days_since,
                'days_optimal': 1 if 14 <= days_since <= 45 else 0,
            })

        # Update ELO system AFTER recording features
        elo.process_race(race_df, track, dist_cat, going)

        # Update last race dates
        for _, row in race_df.iterrows():
            elo.horse_last_date[row['horse_id']] = race_date

    return pd.DataFrame(features)


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add field-relative features (rank in field, vs field average)."""
    print("Adding relative features...")

    df = df.copy()

    # Group by race
    for col in ['horse_elo', 'composite_elo', 'jockey_elo', 'trainer_elo', 'form_score']:
        df[f'{col}_rank'] = df.groupby('race_id')[col].rank(ascending=False, method='min')
        field_avg = df.groupby('race_id')[col].transform('mean')
        df[f'{col}_vs_field'] = df[col] - field_avg

    # Odds rank (lower odds = higher rank = favorite)
    df['odds_rank'] = df.groupby('race_id')['odds'].rank(ascending=True, method='min')
    df['is_favorite'] = (df['odds_rank'] == 1).astype(int)

    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(df: pd.DataFrame, test_year: int = 2024):
    """Train the prediction model."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    # Define features
    feature_cols = [
        # ELO
        'horse_elo', 'horse_track_elo', 'horse_dist_elo', 'horse_going_elo',
        'jockey_elo', 'trainer_elo', 'composite_elo',

        # Relative ELO
        'horse_elo_rank', 'horse_elo_vs_field',
        'composite_elo_rank', 'composite_elo_vs_field',
        'jockey_elo_rank', 'trainer_elo_rank',

        # Form
        'form_avg_pos', 'form_wins', 'form_places', 'form_score',
        'form_score_rank', 'form_score_vs_field',

        # Win rates
        'horse_win_rate', 'horse_place_rate',
        'track_win_rate', 'dist_win_rate', 'going_win_rate',
        'jt_combo_win_rate',

        # Experience/Proven
        'horse_races', 'jockey_races', 'trainer_races',
        'track_races', 'dist_races', 'going_races', 'jt_combo_races',

        # Other
        'days_since_last', 'days_optimal',
        'draw', 'weight', 'age', 'field_size',
        'odds', 'odds_rank', 'is_favorite',
    ]

    # Create target
    df['target'] = (df['position'] == 1).astype(int)

    # Split by year
    df['year'] = df['date'].dt.year
    train_df = df[df['year'] < test_year].copy()
    test_df = df[df['year'] >= test_year].copy()

    print(f"\nTraining set: {len(train_df):,} horses ({train_df['race_id'].nunique():,} races)")
    print(f"Test set: {len(test_df):,} horses ({test_df['race_id'].nunique():,} races)")

    # Prepare data
    available_cols = [c for c in feature_cols if c in df.columns]
    X_train = train_df[available_cols].fillna(0)
    y_train = train_df['target']
    X_test = test_df[available_cols].fillna(0)
    y_test = test_df['target']

    print(f"\nFeatures: {len(available_cols)}")
    print(f"Train positives: {y_train.mean():.2%}")
    print(f"Test positives: {y_test.mean():.2%}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline
    baseline = y_test.mean()
    print(f"\nBaseline (random): {baseline:.2%}")

    # ========== Train XGBoost with GridSearch ==========
    print("\n--- Training XGBoost with Hyperparameter Tuning ---")

    try:
        import xgboost as xgb

        # Quick grid search
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
        }

        xgb_base = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )

        print("Running GridSearchCV...")
        grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train_scaled, y_train)

        print(f"\nBest params: {grid.best_params_}")
        print(f"Best CV AUC: {grid.best_score_:.4f}")

        xgb_model = grid.best_estimator_
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_auc = roc_auc_score(y_test, xgb_proba)

        print(f"\nXGBoost Test Results:")
        print(f"  Accuracy: {xgb_acc:.2%}")
        print(f"  AUC-ROC: {xgb_auc:.4f}")

        has_xgb = True

    except ImportError:
        print("XGBoost not available")
        has_xgb = False
        xgb_model = None

    # ========== Random Forest ==========
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)
    print(f"  Accuracy: {rf_acc:.2%}")
    print(f"  AUC-ROC: {rf_auc:.4f}")

    # ========== Gradient Boosting ==========
    print("\n--- Training Gradient Boosting ---")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=15,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    gb_proba = gb.predict_proba(X_test_scaled)[:, 1]
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_auc = roc_auc_score(y_test, gb_proba)
    print(f"  Accuracy: {gb_acc:.2%}")
    print(f"  AUC-ROC: {gb_auc:.4f}")

    # ========== Stacking Ensemble ==========
    print("\n--- Training Stacking Ensemble ---")
    estimators = [('rf', rf), ('gb', gb)]
    if has_xgb:
        estimators.append(('xgb', xgb_model))

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    stacking.fit(X_train_scaled, y_train)
    stack_pred = stacking.predict(X_test_scaled)
    stack_proba = stacking.predict_proba(X_test_scaled)[:, 1]
    stack_acc = accuracy_score(y_test, stack_pred)
    stack_auc = roc_auc_score(y_test, stack_proba)
    print(f"  Accuracy: {stack_acc:.2%}")
    print(f"  AUC-ROC: {stack_auc:.4f}")

    # ========== Race-Level Accuracy ==========
    print("\n" + "=" * 50)
    print("RACE-LEVEL PREDICTION ACCURACY")
    print("=" * 50)

    # Use best model (highest AUC)
    if has_xgb and xgb_auc >= max(rf_auc, gb_auc, stack_auc):
        best_model = xgb_model
        best_proba = xgb_proba
        best_name = "XGBoost"
    elif stack_auc >= max(rf_auc, gb_auc):
        best_model = stacking
        best_proba = stack_proba
        best_name = "Stacking"
    elif rf_auc >= gb_auc:
        best_model = rf
        best_proba = rf_proba
        best_name = "Random Forest"
    else:
        best_model = gb
        best_proba = gb_proba
        best_name = "Gradient Boosting"

    print(f"\nUsing best model: {best_name}")

    test_df = test_df.copy()
    test_df['pred_proba'] = best_proba

    # Race winner prediction
    winner_correct = 0
    top3_correct = 0
    total_races = 0

    # By field size
    results_by_size = defaultdict(lambda: {'correct': 0, 'top3': 0, 'total': 0})

    for race_id, race in test_df.groupby('race_id'):
        if len(race) < 2:
            continue

        total_races += 1
        field_size = len(race)

        # Our prediction (highest probability)
        pred_winner_idx = race['pred_proba'].idxmax()
        pred_pos = race.loc[pred_winner_idx, 'position']

        if pred_pos == 1:
            winner_correct += 1
            results_by_size[field_size]['correct'] += 1
        if pred_pos <= 3:
            top3_correct += 1
            results_by_size[field_size]['top3'] += 1
        results_by_size[field_size]['total'] += 1

    winner_acc = winner_correct / total_races
    top3_acc = top3_correct / total_races

    print(f"\nWinner Prediction: {winner_acc:.2%} ({winner_correct:,}/{total_races:,} races)")
    print(f"Top 3 Prediction: {top3_acc:.2%} ({top3_correct:,}/{total_races:,} races)")

    # Expected by field size
    print(f"\nExpected by random: ~{1/10:.2%} for 10-horse fields")
    print(f"Edge over random: +{(winner_acc - 1/10)*100:.1f}pp")

    # Results by field size
    print("\nResults by Field Size:")
    for size in sorted(results_by_size.keys()):
        data = results_by_size[size]
        if data['total'] >= 50:  # Only show meaningful samples
            acc = data['correct'] / data['total']
            t3 = data['top3'] / data['total']
            expected = 1 / size
            print(f"  {size:2d} runners: Winner {acc:.1%} (random: {expected:.1%}), Top3 {t3:.1%}, n={data['total']}")

    # ========== Feature Importance ==========
    print("\n" + "=" * 50)
    print("TOP 20 FEATURE IMPORTANCE")
    print("=" * 50)

    if has_xgb:
        importance = list(zip(available_cols, xgb_model.feature_importances_))
    else:
        importance = list(zip(available_cols, rf.feature_importances_))

    importance.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in importance[:20]:
        bar = '#' * int(imp * 100)
        print(f"  {feat:25s} {imp:.4f} {bar}")

    # ========== Save Results ==========
    print("\n" + "=" * 50)
    print("SAVING MODEL")
    print("=" * 50)

    save_data = {
        'model': best_model,
        'model_name': best_name,
        'scaler': scaler,
        'feature_cols': available_cols,
        'metrics': {
            'winner_accuracy': winner_acc,
            'top3_accuracy': top3_acc,
            'auc': max(xgb_auc if has_xgb else 0, rf_auc, gb_auc, stack_auc),
        }
    }

    with open('/Users/lewisjackson/projects/horse-racing-predictor/models/best_model.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Model saved to models/best_model.pkl")

    return best_model, scaler, available_cols, test_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("HORSE RACING PREDICTOR - REAL DATA TRAINING")
    print("=" * 70)

    # Load data
    data_path = '/Users/lewisjackson/projects/horse-racing-predictor/data/raw/raceform.csv'
    df = load_and_clean_data(data_path)

    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique horses: {df['horse_id'].nunique():,}")
    print(f"Unique jockeys: {df['jockey_clean'].nunique():,}")
    print(f"Unique trainers: {df['trainer_clean'].nunique():,}")
    print(f"Unique tracks: {df['course_clean'].nunique():,}")

    # Initialize ELO system
    print("\n" + "=" * 50)
    print("BUILDING ELO RATINGS")
    print("=" * 50)
    elo = FastELO(k_factor=32)

    # Calculate features (this also builds ELO)
    features_df = calculate_features(df, elo)

    # Add relative features
    features_df = add_relative_features(features_df)

    # Show top horses/jockeys/trainers by ELO
    print("\n--- Top 20 Horses by ELO ---")
    top_horses = sorted(elo.horse_elo.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (h, e) in enumerate(top_horses, 1):
        print(f"  {i:2d}. {h[:30]:30s} {e:.0f}")

    print("\n--- Top 10 Jockeys by ELO ---")
    top_jockeys = sorted(elo.jockey_elo.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (j, e) in enumerate(top_jockeys, 1):
        print(f"  {i:2d}. {j[:25]:25s} {e:.0f}")

    print("\n--- Top 10 Trainers by ELO ---")
    top_trainers = sorted(elo.trainer_elo.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (t, e) in enumerate(top_trainers, 1):
        print(f"  {i:2d}. {t[:25]:25s} {e:.0f}")

    # Train model
    model, scaler, feature_cols, test_df = train_model(features_df, test_year=2024)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nYour horse racing predictor is ready!")
    print("Model saved to: models/best_model.pkl")


if __name__ == "__main__":
    main()
