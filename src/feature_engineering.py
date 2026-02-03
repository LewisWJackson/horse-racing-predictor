"""
Feature Engineering for Horse Racing Prediction
================================================
Creates comprehensive features for predicting horse race outcomes.

Based on the same methodology used in Tennis, Golf, and UFC predictors:
- ELO-based features (horse, jockey, trainer)
- Rolling performance statistics
- Head-to-head records
- Physical/condition features
- Historical performance at track/distance/going

Key Features for Horse Racing:
1. ELO Ratings (horse, jockey, trainer, track-specific, distance-specific, going-specific)
2. Recent Form (last 5/10 races, win rate, place rate)
3. Class History (performance at different race classes)
4. Speed Figures (if available)
5. Weight/Burden (carried weight vs typical weight)
6. Days Since Last Race (fitness vs freshness)
7. Barrier/Draw (starting position effect)
8. Distance Performance (proven at distance)
9. Going Performance (proven on track condition)
10. Track Record (wins/places at specific track)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from horse_elo import HorseELO, build_horse_elo
import warnings
warnings.filterwarnings('ignore')


class HorseRacingFeatureEngineer:
    """
    Feature engineering for horse racing prediction.

    Similar to your Tennis and UFC feature engineering, but adapted for
    the unique aspects of horse racing (multi-runner fields, jockey/trainer
    combinations, weight assignments, etc.)
    """

    def __init__(self, k_factor: int = 32):
        """Initialize the feature engineer."""
        self.elo_system = HorseELO(k_factor=k_factor)

        # Track horse statistics
        self.horse_stats: Dict[str, Dict] = defaultdict(lambda: {
            'races': 0,
            'wins': 0,
            'places': 0,  # Top 3
            'recent_positions': [],  # Last 10 positions
            'recent_class': [],  # Last 10 race classes
            'track_stats': defaultdict(lambda: {'races': 0, 'wins': 0, 'places': 0}),
            'distance_stats': defaultdict(lambda: {'races': 0, 'wins': 0, 'places': 0}),
            'going_stats': defaultdict(lambda: {'races': 0, 'wins': 0, 'places': 0}),
            'last_race_date': None,
            'peak_elo': 1500,
            'current_streak': 0,
            'weight_carried': [],  # Recent weights for tracking
        })

        # Track jockey statistics
        self.jockey_stats: Dict[str, Dict] = defaultdict(lambda: {
            'races': 0,
            'wins': 0,
            'places': 0,
            'strike_rate': 0,
            'trainer_combos': defaultdict(lambda: {'races': 0, 'wins': 0}),
        })

        # Track trainer statistics
        self.trainer_stats: Dict[str, Dict] = defaultdict(lambda: {
            'races': 0,
            'wins': 0,
            'places': 0,
            'strike_rate': 0,
            'jockey_combos': defaultdict(lambda: {'races': 0, 'wins': 0}),
        })

        # Track jockey-trainer combinations
        self.jt_combos: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
            'races': 0,
            'wins': 0,
            'places': 0,
        })

        # Head-to-head records (horses that have raced against each other)
        self.h2h: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
            'races': 0,
            'wins': 0,  # First horse beat second
        })

    def calculate_recent_form(self, positions: List[int], n: int = 5) -> Dict[str, float]:
        """
        Calculate form metrics from recent positions.

        Returns:
            Dict with win_rate, place_rate, avg_position, form_score
        """
        if not positions:
            return {
                'win_rate': 0.0,
                'place_rate': 0.0,
                'avg_position': 10.0,  # Default poor position
                'form_score': 0.0,
            }

        recent = positions[-n:] if len(positions) >= n else positions

        wins = sum(1 for p in recent if p == 1)
        places = sum(1 for p in recent if p <= 3)

        win_rate = wins / len(recent)
        place_rate = places / len(recent)
        avg_pos = np.mean(recent)

        # Form score: weighted recent performance (more recent = higher weight)
        weights = [1 + i * 0.2 for i in range(len(recent))]  # [1, 1.2, 1.4, 1.6, 1.8] for 5 races
        form_score = sum((10 - min(p, 10)) * w for p, w in zip(recent, weights)) / sum(weights)
        form_score = form_score / 10  # Normalize to 0-1

        return {
            'win_rate': win_rate,
            'place_rate': place_rate,
            'avg_position': avg_pos,
            'form_score': form_score,
        }

    def calculate_features_for_race(self, race_df: pd.DataFrame, race_info: Dict) -> pd.DataFrame:
        """
        Calculate features for all horses in a single race BEFORE the race happens.

        This is crucial - we must only use information available before the race.

        Args:
            race_df: DataFrame with horses in the race
            race_info: Dict with race metadata

        Returns:
            DataFrame with features for each horse
        """
        track = race_info.get('track', 'Unknown')
        distance = race_info.get('distance_furlongs', 8)
        going = race_info.get('going', 'Good')
        race_class = race_info.get('race_class', 'Unknown')
        race_date = race_info.get('date')
        field_size = len(race_df)

        features_list = []

        for _, row in race_df.iterrows():
            horse_id = row['horse_id']
            horse_name = row.get('horse_name', horse_id)
            jockey = row.get('jockey', 'Unknown')
            trainer = row.get('trainer', 'Unknown')
            position = row.get('position', None)
            weight = row.get('weight', None)
            draw = row.get('draw', None)
            age = row.get('age', None)
            odds = row.get('odds', None)

            # Get current stats BEFORE this race
            h_stats = self.horse_stats[horse_id]
            j_stats = self.jockey_stats[jockey]
            t_stats = self.trainer_stats[trainer]
            jt_combo = self.jt_combos[(jockey, trainer)]

            # ===== ELO Features =====
            horse_elo = self.elo_system.get_horse_rating(horse_id)
            horse_track_elo = self.elo_system.get_horse_track_rating(horse_id, track)
            horse_dist_elo = self.elo_system.get_horse_distance_rating(horse_id, distance)
            horse_going_elo = self.elo_system.get_horse_going_rating(horse_id, going)

            jockey_elo = self.elo_system.get_jockey_rating(jockey)
            jockey_track_elo = self.elo_system.get_jockey_track_rating(jockey, track)

            trainer_elo = self.elo_system.get_trainer_rating(trainer)
            trainer_track_elo = self.elo_system.get_trainer_track_rating(trainer, track)

            composite_elo = self.elo_system.get_composite_rating(
                horse_id, jockey, trainer, track, distance, going
            )

            # ===== Recent Form Features =====
            form_5 = self.calculate_recent_form(h_stats['recent_positions'], n=5)
            form_10 = self.calculate_recent_form(h_stats['recent_positions'], n=10)

            # ===== Experience Features =====
            horse_races = h_stats['races']
            jockey_races = j_stats['races']
            trainer_races = t_stats['races']

            # ===== Win/Place Rates =====
            horse_win_rate = h_stats['wins'] / h_stats['races'] if h_stats['races'] > 0 else 0
            horse_place_rate = h_stats['places'] / h_stats['races'] if h_stats['races'] > 0 else 0

            jockey_strike_rate = j_stats['wins'] / j_stats['races'] if j_stats['races'] > 0 else 0
            trainer_strike_rate = t_stats['wins'] / t_stats['races'] if t_stats['races'] > 0 else 0

            # ===== Track/Distance/Going Performance =====
            track_stats = h_stats['track_stats'][track]
            track_win_rate = track_stats['wins'] / track_stats['races'] if track_stats['races'] > 0 else 0
            track_place_rate = track_stats['places'] / track_stats['races'] if track_stats['races'] > 0 else 0

            dist_cat = self.elo_system.categorize_distance(distance)
            dist_stats = h_stats['distance_stats'][dist_cat]
            dist_win_rate = dist_stats['wins'] / dist_stats['races'] if dist_stats['races'] > 0 else 0
            dist_place_rate = dist_stats['places'] / dist_stats['races'] if dist_stats['races'] > 0 else 0

            going_cat = self.elo_system.categorize_going(going)
            going_stats = h_stats['going_stats'][going_cat]
            going_win_rate = going_stats['wins'] / going_stats['races'] if going_stats['races'] > 0 else 0
            going_place_rate = going_stats['places'] / going_stats['races'] if going_stats['races'] > 0 else 0

            # ===== Jockey-Trainer Combo =====
            jt_races = jt_combo['races']
            jt_win_rate = jt_combo['wins'] / jt_combo['races'] if jt_combo['races'] > 0 else 0

            # ===== Days Since Last Race =====
            if h_stats['last_race_date'] and race_date:
                days_since_last = (pd.to_datetime(race_date) - pd.to_datetime(h_stats['last_race_date'])).days
            else:
                days_since_last = 365  # First timer

            # ===== Current Streak =====
            current_streak = h_stats['current_streak']

            # ===== Peak ELO Comparison =====
            vs_peak_elo = horse_elo - h_stats['peak_elo']

            # ===== Draw/Barrier Analysis =====
            # High draws can be disadvantageous in certain conditions
            draw_adjusted = draw if pd.notna(draw) else field_size // 2

            # ===== Weight Analysis =====
            # Compare to recent weights carried
            weight_val = weight if pd.notna(weight) else 0
            recent_weights = h_stats['weight_carried'][-5:] if h_stats['weight_carried'] else []
            avg_recent_weight = np.mean(recent_weights) if recent_weights else weight_val
            weight_vs_recent = weight_val - avg_recent_weight if weight_val > 0 else 0

            feature_dict = {
                # Identifiers (for reference, not used as features)
                'horse_id': horse_id,
                'horse_name': horse_name,
                'jockey': jockey,
                'trainer': trainer,
                'position': position,  # Target variable (only known after race)

                # ELO Features
                'horse_elo': horse_elo,
                'horse_track_elo': horse_track_elo,
                'horse_distance_elo': horse_dist_elo,
                'horse_going_elo': horse_going_elo,
                'jockey_elo': jockey_elo,
                'jockey_track_elo': jockey_track_elo,
                'trainer_elo': trainer_elo,
                'trainer_track_elo': trainer_track_elo,
                'composite_elo': composite_elo,

                # Recent Form
                'form_win_rate_5': form_5['win_rate'],
                'form_place_rate_5': form_5['place_rate'],
                'form_avg_pos_5': form_5['avg_position'],
                'form_score_5': form_5['form_score'],
                'form_win_rate_10': form_10['win_rate'],
                'form_place_rate_10': form_10['place_rate'],
                'form_avg_pos_10': form_10['avg_position'],
                'form_score_10': form_10['form_score'],

                # Experience
                'horse_races': horse_races,
                'jockey_races': jockey_races,
                'trainer_races': trainer_races,

                # Win/Place Rates
                'horse_win_rate': horse_win_rate,
                'horse_place_rate': horse_place_rate,
                'jockey_strike_rate': jockey_strike_rate,
                'trainer_strike_rate': trainer_strike_rate,

                # Track/Distance/Going Specific
                'track_races': track_stats['races'],
                'track_win_rate': track_win_rate,
                'track_place_rate': track_place_rate,
                'dist_races': dist_stats['races'],
                'dist_win_rate': dist_win_rate,
                'dist_place_rate': dist_place_rate,
                'going_races': going_stats['races'],
                'going_win_rate': going_win_rate,
                'going_place_rate': going_place_rate,

                # Jockey-Trainer Combo
                'jt_combo_races': jt_races,
                'jt_combo_win_rate': jt_win_rate,

                # Freshness/Fitness
                'days_since_last': days_since_last,
                'days_since_last_optimal': 1 if 14 <= days_since_last <= 45 else 0,

                # Streak
                'current_streak': current_streak,

                # Peak comparison
                'vs_peak_elo': vs_peak_elo,

                # Draw/Weight
                'draw': draw_adjusted,
                'draw_pct': draw_adjusted / field_size if field_size > 0 else 0.5,
                'weight': weight_val,
                'weight_vs_recent': weight_vs_recent,

                # Age
                'age': age if pd.notna(age) else 4,  # Default to typical racing age

                # Odds (market indicator)
                'odds': odds if pd.notna(odds) else 20,

                # Field context
                'field_size': field_size,

                # Race metadata (for analysis)
                'track': track,
                'distance': distance,
                'going': going,
                'race_class': race_class,
                'race_date': race_date,
            }

            features_list.append(feature_dict)

        return pd.DataFrame(features_list)

    def update_stats_after_race(self, race_df: pd.DataFrame, race_info: Dict) -> None:
        """
        Update all statistics AFTER a race has completed.

        This must be called after calculate_features_for_race to maintain
        temporal consistency (features from before, then update after).
        """
        track = race_info.get('track', 'Unknown')
        distance = race_info.get('distance_furlongs', 8)
        going = race_info.get('going', 'Good')
        race_date = race_info.get('date')

        dist_cat = self.elo_system.categorize_distance(distance)
        going_cat = self.elo_system.categorize_going(going)

        # First, update ELO system
        self.elo_system.process_race(race_df, race_info)

        # Then update all other statistics
        for _, row in race_df.iterrows():
            horse_id = row['horse_id']
            jockey = row.get('jockey', 'Unknown')
            trainer = row.get('trainer', 'Unknown')
            position = row.get('position')
            weight = row.get('weight')

            if pd.isna(position) or position <= 0:
                continue

            h_stats = self.horse_stats[horse_id]
            j_stats = self.jockey_stats[jockey]
            t_stats = self.trainer_stats[trainer]
            jt_combo = self.jt_combos[(jockey, trainer)]

            # Update horse stats
            h_stats['races'] += 1
            if position == 1:
                h_stats['wins'] += 1
            if position <= 3:
                h_stats['places'] += 1

            h_stats['recent_positions'].append(int(position))
            h_stats['last_race_date'] = race_date

            # Update peak ELO
            current_elo = self.elo_system.get_horse_rating(horse_id)
            if current_elo > h_stats['peak_elo']:
                h_stats['peak_elo'] = current_elo

            # Update streak
            if position == 1:
                h_stats['current_streak'] = max(1, h_stats['current_streak'] + 1)
            else:
                h_stats['current_streak'] = min(-1, h_stats['current_streak'] - 1) if h_stats['current_streak'] <= 0 else -1

            # Update weight tracking
            if pd.notna(weight):
                h_stats['weight_carried'].append(weight)

            # Update track/distance/going stats
            h_stats['track_stats'][track]['races'] += 1
            if position == 1:
                h_stats['track_stats'][track]['wins'] += 1
            if position <= 3:
                h_stats['track_stats'][track]['places'] += 1

            h_stats['distance_stats'][dist_cat]['races'] += 1
            if position == 1:
                h_stats['distance_stats'][dist_cat]['wins'] += 1
            if position <= 3:
                h_stats['distance_stats'][dist_cat]['places'] += 1

            h_stats['going_stats'][going_cat]['races'] += 1
            if position == 1:
                h_stats['going_stats'][going_cat]['wins'] += 1
            if position <= 3:
                h_stats['going_stats'][going_cat]['places'] += 1

            # Update jockey stats
            j_stats['races'] += 1
            if position == 1:
                j_stats['wins'] += 1
            if position <= 3:
                j_stats['places'] += 1

            # Update trainer stats
            t_stats['races'] += 1
            if position == 1:
                t_stats['wins'] += 1
            if position <= 3:
                t_stats['places'] += 1

            # Update jockey-trainer combo
            jt_combo['races'] += 1
            if position == 1:
                jt_combo['wins'] += 1
            if position <= 3:
                jt_combo['places'] += 1

        # Update head-to-head records
        runners = race_df[race_df['position'].notna() & (race_df['position'] > 0)]
        for _, row_a in runners.iterrows():
            for _, row_b in runners.iterrows():
                if row_a['horse_id'] >= row_b['horse_id']:
                    continue
                h2h_key = (row_a['horse_id'], row_b['horse_id'])
                self.h2h[h2h_key]['races'] += 1
                if row_a['position'] < row_b['position']:
                    self.h2h[h2h_key]['wins'] += 1

    def process_all_races(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all races chronologically and generate features.

        Args:
            df: Full race dataset with required columns

        Returns:
            DataFrame with all features for all horses in all races
        """
        print("Processing races and generating features...")

        # Sort by date
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'race_id'])

        # Get unique races
        races = df.groupby('race_id').agg({
            'date': 'first',
            'track': 'first',
            'race_name': 'first',
            'race_class': 'first',
            'distance_furlongs': 'first',
            'going': 'first',
        }).reset_index().sort_values('date')

        all_features = []

        for idx, (_, row) in enumerate(races.iterrows()):
            race_id = row['race_id']

            race_info = {
                'track': row['track'],
                'distance_furlongs': row['distance_furlongs'],
                'going': row['going'],
                'race_class': row['race_class'],
                'date': row['date'],
                'race_name': row['race_name'],
            }

            race_df = df[df['race_id'] == race_id]

            # Calculate features BEFORE the race
            features = self.calculate_features_for_race(race_df, race_info)
            features['race_id'] = race_id
            all_features.append(features)

            # Update stats AFTER the race
            self.update_stats_after_race(race_df, race_info)

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(races)} races")

        result_df = pd.concat(all_features, ignore_index=True)
        print(f"Generated features for {len(result_df)} horse-race combinations")

        return result_df

    def get_elo_system(self) -> HorseELO:
        """Return the ELO system for inspection."""
        return self.elo_system


def calculate_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features relative to the field (for each race).

    This is similar to what was done in Golf (ELO vs field, rank in field).
    """
    df = df.copy()

    # Group by race
    grouped = df.groupby('race_id')

    # Calculate field averages and ranks
    elo_cols = ['horse_elo', 'composite_elo', 'jockey_elo', 'trainer_elo']

    for col in elo_cols:
        # Rank in field (1 = highest ELO)
        df[f'{col}_rank'] = grouped[col].rank(ascending=False, method='min')

        # vs field average
        field_avg = grouped[col].transform('mean')
        df[f'{col}_vs_field'] = df[col] - field_avg

        # vs field best
        field_max = grouped[col].transform('max')
        df[f'{col}_vs_best'] = df[col] - field_max

    # Relative form
    df['form_score_5_rank'] = grouped['form_score_5'].rank(ascending=False, method='min')

    # Relative experience
    df['experience_rank'] = grouped['horse_races'].rank(ascending=False, method='min')

    return df


def prepare_training_data(feature_df: pd.DataFrame,
                         test_date: str = '2024-01-01',
                         target: str = 'win') -> Tuple:
    """
    Prepare features and labels for training.

    Args:
        feature_df: DataFrame with all features
        test_date: Date to split train/test
        target: 'win' (binary: won or not) or 'place' (binary: top 3 or not)

    Returns:
        X_train, y_train, X_test, y_test, train_df, test_df
    """
    print("\n--- Preparing Training Data ---")

    df = feature_df.copy()

    # Create target variable
    if target == 'win':
        df['target'] = (df['position'] == 1).astype(int)
    elif target == 'place':
        df['target'] = (df['position'] <= 3).astype(int)
    else:
        raise ValueError("target must be 'win' or 'place'")

    # Remove races where we don't have position (didn't finish, etc.)
    df = df[df['position'].notna() & (df['position'] > 0)]

    # Calculate relative features
    df = calculate_relative_features(df)

    # Features to use
    feature_cols = [
        # ELO features
        'horse_elo', 'horse_track_elo', 'horse_distance_elo', 'horse_going_elo',
        'jockey_elo', 'jockey_track_elo', 'trainer_elo', 'trainer_track_elo',
        'composite_elo',

        # Relative ELO
        'horse_elo_rank', 'horse_elo_vs_field', 'horse_elo_vs_best',
        'composite_elo_rank', 'composite_elo_vs_field',
        'jockey_elo_rank', 'trainer_elo_rank',

        # Form
        'form_win_rate_5', 'form_place_rate_5', 'form_avg_pos_5', 'form_score_5',
        'form_win_rate_10', 'form_score_10',
        'form_score_5_rank',

        # Experience
        'horse_races', 'jockey_races', 'trainer_races',
        'experience_rank',

        # Win rates
        'horse_win_rate', 'horse_place_rate',
        'jockey_strike_rate', 'trainer_strike_rate',

        # Track/Distance/Going
        'track_races', 'track_win_rate', 'track_place_rate',
        'dist_races', 'dist_win_rate', 'dist_place_rate',
        'going_races', 'going_win_rate', 'going_place_rate',

        # Jockey-Trainer combo
        'jt_combo_races', 'jt_combo_win_rate',

        # Freshness
        'days_since_last', 'days_since_last_optimal',

        # Other
        'current_streak', 'vs_peak_elo',
        'draw', 'draw_pct', 'weight', 'weight_vs_recent',
        'age', 'field_size',
    ]

    # Split by date
    df['race_date'] = pd.to_datetime(df['race_date'])
    train_df = df[df['race_date'] < test_date].copy()
    test_df = df[df['race_date'] >= test_date].copy()

    print(f"Training set: {len(train_df)} horses ({train_df['race_id'].nunique()} races)")
    print(f"Test set: {len(test_df)} horses ({test_df['race_id'].nunique()} races)")

    # Prepare X and y
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['target']

    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"Train class distribution: {y_train.mean():.2%} positive")
    print(f"Test class distribution: {y_test.mean():.2%} positive")

    return X_train, y_train, X_test, y_test, train_df, test_df, feature_cols


if __name__ == "__main__":
    print("Horse Racing Feature Engineering Module")
    print("=" * 60)
    print("\nThis module provides feature engineering for horse racing prediction.")
    print("\nUsage:")
    print("  from feature_engineering import HorseRacingFeatureEngineer")
    print("  fe = HorseRacingFeatureEngineer()")
    print("  features_df = fe.process_all_races(race_data)")
