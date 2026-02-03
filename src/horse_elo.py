"""
Horse Racing ELO Rating System
==============================
Adapts the ELO system for horse racing, similar to how you did it for Golf.

Key Adaptations for Horse Racing:
- Multi-horse competition (like golf tournaments)
- Track-specific ELO (like surface-specific in tennis)
- Distance-specific ELO (sprints vs routes vs marathons)
- Going/track condition specific performance
- Jockey and Trainer ELO (unique to horse racing)

The "virtual matchup" approach from Golf ELO is perfect here:
- Each horse plays a "virtual race" against every other horse
- If you finish higher, you "beat" them; if lower, you "lost"
- ELO adjustments are averaged across all virtual matchups
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class HorseELO:
    """
    ELO Rating System for Horse Racing.

    Tracks multiple types of ratings:
    - Overall ELO for horses
    - Track-specific ELO (e.g., Ascot, Cheltenham, Newmarket)
    - Distance-specific ELO (Sprint: <7f, Mile: 7-9f, Route: 10-12f, Long: >12f)
    - Going-specific ELO (Good, Soft, Heavy, Firm)
    - Jockey ELO
    - Trainer ELO
    """

    def __init__(self, k_factor: int = 32, initial_elo: int = 1500):
        """
        Initialize the Horse Racing ELO system.

        Args:
            k_factor: How much a single result affects rating (higher = more volatile)
            initial_elo: Starting ELO for new horses/jockeys/trainers
        """
        self.k_factor = k_factor
        self.initial_elo = initial_elo

        # Horse ratings - use regular dicts for pickle compatibility
        self.horse_ratings: Dict[str, float] = {}
        self.horse_track_ratings: Dict[Tuple[str, str], float] = {}
        self.horse_distance_ratings: Dict[Tuple[str, str], float] = {}
        self.horse_going_ratings: Dict[Tuple[str, str], float] = {}

        # Jockey ratings
        self.jockey_ratings: Dict[str, float] = {}
        self.jockey_track_ratings: Dict[Tuple[str, str], float] = {}

        # Trainer ratings
        self.trainer_ratings: Dict[str, float] = {}
        self.trainer_track_ratings: Dict[Tuple[str, str], float] = {}

        # History tracking
        self.horse_history: List[Dict] = []
        self.jockey_history: List[Dict] = []
        self.trainer_history: List[Dict] = []

        # Race counts for experience tracking
        self.horse_race_counts: Dict[str, int] = defaultdict(int)
        self.jockey_race_counts: Dict[str, int] = defaultdict(int)
        self.trainer_race_counts: Dict[str, int] = defaultdict(int)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected probability of A beating B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def categorize_distance(self, furlongs: float) -> str:
        """
        Categorize race distance into Sprint/Mile/Route/Long.

        Args:
            furlongs: Distance in furlongs (8 furlongs = 1 mile)
        """
        if furlongs < 7:
            return 'Sprint'
        elif furlongs < 10:
            return 'Mile'
        elif furlongs < 14:
            return 'Route'
        else:
            return 'Marathon'

    def categorize_going(self, going: str) -> str:
        """
        Standardize going/track condition descriptions.

        Args:
            going: Raw going description from data
        """
        if pd.isna(going):
            return 'Good'

        going_lower = str(going).lower()

        if any(x in going_lower for x in ['heavy', 'soft', 'yielding']):
            return 'Soft'
        elif any(x in going_lower for x in ['firm', 'hard', 'fast']):
            return 'Firm'
        elif any(x in going_lower for x in ['good to soft', 'good-soft']):
            return 'Good_to_Soft'
        elif any(x in going_lower for x in ['good to firm', 'good-firm']):
            return 'Good_to_Firm'
        else:
            return 'Good'

    def get_k_factor(self, entity_type: str, entity_id: str, race_class: Optional[str] = None) -> float:
        """
        Calculate dynamic K-factor based on experience and race importance.

        New horses/jockeys get higher K-factor (ratings change faster)
        Group/Stakes races have higher K-factor (more meaningful)
        """
        k = self.k_factor

        # Get race count based on entity type
        if entity_type == 'horse':
            count = self.horse_race_counts[entity_id]
        elif entity_type == 'jockey':
            count = self.jockey_race_counts[entity_id]
        else:  # trainer
            count = self.trainer_race_counts[entity_id]

        # New entities: higher K
        if count < 3:
            k *= 1.5
        elif count < 8:
            k *= 1.2
        elif count > 50:
            k *= 0.9  # More stable ratings for experienced

        # Race class modifier
        if race_class:
            class_lower = str(race_class).lower()
            if any(x in class_lower for x in ['group 1', 'grade 1', 'g1']):
                k *= 1.5
            elif any(x in class_lower for x in ['group 2', 'grade 2', 'g2']):
                k *= 1.3
            elif any(x in class_lower for x in ['group 3', 'grade 3', 'g3', 'listed']):
                k *= 1.2
            elif any(x in class_lower for x in ['stakes', 'pattern']):
                k *= 1.1

        return k

    def process_race(self, race_df: pd.DataFrame, race_info: Dict) -> None:
        """
        Process a single race and update all ELO ratings.

        Args:
            race_df: DataFrame with columns [horse_id, horse_name, position, jockey, trainer]
            race_info: Dict with race metadata (track, distance, going, race_class, date)
        """
        if race_df.empty or len(race_df) < 2:
            return

        track = race_info.get('track', 'Unknown')
        distance_furlongs = race_info.get('distance_furlongs', 8)
        going = race_info.get('going', 'Good')
        race_class = race_info.get('race_class', None)
        race_date = race_info.get('date', None)
        race_name = race_info.get('race_name', 'Unknown')

        # Categorize distance and going
        distance_cat = self.categorize_distance(distance_furlongs)
        going_cat = self.categorize_going(going)

        # Filter to horses with valid positions
        valid = race_df[race_df['position'].notna() & (race_df['position'] > 0)].copy()
        if len(valid) < 2:
            return

        runners = valid[['horse_id', 'horse_name', 'position', 'jockey', 'trainer']].values.tolist()

        # ===== HORSE ELO Updates =====
        horse_changes = defaultdict(float)
        horse_matchup_counts = defaultdict(int)

        for i, (hid_a, name_a, pos_a, jock_a, train_a) in enumerate(runners):
            for j, (hid_b, name_b, pos_b, jock_b, train_b) in enumerate(runners):
                if i >= j:
                    continue

                # Get current overall ratings
                elo_a = self.horse_ratings[hid_a]
                elo_b = self.horse_ratings[hid_b]

                # Expected scores
                exp_a = self.expected_score(elo_a, elo_b)
                exp_b = 1 - exp_a

                # Actual scores (1 = win, 0.5 = dead heat, 0 = loss)
                if pos_a < pos_b:  # A finished higher (lower pos = better)
                    actual_a, actual_b = 1, 0
                elif pos_a > pos_b:
                    actual_a, actual_b = 0, 1
                else:  # Dead heat
                    actual_a, actual_b = 0.5, 0.5

                # Get K-factors
                k_a = self.get_k_factor('horse', hid_a, race_class)
                k_b = self.get_k_factor('horse', hid_b, race_class)

                # Accumulate changes
                horse_changes[hid_a] += k_a * (actual_a - exp_a)
                horse_changes[hid_b] += k_b * (actual_b - exp_b)
                horse_matchup_counts[hid_a] += 1
                horse_matchup_counts[hid_b] += 1

        # Apply horse ELO changes
        for hid in horse_changes:
            if horse_matchup_counts[hid] > 0:
                # Scale down based on number of matchups (like Golf ELO)
                change = horse_changes[hid] / np.sqrt(horse_matchup_counts[hid])

                old_rating = self.horse_ratings[hid]
                self.horse_ratings[hid] += change

                # Update track-specific
                track_key = (hid, track)
                self.horse_track_ratings[track_key] += change * 0.5

                # Update distance-specific
                dist_key = (hid, distance_cat)
                self.horse_distance_ratings[dist_key] += change * 0.5

                # Update going-specific
                going_key = (hid, going_cat)
                self.horse_going_ratings[going_key] += change * 0.5

                # Increment race count
                self.horse_race_counts[hid] += 1

                # Record history
                horse_name = valid[valid['horse_id'] == hid]['horse_name'].iloc[0]
                pos = valid[valid['horse_id'] == hid]['position'].iloc[0]

                self.horse_history.append({
                    'horse_id': hid,
                    'horse_name': horse_name,
                    'race_name': race_name,
                    'track': track,
                    'date': race_date,
                    'position': pos,
                    'field_size': len(runners),
                    'old_elo': old_rating,
                    'new_elo': self.horse_ratings[hid],
                    'elo_change': change,
                    'track_elo': self.horse_track_ratings[track_key],
                    'distance_elo': self.horse_distance_ratings[dist_key],
                    'going_elo': self.horse_going_ratings[going_key],
                })

        # ===== JOCKEY ELO Updates =====
        self._update_jockey_elo(runners, track, race_class, race_date, race_name)

        # ===== TRAINER ELO Updates =====
        self._update_trainer_elo(runners, track, race_class, race_date, race_name)

    def _update_jockey_elo(self, runners: List, track: str, race_class: str,
                           race_date, race_name: str) -> None:
        """Update jockey ELO ratings."""
        jockey_changes = defaultdict(float)
        jockey_matchup_counts = defaultdict(int)

        for i, (_, _, pos_a, jock_a, _) in enumerate(runners):
            if pd.isna(jock_a):
                continue
            for j, (_, _, pos_b, jock_b, _) in enumerate(runners):
                if i >= j or pd.isna(jock_b):
                    continue

                elo_a = self.jockey_ratings[jock_a]
                elo_b = self.jockey_ratings[jock_b]

                exp_a = self.expected_score(elo_a, elo_b)
                exp_b = 1 - exp_a

                if pos_a < pos_b:
                    actual_a, actual_b = 1, 0
                elif pos_a > pos_b:
                    actual_a, actual_b = 0, 1
                else:
                    actual_a, actual_b = 0.5, 0.5

                k_a = self.get_k_factor('jockey', jock_a, race_class)
                k_b = self.get_k_factor('jockey', jock_b, race_class)

                jockey_changes[jock_a] += k_a * (actual_a - exp_a)
                jockey_changes[jock_b] += k_b * (actual_b - exp_b)
                jockey_matchup_counts[jock_a] += 1
                jockey_matchup_counts[jock_b] += 1

        for jockey in jockey_changes:
            if jockey_matchup_counts[jockey] > 0:
                change = jockey_changes[jockey] / np.sqrt(jockey_matchup_counts[jockey])
                old_rating = self.jockey_ratings[jockey]
                self.jockey_ratings[jockey] += change
                self.jockey_track_ratings[(jockey, track)] += change * 0.5
                self.jockey_race_counts[jockey] += 1

    def _update_trainer_elo(self, runners: List, track: str, race_class: str,
                            race_date, race_name: str) -> None:
        """Update trainer ELO ratings."""
        trainer_changes = defaultdict(float)
        trainer_matchup_counts = defaultdict(int)

        for i, (_, _, pos_a, _, train_a) in enumerate(runners):
            if pd.isna(train_a):
                continue
            for j, (_, _, pos_b, _, train_b) in enumerate(runners):
                if i >= j or pd.isna(train_b):
                    continue

                elo_a = self.trainer_ratings[train_a]
                elo_b = self.trainer_ratings[train_b]

                exp_a = self.expected_score(elo_a, elo_b)
                exp_b = 1 - exp_a

                if pos_a < pos_b:
                    actual_a, actual_b = 1, 0
                elif pos_a > pos_b:
                    actual_a, actual_b = 0, 1
                else:
                    actual_a, actual_b = 0.5, 0.5

                k_a = self.get_k_factor('trainer', train_a, race_class)
                k_b = self.get_k_factor('trainer', train_b, race_class)

                trainer_changes[train_a] += k_a * (actual_a - exp_a)
                trainer_changes[train_b] += k_b * (actual_b - exp_b)
                trainer_matchup_counts[train_a] += 1
                trainer_matchup_counts[train_b] += 1

        for trainer in trainer_changes:
            if trainer_matchup_counts[trainer] > 0:
                change = trainer_changes[trainer] / np.sqrt(trainer_matchup_counts[trainer])
                self.trainer_ratings[trainer] += change
                self.trainer_track_ratings[(trainer, track)] += change * 0.5
                self.trainer_race_counts[trainer] += 1

    # ===== Getter Methods =====

    def get_horse_rating(self, horse_id: str) -> float:
        """Get horse's current overall ELO."""
        return self.horse_ratings.get(horse_id, self.initial_elo)

    def get_horse_track_rating(self, horse_id: str, track: str) -> float:
        """Get horse's track-specific ELO."""
        return self.horse_track_ratings.get((horse_id, track), self.initial_elo)

    def get_horse_distance_rating(self, horse_id: str, distance_furlongs: float) -> float:
        """Get horse's distance-specific ELO."""
        distance_cat = self.categorize_distance(distance_furlongs)
        return self.horse_distance_ratings.get((horse_id, distance_cat), self.initial_elo)

    def get_horse_going_rating(self, horse_id: str, going: str) -> float:
        """Get horse's going-specific ELO."""
        going_cat = self.categorize_going(going)
        return self.horse_going_ratings.get((horse_id, going_cat), self.initial_elo)

    def get_jockey_rating(self, jockey: str) -> float:
        """Get jockey's current overall ELO."""
        return self.jockey_ratings.get(jockey, self.initial_elo)

    def get_jockey_track_rating(self, jockey: str, track: str) -> float:
        """Get jockey's track-specific ELO."""
        return self.jockey_track_ratings.get((jockey, track), self.initial_elo)

    def get_trainer_rating(self, trainer: str) -> float:
        """Get trainer's current overall ELO."""
        return self.trainer_ratings.get(trainer, self.initial_elo)

    def get_trainer_track_rating(self, trainer: str, track: str) -> float:
        """Get trainer's track-specific ELO."""
        return self.trainer_track_ratings.get((trainer, track), self.initial_elo)

    def get_composite_rating(self, horse_id: str, jockey: str, trainer: str,
                            track: str, distance: float, going: str,
                            weights: Dict[str, float] = None) -> float:
        """
        Calculate a weighted composite rating combining horse, jockey, trainer.

        Default weights:
        - Horse overall: 40%
        - Horse track: 10%
        - Horse distance: 10%
        - Horse going: 5%
        - Jockey overall: 15%
        - Jockey track: 5%
        - Trainer overall: 10%
        - Trainer track: 5%
        """
        if weights is None:
            weights = {
                'horse_overall': 0.40,
                'horse_track': 0.10,
                'horse_distance': 0.10,
                'horse_going': 0.05,
                'jockey_overall': 0.15,
                'jockey_track': 0.05,
                'trainer_overall': 0.10,
                'trainer_track': 0.05,
            }

        composite = (
            weights.get('horse_overall', 0.4) * self.get_horse_rating(horse_id) +
            weights.get('horse_track', 0.1) * self.get_horse_track_rating(horse_id, track) +
            weights.get('horse_distance', 0.1) * self.get_horse_distance_rating(horse_id, distance) +
            weights.get('horse_going', 0.05) * self.get_horse_going_rating(horse_id, going) +
            weights.get('jockey_overall', 0.15) * self.get_jockey_rating(jockey) +
            weights.get('jockey_track', 0.05) * self.get_jockey_track_rating(jockey, track) +
            weights.get('trainer_overall', 0.1) * self.get_trainer_rating(trainer) +
            weights.get('trainer_track', 0.05) * self.get_trainer_track_rating(trainer, track)
        )

        return composite

    def get_horse_rankings(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """Get current top N horses by ELO."""
        sorted_horses = sorted(
            self.horse_ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_horses[:top_n]

    def get_jockey_rankings(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """Get current top N jockeys by ELO."""
        sorted_jockeys = sorted(
            self.jockey_ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_jockeys[:top_n]

    def get_trainer_rankings(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """Get current top N trainers by ELO."""
        sorted_trainers = sorted(
            self.trainer_ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_trainers[:top_n]

    def get_horse_history_df(self) -> pd.DataFrame:
        """Get horse rating history as DataFrame."""
        return pd.DataFrame(self.horse_history)


def build_horse_elo(df: pd.DataFrame, k_factor: int = 32) -> HorseELO:
    """
    Build Horse ELO ratings from race data.

    Expected columns in df:
    - race_id: Unique identifier for each race
    - date: Race date
    - track: Track/course name
    - race_name: Name of the race
    - race_class: Class of the race (e.g., Group 1, Handicap)
    - distance_furlongs: Race distance in furlongs
    - going: Track condition
    - horse_id: Unique horse identifier
    - horse_name: Horse name
    - position: Finishing position (1 = winner)
    - jockey: Jockey name
    - trainer: Trainer name

    Returns:
        HorseELO object with computed ratings
    """
    elo = HorseELO(k_factor=k_factor)

    # Sort by date
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Get unique races in chronological order
    races = df.groupby('race_id').agg({
        'date': 'first',
        'track': 'first',
        'race_name': 'first',
        'race_class': 'first',
        'distance_furlongs': 'first',
        'going': 'first',
    }).reset_index().sort_values('date')

    print(f"Processing {len(races)} races...")

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

        race_df = df[df['race_id'] == race_id][['horse_id', 'horse_name', 'position', 'jockey', 'trainer']]
        elo.process_race(race_df, race_info)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(races)} races")

    print(f"Done! Rated {len(elo.horse_ratings)} horses, {len(elo.jockey_ratings)} jockeys, {len(elo.trainer_ratings)} trainers")
    return elo


if __name__ == "__main__":
    # Example usage - this would be run with actual data
    print("Horse Racing ELO System")
    print("=" * 60)
    print("\nThis module provides ELO rating calculations for horse racing.")
    print("\nUsage:")
    print("  from horse_elo import build_horse_elo, HorseELO")
    print("  elo = build_horse_elo(race_data_df)")
    print("  print(elo.get_horse_rankings(20))")
