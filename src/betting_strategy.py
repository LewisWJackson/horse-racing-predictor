"""
Betting Strategy Module
=======================
Advanced betting strategies and calculations for horse racing.

Features:
- Kelly Criterion staking
- Value bet identification
- Accumulator optimization
- Each-way bet analysis
- Dutching calculator
- Expected value calculations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations


@dataclass
class BetSelection:
    """Represents a single betting selection."""
    horse: str
    race: str
    win_prob: float
    place_prob: float  # Probability of top 3
    win_odds: float
    place_odds: float  # Usually win_odds / 4 or 5
    each_way_terms: Tuple[int, int] = (1, 4)  # e.g., 1/4 odds for places


@dataclass
class BetRecommendation:
    """Betting recommendation with staking."""
    selection: BetSelection
    bet_type: str  # 'WIN', 'PLACE', 'EACH_WAY'
    stake: float
    expected_value: float
    value_edge: float
    kelly_fraction: float
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'


class BettingCalculator:
    """Core betting calculations."""

    @staticmethod
    def implied_probability(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if decimal_odds <= 1:
            return 1.0
        return 1.0 / decimal_odds

    @staticmethod
    def decimal_to_fractional(decimal_odds: float) -> str:
        """Convert decimal odds to fractional format."""
        if decimal_odds <= 1:
            return "1/1"
        fraction = decimal_odds - 1
        # Common fractional odds
        common = {0.5: "1/2", 1.0: "1/1", 2.0: "2/1", 3.0: "3/1", 4.0: "4/1",
                  5.0: "5/1", 6.0: "6/1", 7.0: "7/1", 8.0: "8/1", 9.0: "9/1",
                  10.0: "10/1", 0.33: "1/3", 0.25: "1/4", 0.2: "1/5"}
        if fraction in common:
            return common[fraction]
        return f"{fraction:.1f}/1"

    @staticmethod
    def expected_value(prob: float, odds: float, stake: float = 1.0) -> float:
        """Calculate expected value of a bet."""
        win_return = stake * odds
        ev = (prob * win_return) - stake
        return ev

    @staticmethod
    def kelly_criterion(prob: float, odds: float, fraction: float = 0.25) -> float:
        """
        Calculate Kelly Criterion stake as fraction of bankroll.

        Args:
            prob: Probability of winning
            odds: Decimal odds
            fraction: Kelly fraction (0.25 = quarter Kelly for safety)

        Returns:
            Recommended stake as fraction of bankroll
        """
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0.0

        b = odds - 1  # Net odds (profit if win)
        q = 1 - prob  # Probability of losing

        kelly = (b * prob - q) / b

        return max(0, kelly * fraction)

    @staticmethod
    def value_edge(model_prob: float, odds: float) -> float:
        """Calculate value edge (model prob - implied prob)."""
        implied = BettingCalculator.implied_probability(odds)
        return model_prob - implied

    @staticmethod
    def each_way_ev(win_prob: float, place_prob: float,
                    win_odds: float, ew_terms: Tuple[int, int] = (1, 4),
                    stake: float = 1.0) -> Dict:
        """
        Calculate each-way bet expected value.

        Args:
            win_prob: Probability of winning
            place_prob: Probability of placing (top 3)
            win_odds: Decimal win odds
            ew_terms: Each-way terms (numerator, denominator) e.g., (1, 4)
            stake: Stake per part (total stake = 2 * stake)

        Returns:
            Dict with EV breakdown
        """
        place_only_prob = place_prob - win_prob  # Place but not win
        lose_prob = 1 - place_prob

        # Calculate place odds from win odds
        place_odds = 1 + (win_odds - 1) * ew_terms[0] / ew_terms[1]

        # Win part returns
        win_return = stake * win_odds

        # Place part returns (triggers on win OR place)
        place_return = stake * place_odds

        # Expected values for each outcome
        ev_win = win_prob * (win_return + place_return - 2 * stake)  # Both parts win
        ev_place = place_only_prob * (place_return - 2 * stake)  # Only place wins
        ev_lose = lose_prob * (-2 * stake)  # Both lose

        total_ev = ev_win + ev_place + ev_lose

        return {
            'total_ev': total_ev,
            'ev_per_stake': total_ev / (2 * stake),
            'win_odds': win_odds,
            'place_odds': place_odds,
            'total_stake': 2 * stake,
            'max_return': (win_return + place_return),
            'is_value': total_ev > 0
        }


class ValueBetFinder:
    """Find value bets from race predictions."""

    def __init__(self, min_value_edge: float = 0.05, min_prob: float = 0.10):
        """
        Args:
            min_value_edge: Minimum edge to consider a value bet (5% default)
            min_prob: Minimum probability to consider (10% default)
        """
        self.min_edge = min_value_edge
        self.min_prob = min_prob
        self.calc = BettingCalculator()

    def find_value_bets(self, predictions: pd.DataFrame,
                        odds_column: str = 'odds',
                        prob_column: str = 'win_probability') -> List[BetRecommendation]:
        """
        Find value bets from predictions dataframe.

        Args:
            predictions: DataFrame with predictions
            odds_column: Column name for odds
            prob_column: Column name for win probability

        Returns:
            List of BetRecommendation objects
        """
        recommendations = []

        for _, row in predictions.iterrows():
            prob = row[prob_column]
            odds = row[odds_column]

            if prob < self.min_prob:
                continue

            edge = self.calc.value_edge(prob, odds)

            if edge >= self.min_edge:
                # Estimate place probability (roughly 2.5x win prob, capped at 95%)
                place_prob = min(prob * 2.5, 0.95)

                selection = BetSelection(
                    horse=row.get('horse_name', row.get('horse', 'Unknown')),
                    race=row.get('race_id', 'Unknown'),
                    win_prob=prob,
                    place_prob=place_prob,
                    win_odds=odds,
                    place_odds=1 + (odds - 1) / 4
                )

                # Determine best bet type
                win_ev = self.calc.expected_value(prob, odds)
                ew_result = self.calc.each_way_ev(prob, place_prob, odds)

                if win_ev > ew_result['total_ev'] / 2:
                    bet_type = 'WIN'
                    ev = win_ev
                else:
                    bet_type = 'EACH_WAY'
                    ev = ew_result['total_ev']

                kelly = self.calc.kelly_criterion(prob, odds)

                # Confidence level
                if edge > 0.15:
                    confidence = 'HIGH'
                elif edge > 0.08:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'

                recommendations.append(BetRecommendation(
                    selection=selection,
                    bet_type=bet_type,
                    stake=kelly,  # As fraction of bankroll
                    expected_value=ev,
                    value_edge=edge,
                    kelly_fraction=kelly,
                    confidence=confidence
                ))

        # Sort by expected value
        recommendations.sort(key=lambda x: x.expected_value, reverse=True)

        return recommendations


class AccumulatorOptimizer:
    """Optimize accumulator bet selections."""

    def __init__(self, max_selections: int = 6, min_leg_prob: float = 0.25):
        """
        Args:
            max_selections: Maximum legs in accumulator
            min_leg_prob: Minimum probability per leg
        """
        self.max_selections = max_selections
        self.min_leg_prob = min_leg_prob
        self.calc = BettingCalculator()

    def calculate_acca_stats(self, selections: List[BetSelection]) -> Dict:
        """Calculate accumulator statistics."""
        if not selections:
            return {'error': 'No selections'}

        combined_odds = 1.0
        combined_prob = 1.0
        legs = []

        for sel in selections:
            combined_odds *= sel.win_odds
            combined_prob *= sel.win_prob
            legs.append({
                'horse': sel.horse,
                'race': sel.race,
                'odds': sel.win_odds,
                'prob': sel.win_prob
            })

        implied_prob = self.calc.implied_probability(combined_odds)
        edge = combined_prob - implied_prob
        ev = self.calc.expected_value(combined_prob, combined_odds)

        return {
            'legs': legs,
            'num_selections': len(selections),
            'combined_odds': combined_odds,
            'combined_probability': combined_prob,
            'implied_probability': implied_prob,
            'value_edge': edge,
            'expected_value': ev,
            'is_value': edge > 0,
            'kelly_stake': self.calc.kelly_criterion(combined_prob, combined_odds)
        }

    def find_best_accumulators(self, selections: List[BetSelection],
                               acca_sizes: List[int] = [2, 3, 4]) -> List[Dict]:
        """
        Find the best accumulator combinations.

        Args:
            selections: Available selections
            acca_sizes: Sizes of accumulators to consider

        Returns:
            List of best accumulators sorted by expected value
        """
        all_accas = []

        # Filter selections by minimum probability
        valid_selections = [s for s in selections if s.win_prob >= self.min_leg_prob]

        for size in acca_sizes:
            if size > len(valid_selections):
                continue

            for combo in combinations(valid_selections, size):
                stats = self.calculate_acca_stats(list(combo))
                if stats.get('is_value', False):
                    all_accas.append(stats)

        # Sort by expected value
        all_accas.sort(key=lambda x: x['expected_value'], reverse=True)

        return all_accas[:20]  # Return top 20


class DutchingCalculator:
    """Calculate dutching stakes to guarantee profit regardless of winner."""

    def __init__(self, total_stake: float = 100.0):
        self.total_stake = total_stake

    def calculate_dutch(self, odds_list: List[float]) -> Dict:
        """
        Calculate dutching stakes for guaranteed equal profit.

        Args:
            odds_list: List of decimal odds for each selection

        Returns:
            Dict with stakes and expected profit
        """
        if not odds_list or any(o <= 1 for o in odds_list):
            return {'error': 'Invalid odds'}

        # Calculate implied probability sum
        implied_sum = sum(1/o for o in odds_list)

        # If sum > 1, no arbitrage opportunity
        if implied_sum >= 1:
            return {
                'is_arbitrage': False,
                'margin': (implied_sum - 1) * 100,
                'stakes': None,
                'profit': None
            }

        # Calculate stakes proportional to 1/odds
        stakes = [(self.total_stake / o) / implied_sum for o in odds_list]

        # Verify: all returns should be equal
        returns = [s * o for s, o in zip(stakes, odds_list)]
        guaranteed_return = returns[0]
        profit = guaranteed_return - self.total_stake
        roi = (profit / self.total_stake) * 100

        return {
            'is_arbitrage': True,
            'total_stake': self.total_stake,
            'stakes': stakes,
            'returns': returns,
            'guaranteed_return': guaranteed_return,
            'guaranteed_profit': profit,
            'roi': roi
        }


class StakingStrategy:
    """Different staking strategies for betting."""

    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll
        self.calc = BettingCalculator()

    def flat_stake(self, percentage: float = 0.02) -> float:
        """Fixed percentage of bankroll."""
        return self.bankroll * percentage

    def kelly_stake(self, prob: float, odds: float, fraction: float = 0.25) -> float:
        """Kelly Criterion with fractional adjustment."""
        kelly = self.calc.kelly_criterion(prob, odds, fraction)
        return self.bankroll * kelly

    def percentage_confidence(self, confidence: str) -> float:
        """Stake based on confidence level."""
        stakes = {
            'HIGH': 0.05,
            'MEDIUM': 0.03,
            'LOW': 0.01
        }
        return self.bankroll * stakes.get(confidence, 0.02)

    def proportional_value(self, edge: float, max_stake_pct: float = 0.05) -> float:
        """Stake proportional to value edge."""
        if edge <= 0:
            return 0
        # Scale edge to stake (e.g., 10% edge = 2.5% stake)
        stake_pct = min(edge * 0.25, max_stake_pct)
        return self.bankroll * stake_pct


class BettingReportGenerator:
    """Generate betting reports and summaries."""

    def __init__(self):
        self.value_finder = ValueBetFinder()
        self.acca_optimizer = AccumulatorOptimizer()

    def generate_daily_report(self, predictions: pd.DataFrame,
                              bankroll: float = 1000.0) -> Dict:
        """
        Generate comprehensive daily betting report.

        Args:
            predictions: DataFrame with race predictions
            bankroll: Current bankroll

        Returns:
            Dict with betting recommendations
        """
        staking = StakingStrategy(bankroll)

        # Find value bets
        value_bets = self.value_finder.find_value_bets(predictions)

        # Create selections for accumulator analysis
        selections = []
        for rec in value_bets:
            selections.append(rec.selection)

        # Find best accumulators
        best_accas = self.acca_optimizer.find_best_accumulators(selections)

        # Calculate suggested stakes
        win_bets = []
        each_way_bets = []

        for rec in value_bets:
            stake = staking.kelly_stake(rec.selection.win_prob, rec.selection.win_odds)

            bet_info = {
                'horse': rec.selection.horse,
                'race': rec.selection.race,
                'odds': rec.selection.win_odds,
                'probability': rec.selection.win_prob,
                'edge': rec.value_edge,
                'stake': round(stake, 2),
                'ev': round(BettingCalculator.expected_value(rec.selection.win_prob, rec.selection.win_odds, stake), 2),
                'confidence': rec.confidence
            }

            if rec.bet_type == 'WIN':
                win_bets.append(bet_info)
            else:
                each_way_bets.append(bet_info)

        return {
            'summary': {
                'total_value_bets': len(value_bets),
                'high_confidence': len([v for v in value_bets if v.confidence == 'HIGH']),
                'medium_confidence': len([v for v in value_bets if v.confidence == 'MEDIUM']),
                'total_suggested_stake': sum(b['stake'] for b in win_bets + each_way_bets),
                'total_expected_value': sum(b['ev'] for b in win_bets + each_way_bets)
            },
            'win_bets': win_bets,
            'each_way_bets': each_way_bets,
            'accumulators': best_accas[:5],
            'bankroll': bankroll
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("BETTING STRATEGY MODULE")
    print("=" * 60)

    calc = BettingCalculator()

    # Example calculations
    print("\n--- Kelly Criterion Example ---")
    prob = 0.35
    odds = 3.5
    kelly = calc.kelly_criterion(prob, odds)
    print(f"Probability: {prob*100}%")
    print(f"Odds: {odds}")
    print(f"Kelly Stake (25%): {kelly*100:.2f}% of bankroll")

    print("\n--- Each-Way Analysis ---")
    ew = calc.each_way_ev(0.20, 0.55, 6.0)
    print(f"Win Odds: {ew['win_odds']}")
    print(f"Place Odds: {ew['place_odds']:.2f}")
    print(f"Total EV: £{ew['total_ev']:.2f} per £1 each-way")
    print(f"Is Value: {ew['is_value']}")

    print("\n--- Dutching Example ---")
    dutch = DutchingCalculator(100)
    result = dutch.calculate_dutch([3.0, 4.0, 6.0])
    if result.get('is_arbitrage'):
        print(f"Arbitrage found!")
        print(f"Stakes: {[f'£{s:.2f}' for s in result['stakes']]}")
        print(f"Guaranteed Profit: £{result['guaranteed_profit']:.2f}")
        print(f"ROI: {result['roi']:.2f}%")
    else:
        print(f"No arbitrage. Bookmaker margin: {result['margin']:.2f}%")
