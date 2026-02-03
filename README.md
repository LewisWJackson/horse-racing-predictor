# Horse Racing Predictor

Machine learning prediction system for horse racing outcomes, built using the same methodology as your Tennis, Golf, and UFC predictors.

## Methodology Overview

This predictor follows the exact approach from the video:

1. **ELO Rating System** - Adapted for multi-runner horse races (like Golf ELO)
2. **Feature Engineering** - 50+ features including form, track/distance/going performance
3. **Multiple ML Models** - Decision Tree, Random Forest, XGBoost, Neural Network
4. **Ensemble Methods** - Voting and Stacking classifiers (like UFC predictor)

## Expected Outcomes

Based on academic research and similar implementations:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **Win Prediction Accuracy** | 25-35% | Picking the winner of a race |
| **Top 3 Accuracy** | 55-65% | Top pick finishes in top 3 |
| **Binary Classification (win/not)** | 70-80% | Similar to Tennis (70%) |
| **AUC-ROC** | 0.75-0.85 | Strong ranking ability |

### Why Horse Racing is Harder Than Tennis

- **Multi-runner fields**: 8-20 horses vs 2 players
- **Random baseline for winning**: ~8% (vs 50% in tennis)
- **More variables**: jockey, trainer, weight, draw, going
- **Less historical data per horse**: ~20-40 races lifetime vs 500+ matches

### Research Benchmarks

From recent studies:
- [CatBoost, LightGBM, XGBoost achieve R² ≈ 0.85](https://medium.com/@cagdasgul/high-precision-prediction-of-horse-racing-durations-using-ensemble-machine-learning-models-a-d6af16a1ebf1) for finish time prediction
- Hong Kong studies report 70%+ binary classification accuracy
- Winner prediction typically ranges 25-35% (vs ~12% baseline)

## Key Features

### ELO System (horse_elo.py)

Like your Golf ELO, uses "virtual matchups":
- Each horse plays a virtual race against every other horse
- Winner gets points, loser loses points
- Averaged across all matchups

**Multiple ELO Types:**
- Overall Horse ELO
- Track-specific ELO (e.g., Ascot specialist)
- Distance-specific ELO (Sprint, Mile, Route, Marathon)
- Going-specific ELO (Good, Soft, Firm)
- Jockey ELO
- Trainer ELO
- Composite ELO (weighted combination)

### Feature Engineering (feature_engineering.py)

**50+ Features Including:**
- ELO ratings (horse, jockey, trainer, track/distance/going specific)
- Recent form (last 5/10 races win rate, place rate, form score)
- Track/distance/going performance history
- Jockey-trainer combo strike rate
- Days since last race (fitness vs freshness)
- Current streak
- Weight analysis (vs recent carried weights)
- Draw/barrier position
- Field-relative features (ELO rank in field, ELO vs field average)

### Models (horse_racing_predictor.py)

Same models as your other predictors:
1. **Decision Tree** - Baseline, interpretable
2. **Random Forest** - Ensemble of trees
3. **Extra Trees** - More randomized forest
4. **Gradient Boosting** - Sequential boosting
5. **XGBoost** - Optimized gradient boosting
6. **Neural Network** - Deep learning
7. **Voting Ensemble** - Soft voting across models
8. **Stacking Ensemble** - Meta-learner on model outputs

## Installation

```bash
cd horse-racing-predictor
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Your Data

```python
from src.data_preprocessing import load_and_preprocess, create_sample_data

# Load real data
df = load_and_preprocess('data/raw/your_racing_data.csv')

# Or create sample data for testing
df = create_sample_data(n_races=500)
```

### 2. Train the Model

```python
from src.horse_racing_predictor import HorseRacingPredictor

predictor = HorseRacingPredictor()
results = predictor.train(df, test_date='2024-01-01', target='win')

# Save the model
predictor.save('models/horse_racing_model.pkl')
```

### 3. Make Predictions

```python
# Load trained model
predictor = HorseRacingPredictor.load('models/horse_racing_model.pkl')

# Predict a new race
race_info = {
    'track': 'Ascot',
    'distance_furlongs': 8,
    'going': 'Good',
    'race_class': 'Group_1',
    'date': '2025-06-15',
    'race_name': 'Royal Ascot Stakes'
}

predictions = predictor.predict_race(race_df, race_info)
print(predictions[['horse_name', 'win_probability', 'predicted_rank']])
```

## Data Format

Required columns:
- `race_id`: Unique race identifier
- `date`: Race date
- `track`: Course/track name
- `race_name`: Name of the race
- `race_class`: Class (Group 1, Handicap, etc.)
- `distance_furlongs`: Distance in furlongs
- `going`: Track condition (Good, Soft, etc.)
- `horse_id`: Unique horse identifier
- `horse_name`: Horse name
- `position`: Finishing position (1 = winner)
- `jockey`: Jockey name
- `trainer`: Trainer name

Optional columns:
- `weight`: Carried weight
- `draw`: Starting position/barrier
- `age`: Horse age
- `odds`: Betting odds

## Data Sources

1. **Kaggle - Hong Kong Horse Racing (1997-2005)**
   - 6,349 races with detailed runner information

2. **Kaggle - Horse Racing Dataset (1990-2020)**
   - [github.com/Samuelson777/Horse-Race-Prediction](https://github.com/Samuelson777/Horse-Race-Prediction)

3. **UK Racing API** (subscription)
   - theracingapi.com

4. **Betfair Historical Data**
   - developer.betfair.com

## File Structure

```
horse-racing-predictor/
├── src/
│   ├── horse_elo.py              # ELO rating system
│   ├── feature_engineering.py    # Feature calculation
│   ├── horse_racing_predictor.py # Main predictor
│   └── data_preprocessing.py     # Data cleaning
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed features
├── models/                       # Saved models
├── requirements.txt
└── README.md
```

## Comparison to Your Other Sports Predictors

| Aspect | Tennis | Golf | UFC | Horse Racing |
|--------|--------|------|-----|--------------|
| **Format** | 1v1 | Multi | 1v1 | Multi |
| **ELO Type** | Head-to-head | Virtual matchups | Head-to-head | Virtual matchups |
| **Surface/Track** | Clay/Grass/Hard | Course-specific | Weight class | Track/Going/Distance |
| **Best Model** | XGBoost (70%) | XGBoost (42.6% T10) | Stacking | TBD |
| **Key Features** | ELO, H2H, Form | ELO, SG stats | ELO, Momentum | ELO, J/T combo, Form |

## Expected Feature Importance

Based on similar implementations and your other predictors:

1. **composite_elo_rank** (~25-30%) - Most predictive
2. **horse_elo_vs_field** (~15-20%)
3. **form_score_5** (~10-15%)
4. **jockey_strike_rate** (~8-10%)
5. **track_win_rate** (~5-8%)

## Tips for Best Results

1. **More data = better predictions** - Aim for 10,000+ races
2. **Recent data matters** - Use last 3-5 years
3. **Track-specific models** - Consider training separate models for different tracks
4. **Distance matters** - Horses specialize; track this carefully
5. **Going is crucial** - Some horses are "mud lovers", others hate soft ground

## License

MIT - Use freely, but gamble responsibly!
