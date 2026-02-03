"""
Horse Racing Predictor - Demo Script
====================================
Demonstrates the full pipeline with sample data.

Run: python demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import create_sample_data
from horse_racing_predictor import HorseRacingPredictor
import pandas as pd

def main():
    print("=" * 70)
    print("HORSE RACING PREDICTOR - DEMO")
    print("=" * 70)
    print("\nFollowing the same methodology as Tennis/Golf/UFC predictors:")
    print("- ELO Rating System (adapted for multi-runner races)")
    print("- Feature Engineering (50+ features)")
    print("- Random Forest, XGBoost, Ensemble Models")
    print("- Confidence-based predictions")

    # Step 1: Create sample data
    print("\n" + "=" * 50)
    print("STEP 1: Creating Sample Data")
    print("=" * 50)

    # Create more races for better training
    df = create_sample_data(n_races=500, n_horses_per_race=10)
    print(f"\nCreated {len(df)} horse-race combinations")
    print(f"Unique races: {df['race_id'].nunique()}")
    print(f"Unique horses: {df['horse_id'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Step 2: Train the predictor
    print("\n" + "=" * 50)
    print("STEP 2: Training the Predictor")
    print("=" * 50)

    predictor = HorseRacingPredictor()

    # Use 80% for training, 20% for testing
    split_date = df['date'].quantile(0.8)
    results = predictor.train(df, test_date=str(split_date.date()), target='win')

    # Step 3: Show Top ELO Rankings
    print("\n" + "=" * 50)
    print("STEP 3: Top ELO Rankings")
    print("=" * 50)

    elo = predictor.feature_engineer.get_elo_system()

    print("\nTop 20 Horses by ELO:")
    for i, (horse_id, rating) in enumerate(elo.get_horse_rankings(20), 1):
        print(f"  {i:2d}. {horse_id:20s} {rating:.0f}")

    print("\nTop 10 Jockeys by ELO:")
    for i, (jockey, rating) in enumerate(elo.get_jockey_rankings(10), 1):
        print(f"  {i:2d}. {jockey:20s} {rating:.0f}")

    print("\nTop 10 Trainers by ELO:")
    for i, (trainer, rating) in enumerate(elo.get_trainer_rankings(10), 1):
        print(f"  {i:2d}. {trainer:20s} {rating:.0f}")

    # Step 4: Demo Prediction on New Race
    print("\n" + "=" * 50)
    print("STEP 4: Predicting a New Race")
    print("=" * 50)

    # Create a hypothetical new race
    new_race_df = pd.DataFrame([
        {'horse_id': 'horse_0', 'horse_name': 'Horse_0', 'jockey': 'J.Smith', 'trainer': 'T.Anderson'},
        {'horse_id': 'horse_1', 'horse_name': 'Horse_1', 'jockey': 'J.Johnson', 'trainer': 'T.Thomas'},
        {'horse_id': 'horse_10', 'horse_name': 'Horse_10', 'jockey': 'J.Brown', 'trainer': 'T.Jackson'},
        {'horse_id': 'horse_15', 'horse_name': 'Horse_15', 'jockey': 'J.Williams', 'trainer': 'T.White'},
        {'horse_id': 'horse_20', 'horse_name': 'Horse_20', 'jockey': 'J.Jones', 'trainer': 'T.Harris'},
        {'horse_id': 'horse_25', 'horse_name': 'Horse_25', 'jockey': 'J.Taylor', 'trainer': 'T.Martin'},
        {'horse_id': 'horse_50', 'horse_name': 'Horse_50', 'jockey': 'J.Smith', 'trainer': 'T.Anderson'},
        {'horse_id': 'horse_100', 'horse_name': 'Horse_100', 'jockey': 'J.Johnson', 'trainer': 'T.Thomas'},
    ])

    race_info = {
        'track': 'Ascot',
        'distance_furlongs': 8,
        'going': 'Good',
        'race_class': 'Group_1',
        'date': '2025-06-15',
        'race_name': 'Demo Stakes'
    }

    print(f"\nPredicting: {race_info['race_name']} at {race_info['track']}")
    print(f"Distance: {race_info['distance_furlongs']}f, Going: {race_info['going']}")

    predictions = predictor.predict_race(new_race_df, race_info)

    print("\nPredicted Order of Finish:")
    print("-" * 60)
    for i, (_, row) in enumerate(predictions.iterrows(), 1):
        prob = row['win_probability']
        bar = '#' * int(prob * 40)
        print(f"  {i}. {row['horse_name']:15s} {prob:.1%} {bar}")
        if i <= 3:
            print(f"     Horse ELO: {row['horse_elo']:.0f}, Jockey: {row['jockey']}, Trainer: {row['trainer']}")

    # Step 5: Save the model
    print("\n" + "=" * 50)
    print("STEP 5: Saving Model")
    print("=" * 50)

    os.makedirs('models', exist_ok=True)
    predictor.save('models/demo_model.pkl')

    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("""
Next Steps:
1. Get real horse racing data from Kaggle or racing APIs
2. Run: predictor.train(your_real_data)
3. Use predictor.predict_race() for upcoming races

Data Sources:
- Kaggle: search "horse racing" datasets
- github.com/Samuelson777/Horse-Race-Prediction
- theracingapi.com (UK racing)
- betfair.com historical data
""")


if __name__ == "__main__":
    main()
