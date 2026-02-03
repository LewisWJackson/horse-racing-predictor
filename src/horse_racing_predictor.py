"""
Horse Racing Predictor
======================
Machine Learning prediction system for horse racing outcomes.

Following the same methodology used in:
- Tennis Predictor (Random Forest, XGBoost, 70% accuracy)
- Golf Predictor (XGBoost, 42.6% Top-10 accuracy)
- UFC Predictor (Stacking Ensemble, voting classifier)

Models included:
1. Decision Tree (baseline)
2. Random Forest
3. XGBoost
4. Gradient Boosting
5. Neural Network
6. Voting Ensemble
7. Stacking Ensemble

Features are ranked, and the best model is saved for predictions.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import HorseRacingFeatureEngineer, prepare_training_data

np.random.seed(42)


class HorseRacingPredictor:
    """
    Complete horse racing prediction system.

    Similar architecture to your Tennis and UFC predictors.
    """

    def __init__(self):
        self.feature_engineer = HorseRacingFeatureEngineer()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_cols = None

    def train(self, df: pd.DataFrame, test_date: str = '2024-01-01',
              target: str = 'win') -> dict:
        """
        Train all models and evaluate performance.

        Args:
            df: Full race dataset
            test_date: Date to split train/test
            target: 'win' or 'place'

        Returns:
            Dictionary of model results
        """
        print("=" * 70)
        print("HORSE RACING PREDICTOR - TRAINING")
        print("=" * 70)

        # Step 1: Feature Engineering
        print("\n" + "=" * 50)
        print("STEP 1: Feature Engineering")
        print("=" * 50)

        features_df = self.feature_engineer.process_all_races(df)

        # Step 2: Prepare Training Data
        print("\n" + "=" * 50)
        print("STEP 2: Preparing Training Data")
        print("=" * 50)

        X_train, y_train, X_test, y_test, train_df, test_df, feature_cols = \
            prepare_training_data(features_df, test_date, target)

        self.feature_cols = feature_cols

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Baseline
        baseline = y_test.mean()
        print(f"\nBaseline (random selection): {baseline * 100:.1f}%")

        # Step 3: Train Models
        print("\n" + "=" * 50)
        print("STEP 3: Training Models")
        print("=" * 50)

        results = {}

        # 1. Decision Tree
        print("\n--- Decision Tree ---")
        dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
        dt.fit(X_train_scaled, y_train)
        dt_pred = dt.predict(X_test_scaled)
        dt_acc = accuracy_score(y_test, dt_pred)
        print(f"Accuracy: {dt_acc:.2%}")
        self.models['Decision Tree'] = dt
        results['Decision Tree'] = {'accuracy': dt_acc, 'model': dt}

        # 2. Random Forest
        print("\n--- Random Forest ---")
        rf = RandomForestClassifier(
            n_estimators=200,
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
        print(f"Accuracy: {rf_acc:.2%}")
        print(f"AUC-ROC: {rf_auc:.4f}")
        self.models['Random Forest'] = rf
        results['Random Forest'] = {'accuracy': rf_acc, 'auc': rf_auc, 'model': rf}

        # Feature importance from Random Forest
        print("\nTop 15 Features (Random Forest):")
        importance = list(zip(feature_cols, rf.feature_importances_))
        importance.sort(key=lambda x: x[1], reverse=True)
        for feat, imp in importance[:15]:
            bar = '#' * int(imp * 200)
            print(f"  {feat:25s} {imp:.4f} {bar}")

        # 3. Extra Trees
        print("\n--- Extra Trees ---")
        et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        et.fit(X_train_scaled, y_train)
        et_pred = et.predict(X_test_scaled)
        et_proba = et.predict_proba(X_test_scaled)[:, 1]
        et_acc = accuracy_score(y_test, et_pred)
        et_auc = roc_auc_score(y_test, et_proba)
        print(f"Accuracy: {et_acc:.2%}")
        print(f"AUC-ROC: {et_auc:.4f}")
        self.models['Extra Trees'] = et
        results['Extra Trees'] = {'accuracy': et_acc, 'auc': et_auc, 'model': et}

        # 4. Gradient Boosting
        print("\n--- Gradient Boosting ---")
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=15,
            random_state=42
        )
        gb.fit(X_train_scaled, y_train)
        gb_pred = gb.predict(X_test_scaled)
        gb_proba = gb.predict_proba(X_test_scaled)[:, 1]
        gb_acc = accuracy_score(y_test, gb_pred)
        gb_auc = roc_auc_score(y_test, gb_proba)
        print(f"Accuracy: {gb_acc:.2%}")
        print(f"AUC-ROC: {gb_auc:.4f}")
        self.models['Gradient Boosting'] = gb
        results['Gradient Boosting'] = {'accuracy': gb_acc, 'auc': gb_auc, 'model': gb}

        # 5. XGBoost
        print("\n--- XGBoost ---")
        try:
            import xgboost as xgb

            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
            xgb_acc = accuracy_score(y_test, xgb_pred)
            xgb_auc = roc_auc_score(y_test, xgb_proba)
            print(f"Accuracy: {xgb_acc:.2%}")
            print(f"AUC-ROC: {xgb_auc:.4f}")
            self.models['XGBoost'] = xgb_model
            results['XGBoost'] = {'accuracy': xgb_acc, 'auc': xgb_auc, 'model': xgb_model}

            # XGBoost feature importance
            print("\nTop 15 Features (XGBoost):")
            xgb_importance = list(zip(feature_cols, xgb_model.feature_importances_))
            xgb_importance.sort(key=lambda x: x[1], reverse=True)
            for feat, imp in xgb_importance[:15]:
                bar = '#' * int(imp * 200)
                print(f"  {feat:25s} {imp:.4f} {bar}")

            has_xgb = True
        except ImportError:
            print("XGBoost not available. Install with: pip install xgboost")
            has_xgb = False

        # 6. Neural Network
        print("\n--- Neural Network ---")
        nn = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        nn.fit(X_train_scaled, y_train)
        nn_pred = nn.predict(X_test_scaled)
        nn_proba = nn.predict_proba(X_test_scaled)[:, 1]
        nn_acc = accuracy_score(y_test, nn_pred)
        nn_auc = roc_auc_score(y_test, nn_proba)
        print(f"Accuracy: {nn_acc:.2%}")
        print(f"AUC-ROC: {nn_auc:.4f}")
        self.models['Neural Network'] = nn
        results['Neural Network'] = {'accuracy': nn_acc, 'auc': nn_auc, 'model': nn}

        # 7. Logistic Regression (for stacking)
        print("\n--- Logistic Regression ---")
        lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
        lr_acc = accuracy_score(y_test, lr_pred)
        lr_auc = roc_auc_score(y_test, lr_proba)
        print(f"Accuracy: {lr_acc:.2%}")
        print(f"AUC-ROC: {lr_auc:.4f}")
        self.models['Logistic Regression'] = lr
        results['Logistic Regression'] = {'accuracy': lr_acc, 'auc': lr_auc, 'model': lr}

        # 8. Voting Ensemble
        print("\n--- Voting Ensemble ---")
        estimators = [
            ('rf', rf),
            ('et', et),
            ('gb', gb),
            ('lr', lr),
        ]
        if has_xgb:
            estimators.append(('xgb', xgb_model))

        voting = VotingClassifier(estimators=estimators, voting='soft')
        voting.fit(X_train_scaled, y_train)
        voting_pred = voting.predict(X_test_scaled)
        voting_proba = voting.predict_proba(X_test_scaled)[:, 1]
        voting_acc = accuracy_score(y_test, voting_pred)
        voting_auc = roc_auc_score(y_test, voting_proba)
        print(f"Accuracy: {voting_acc:.2%}")
        print(f"AUC-ROC: {voting_auc:.4f}")
        self.models['Voting Ensemble'] = voting
        results['Voting Ensemble'] = {'accuracy': voting_acc, 'auc': voting_auc, 'model': voting}

        # 9. Stacking Ensemble
        print("\n--- Stacking Ensemble ---")
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        stacking.fit(X_train_scaled, y_train)
        stacking_pred = stacking.predict(X_test_scaled)
        stacking_proba = stacking.predict_proba(X_test_scaled)[:, 1]
        stacking_acc = accuracy_score(y_test, stacking_pred)
        stacking_auc = roc_auc_score(y_test, stacking_proba)
        print(f"Accuracy: {stacking_acc:.2%}")
        print(f"AUC-ROC: {stacking_auc:.4f}")
        self.models['Stacking Ensemble'] = stacking
        results['Stacking Ensemble'] = {'accuracy': stacking_acc, 'auc': stacking_auc, 'model': stacking}

        # Step 4: Confidence Analysis
        print("\n" + "=" * 50)
        print("STEP 4: Confidence-Based Accuracy")
        print("=" * 50)

        test_df_copy = test_df.copy()
        test_df_copy['pred_proba'] = stacking_proba
        test_df_copy['prediction'] = stacking_pred

        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]:
            confident = test_df_copy[
                (test_df_copy['pred_proba'] >= threshold) |
                (test_df_copy['pred_proba'] <= (1 - threshold))
            ]
            if len(confident) > 0:
                conf_acc = accuracy_score(confident['target'], confident['prediction'])
                print(f"Confidence >= {threshold:.0%}: {conf_acc:.2%} ({len(confident)} predictions)")

        # Step 5: Race-Level Accuracy
        print("\n" + "=" * 50)
        print("STEP 5: Race-Level Accuracy (Picking Winners)")
        print("=" * 50)

        # For each race, check if we correctly identified the winner
        race_results = []
        for race_id in test_df_copy['race_id'].unique():
            race = test_df_copy[test_df_copy['race_id'] == race_id]
            if len(race) < 2:
                continue

            # Get our predicted winner (highest probability)
            predicted_winner_idx = race['pred_proba'].idxmax()
            predicted_winner = race.loc[predicted_winner_idx, 'horse_name']
            actual_winner = race[race['position'] == 1]['horse_name'].values

            correct = 1 if len(actual_winner) > 0 and predicted_winner == actual_winner[0] else 0
            race_results.append(correct)

        race_accuracy = np.mean(race_results) if race_results else 0
        print(f"Race winner prediction accuracy: {race_accuracy:.2%} ({sum(race_results)}/{len(race_results)} races)")

        # Top 3 prediction accuracy
        top3_results = []
        for race_id in test_df_copy['race_id'].unique():
            race = test_df_copy[test_df_copy['race_id'] == race_id]
            if len(race) < 2:
                continue

            predicted_winner_idx = race['pred_proba'].idxmax()
            actual_position = race.loc[predicted_winner_idx, 'position']

            correct = 1 if actual_position <= 3 else 0
            top3_results.append(correct)

        top3_accuracy = np.mean(top3_results) if top3_results else 0
        print(f"Top pick finishes in Top 3: {top3_accuracy:.2%} ({sum(top3_results)}/{len(top3_results)} races)")

        # Step 6: Final Results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        # Find best model by AUC
        best_name = max(results.keys(), key=lambda k: results[k].get('auc', results[k]['accuracy']))
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name

        print(f"\nBaseline (random): {baseline:.2%}")
        print(f"\nModel Performance (sorted by AUC):")
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('auc', x[1]['accuracy']), reverse=True)
        for name, data in sorted_results:
            auc_str = f"AUC: {data['auc']:.4f}" if 'auc' in data else ""
            marker = " <-- BEST" if name == best_name else ""
            print(f"  {name:20s}: Acc: {data['accuracy']:.2%}  {auc_str}{marker}")

        print(f"\nRace-level winner accuracy: {race_accuracy:.2%}")
        print(f"Race-level top 3 accuracy: {top3_accuracy:.2%}")

        return results

    def predict_race(self, race_df: pd.DataFrame, race_info: dict) -> pd.DataFrame:
        """
        Predict outcomes for a new race.

        Args:
            race_df: DataFrame with horse entries
            race_info: Dict with race metadata

        Returns:
            DataFrame with predictions
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Calculate features
        features = self.feature_engineer.calculate_features_for_race(race_df, race_info)

        # Calculate relative features for this race
        from feature_engineering import calculate_relative_features
        features['race_id'] = 'prediction_race'
        features = calculate_relative_features(features)

        # Get available features (some relative features may not apply to single race)
        available_cols = [c for c in self.feature_cols if c in features.columns]
        missing_cols = [c for c in self.feature_cols if c not in features.columns]

        # Fill missing with zeros
        for col in missing_cols:
            features[col] = 0

        # Scale and predict
        X = features[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        proba = self.best_model.predict_proba(X_scaled)[:, 1]

        features['win_probability'] = proba
        features['predicted_rank'] = features['win_probability'].rank(ascending=False, method='min')

        return features.sort_values('win_probability', ascending=False)

    def save(self, filepath: str):
        """Save the trained model."""
        save_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_engineer': self.feature_engineer,
            'all_models': self.models,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'HorseRacingPredictor':
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        predictor = cls()
        predictor.best_model = data['best_model']
        predictor.best_model_name = data['best_model_name']
        predictor.scaler = data['scaler']
        predictor.feature_cols = data['feature_cols']
        predictor.feature_engineer = data['feature_engineer']
        predictor.models = data.get('all_models', {})

        print(f"Loaded {predictor.best_model_name} model from {filepath}")
        return predictor


def main():
    """Example main function - replace with actual data loading."""
    print("=" * 70)
    print("HORSE RACING PREDICTOR")
    print("=" * 70)
    print("\nThis script requires horse racing data to run.")
    print("\nExpected data format:")
    print("  - race_id: Unique identifier for each race")
    print("  - date: Race date")
    print("  - track: Track/course name")
    print("  - race_name: Name of the race")
    print("  - race_class: Class of race (Group 1, Handicap, etc.)")
    print("  - distance_furlongs: Race distance in furlongs")
    print("  - going: Track condition (Good, Soft, Firm, etc.)")
    print("  - horse_id: Unique horse identifier")
    print("  - horse_name: Horse name")
    print("  - position: Finishing position (1 = winner)")
    print("  - jockey: Jockey name")
    print("  - trainer: Trainer name")
    print("  - Optional: weight, draw, age, odds")

    print("\nExample usage:")
    print("  predictor = HorseRacingPredictor()")
    print("  results = predictor.train(race_data_df)")
    print("  predictor.save('models/horse_racing_model.pkl')")

    print("\n" + "=" * 70)
    print("DATA SOURCES FOR HORSE RACING")
    print("=" * 70)
    print("""
Available datasets:

1. Kaggle - Hong Kong Horse Racing (1997-2005)
   - 6,349 races with 4,405 runners
   - URL: kaggle.com/datasets

2. Kaggle - Horse Racing Dataset (1990-2020)
   - Comprehensive historical data
   - URL: github.com/Samuelson777/Horse-Race-Prediction

3. UK Racing Data (subscription)
   - Full UK/Ireland race data
   - URL: theracingapi.com

4. Turkish Jockey Club (TJK)
   - 700,000+ races
   - Requires scraping

5. Betfair Historical Data
   - Exchange odds data
   - URL: developer.betfair.com
""")


if __name__ == "__main__":
    main()
