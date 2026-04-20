import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from tabmonet.models.autogluon import RealMLPModel

def test_realmlp():
    # 1. Load and Split the data
    data_path = 'datasets/AirfoilSelfNoise.csv'
    df = pd.read_csv(data_path)
    label = 'SSPL'
    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples.")

    # 2. Fit the autogluon model to the data
    # We pass the RealMLPModel class directly to hyperparameters
    predictor = TabularPredictor(label=label, problem_type='regression').fit(
        train_data=train_data,
        hyperparameters={RealMLPModel: {}},
        num_gpus=0,  # Ensure CPU usage
        holdout_frac=0.2
    )

    # 3. Show the leaderboard and output of the model
    print("\n--- Leaderboard ---")
    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)

    print("\n--- Model Predictions (Top 5) ---")
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    y_true = test_data[label]
    
    results = pd.DataFrame({
        'True SSPL': y_true,
        'Predicted SSPL': y_pred
    }).head()
    print(results)

if __name__ == "__main__":
    test_realmlp()
