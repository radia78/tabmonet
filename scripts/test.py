import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from tabmonet.models.autogluon import TabMONetModel
from sklearn.model_selection import train_test_split
import torch


def test_tabmonet():
    # 1. Load a small dataset from AutoGluon (Knot Theory is a good small one)
    print("Loading dataset...")
    X = pd.read_csv("datasets/AirfoilSelfNoise.csv")
    X_train, X_test = train_test_split(X, test_size=0.2)
    train_data = TabularDataset(X_train)
    test_data = TabularDataset(X_test)
    label = "SSPL"

    print(f"Dataset loaded. Label: {label}")
    print(f"Train shape: {train_data.shape}")

    # 2. Initialize and Train the TabularPredictor with TabMONetModel
    # We pass small hyperparameters to keep the test fast
    hyperparameters = {
        TabMONetModel: {
            "emb_dim": 256,
            "n_blocks": 1,
            "model_type": "v1",
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        }
    }

    print("Starting training with TabMONetModel...")
    predictor = TabularPredictor(
        label=label, eval_metric="root_mean_squared_error"
    ).fit(
        train_data,
        hyperparameters=hyperparameters,
        time_limit=300,  # Strict time limit for testing
    )

    # 3. Perform Inference
    print("\nTraining completed. Starting inference...")
    y_pred = predictor.predict(test_data)

    # 4. Evaluate
    perf = predictor.evaluate(test_data)
    print(f"\nEvaluation performance: {perf}")
    print(f"Predictions head: \n{y_pred.head()}")

    # 5. Check model summary
    print("\nModel Leaderboard:")
    print(predictor.leaderboard(test_data))


if __name__ == "__main__":
    test_tabmonet()
