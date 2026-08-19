"""Reproduces the notebook's exact pipeline (feature engineering + heuristic
scoring), but fixes a real bug found during review: the original notebook
trained and predicted on the *same* data (`model.fit(X, y)` then
`model.predict(X)`), which makes any reported R2/MAE meaningless -- a model
that has already seen every row it's "evaluated" on will look artificially
good. This version adds a proper train/test split and reports metrics on
the held-out test set only.

Also: this evaluates how well a RandomForest can recover the hand-designed
heuristic score from engineered features -- it is explicitly NOT a real
default-risk prediction, since no real default/liquidation-outcome ground
truth exists in this dataset. That distinction is the whole point of
reporting it honestly rather than implying more than it measures.

Run: python build_and_eval.py
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path: str = "user-wallet-transactions.json") -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["amount"] = df["actionData"].apply(
        lambda x: float(x.get("amount", 0)) if isinstance(x, dict) and x.get("amount") else 0
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    actions = ["deposit", "borrow", "repay", "redeemunderlying", "liquidationcall"]
    features = df.groupby("userWallet").agg(
        total_txn=("action", "count"),
        total_volume=("amount", "sum"),
        active_days=("timestamp", lambda x: x.dt.date.nunique()),
        avg_amount=("amount", "mean"),
    ).reset_index()

    for action in actions:
        action_df = df[df["action"] == action].groupby("userWallet").agg({"amount": ["sum", "count"]})
        action_df.columns = [f"{action}_amount", f"{action}_count"]
        action_df = action_df.reset_index()
        features = features.merge(action_df, on="userWallet", how="left")

    features.fillna(0, inplace=True)
    return features


def create_heuristic_score(df: pd.DataFrame) -> "pd.Series":
    score = (
        df["deposit_amount"] * 0.2
        + df["repay_amount"] * 0.2
        - df["borrow_amount"] * 0.3
        - df["liquidationcall_count"] * 50
        + df["active_days"] * 2
    )
    scaled = MinMaxScaler((0, 1000)).fit_transform(score.values.reshape(-1, 1)).flatten()
    return scaled


def main() -> None:
    print("Loading real Aave V2 transaction data...")
    df = load_data()
    print(f"  {len(df)} raw transactions, {df['userWallet'].nunique()} unique wallets")

    features = engineer_features(df)
    features["score"] = create_heuristic_score(features)
    print(f"  {len(features)} wallets after feature engineering")

    X = features.drop(columns=["userWallet", "score"])
    y = features["score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print("\n--- Original notebook's approach: fit and 'evaluate' on the SAME data ---")
    same_data_r2 = r2_score(y_train, train_preds)
    print(f"R2 (train, same data as fit -- meaningless, shown to illustrate the bug): {same_data_r2:.4f}")

    print("\n--- Fixed: held-out test set (20%, never seen during training) ---")
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"R2 (test):  {test_r2:.4f}")
    print(f"MAE (test): {test_mae:.2f}")

    importances = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop 5 feature importances:")
    for name, imp in importances[:5]:
        print(f"  {name:20s} {imp:.4f}")


if __name__ == "__main__":
    main()
