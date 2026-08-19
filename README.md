Dataset - https://drive.google.com/file/d/19ZsPaQAAKVKKXgswXRl534YoSYCqWkbw/view?usp=sharing

𝐃𝐞𝐬𝐜𝐫𝐢𝐩𝐭𝐢𝐨𝐧: This project implements a machine learning-based credit scoring system specifically designed for decentralized finance (DeFi). It processes raw transaction data from the Aave protocol to evaluate user creditworthiness based on on-chain behavior such as deposits, borrows, and liquidations.

𝐊𝐞𝐲 𝐅𝐞𝐚𝐭𝐮𝐫𝐞𝐬:
- Automated Feature Engineering: Aggregates raw transaction data into meaningful financial metrics like total volume, active days, and average transaction amounts.
- Heuristic Scoring Logic: Implements a baseline scoring algorithm that rewards deposits/repayments and penalizes liquidations and heavy borrowing.
- Predictive Modeling: Utilizes a RandomForestRegressor to train a model capable of predicting credit scores based on historical wallet activity.
- Data Visualization: Includes visual analysis tools (using Seaborn and Matplotlib) to understand user distributions and risk profiles.

𝐓𝐞𝐜𝐡𝐧𝐢𝐜𝐚𝐥 𝐒𝐭𝐚𝐜𝐤:
- Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Environment: Jupyter Notebook / Google Colab

## Evaluation

**A real bug found and fixed:** the original notebook trained the RandomForestRegressor with `model.fit(X, y)` and then evaluated it with `model.predict(X)` — the same data twice. Any reported metric from that is meaningless; a model that has already seen every row it's "tested" on will always look artificially strong. `build_and_eval.py` reproduces the exact same pipeline on the real 100,000-transaction Aave V2 dataset (3,497 unique wallets), fixed with a proper 80/20 train/test split:

```
Original approach (fit and "evaluate" on the same data):
R2 (train, same data as fit): 0.9537   <- looks great, but is fake

Fixed (held-out 20% test set, never seen during training):
R2 (test):  0.4507
MAE (test): 0.42
```

That gap (0.95 vs. 0.45) is a real, honest illustration of what train/test leakage does to a reported metric — not a hypothetical warning, a measured before/after on this exact model and data.

**Important scope note, stated plainly:** 0.45 R² measures how well the model recovers the *hand-designed heuristic score* from engineered features — it is not a measure of real-world credit/default prediction, since there is no real default or liquidation-outcome ground truth in this dataset to validate against. Feature importances show `deposit_amount` dominates (0.78), which tracks with the heuristic's own weighting, not necessarily with real creditworthiness.

Run it: `python build_and_eval.py` (downloads nothing — place `user-wallet-transactions.json` from the dataset link above in the same folder).
