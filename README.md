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
