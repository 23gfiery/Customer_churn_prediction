📊 Customer Churn & Segmentation Analytics

An end-to-end analytics pipeline that explores customer behavior, segments customers into meaningful groups, predicts churn risk, and surfaces it all in an interactive dashboard with concrete business recommendations.

Built to mirror the kind of work described in banking/fintech Business Analytics roles — gathering operational data, identifying patterns and trends, translating findings into insights, and recommending actions to support decision-making and retention strategy.


🔍 What it does


Exploratory Data Analysis — explores churn patterns across geography, product count, activity status, age, tenure, and balance.
Customer Segmentation — clusters customers into behavioral segments (e.g., High-Value Stable, At-Risk High-Balance, Dormant / Flight Risk) using k-means on balance, tenure, product count, activity, and salary.
Churn Risk Modeling — trains a Random Forest classifier (with Logistic Regression as a baseline) to score every customer's churn risk, evaluated on precision/recall/ROC-AUC rather than raw accuracy, since churn data is imbalanced.
Interactive Dashboard — a Streamlit app with KPIs, segment breakdowns, churn-driver charts, a filterable at-risk customer table, and written recommendations for stakeholders.



🗂️ Dataset

This repo ships with generate_synthetic_data.py, which creates a 10,000-row synthetic dataset matching the schema of the popular Kaggle Bank Customer Churn Prediction dataset — so you can run everything immediately without waiting on a download.

To use the real dataset instead:


Download it from Kaggle.
Save it as data/Churn_Modelling.csv with the same column names:
CustomerId, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited
Re-run the pipeline — nothing else needs to change.



⚙️ Setup

bashgit clone https://github.com/<your-username>/churn-analytics.git
cd churn-analytics
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt


🚀 Run the full pipeline

bashpython run_pipeline.py

This runs, in order:
generate synthetic data → EDA → segmentation → churn modeling,
saving intermediate CSVs to data/ and chart images to outputs/.

Or run steps individually:

bashpython generate_synthetic_data.py   # creates data/Churn_Modelling.csv
python eda.py                       # prints stats, saves charts to outputs/
python segmentation.py              # creates data/customers_segmented.csv
python churn_model.py               # creates data/customers_with_risk_scores.csv


📈 Launch the dashboard

bashstreamlit run dashboard.py

Opens at http://localhost:8501.


📁 Project structure

churn-analytics/
├── generate_synthetic_data.py   # creates a realistic demo dataset
├── eda.py                       # exploratory data analysis + charts
├── segmentation.py              # k-means customer segmentation
├── churn_model.py               # churn risk classifier + feature importance
├── dashboard.py                 # Streamlit dashboard
├── run_pipeline.py              # runs everything in order
├── requirements.txt
├── README.md
├── data/                        # generated + intermediate CSVs (gitignored)
└── outputs/                     # EDA chart images (gitignored)


🧠 Key findings

(from the synthetic dataset — re-run on real data for actual figures)

DriverFindingProduct countCustomers with 3+ products churn at roughly 2x the rate of customers with 1–2 productsActivity statusInactive members churn at roughly 2x the rate of active membersGeographyGermany shows a noticeably higher churn rate than France or SpainSegment riskA small "At-Risk High-Balance" segment carries disproportionate revenue risk relative to its size


🛠️ Tech stack

Python · pandas · scikit-learn · matplotlib / seaborn · Streamlit · Plotly


💡 Sample business recommendation


Customers with 3+ products show sharply higher churn — worth investigating whether cross-sell/bundling is overwhelming customers rather than retaining them. A targeted retention check-in for this segment could reduce losses before they escalate.




📝 License

This project is open for personal/educational use. Feel free to fork and adapt.
