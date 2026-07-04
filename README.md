# Customer Churn & Segmentation Analytics

An end-to-end analytics project: explore customer behavior, segment customers
into meaningful groups, predict churn risk, and surface it all in an
interactive dashboard with concrete business recommendations.

Built to mirror the kind of work described in banking/fintech Business
Analytics Analyst roles — gathering operational data, identifying patterns
and trends, translating findings into insights, and recommending actions to
support decision-making and retention strategy.

## What it does

1. **EDA** — explores churn patterns across geography, product count,
   activity status, age, tenure, and balance.
2. **Segmentation** — clusters customers into behavioral segments
   (e.g., "High-Value Stable", "At-Risk High-Balance", "Dormant / Flight Risk")
   using k-means on balance, tenure, product count, activity, and salary.
3. **Churn risk model** — trains a Random Forest classifier (with Logistic
   Regression as a baseline) to score every customer's churn risk, evaluated
   on precision/recall/ROC-AUC rather than raw accuracy (churn data is
   imbalanced, so accuracy alone is misleading).
4. **Dashboard** — a Streamlit app with KPIs, segment breakdowns, churn-driver
   charts, a filterable at-risk customer table, and written recommendations.

## Dataset

This repo ships with `generate_synthetic_data.py`, which creates a 10,000-row
synthetic dataset matching the schema of the well-known Kaggle
["Bank Customer Churn Prediction"](https://www.kaggle.com/datasets) dataset,
so you can run everything immediately.

To use the real dataset instead: download it from Kaggle, save it as
`data/Churn_Modelling.csv` with the same column names
(`CustomerId, CreditScore, Geography, Gender, Age, Tenure, Balance,
NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited`), and
re-run the pipeline — nothing else needs to change.

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python run_pipeline.py
```

This runs data generation (if needed) → EDA → segmentation → churn modeling,
in order, and saves all intermediate outputs to `data/` and chart images to
`outputs/`.

Or run steps individually:

```bash
python generate_synthetic_data.py   # creates data/Churn_Modelling.csv
python eda.py                       # prints stats, saves charts to outputs/
python segmentation.py              # creates data/customers_segmented.csv
python churn_model.py               # creates data/customers_with_risk_scores.csv
```

## Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`.

## Project structure

```
churn_project/
├── generate_synthetic_data.py   # creates a realistic demo dataset
├── eda.py                       # exploratory data analysis + charts
├── segmentation.py              # k-means customer segmentation
├── churn_model.py               # churn risk classifier + feature importance
├── dashboard.py                 # Streamlit dashboard
├── run_pipeline.py              # runs everything in order
├── requirements.txt
├── data/                        # generated + intermediate CSVs
└── outputs/                     # EDA chart images
```

## Key findings (on the synthetic data — re-run on real data for actual figures)

- Customers with 3+ products churn at roughly **2x** the rate of customers
  with 1-2 products — worth investigating whether cross-sell is helping or
  overwhelming customers.
- Inactive members churn at roughly **2x** the rate of active members.
- Germany shows a noticeably higher churn rate than France or Spain.
- A small "At-Risk High-Balance" segment represents disproportionate revenue
  risk relative to its size — a good target for proactive retention outreach.

## Suggested resume bullet

> Built an end-to-end churn analytics pipeline segmenting 10K+ customers by
> behavior, scoring individual churn risk with a Random Forest classifier,
> and translating findings into an interactive dashboard with retention
> recommendations for stakeholders.
