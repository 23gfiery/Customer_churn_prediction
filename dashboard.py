"""
dashboard.py

Streamlit dashboard for the customer churn analytics project.

Run with:
    streamlit run dashboard.py

Expects data/customers_with_risk_scores.csv to already exist
(run generate_synthetic_data.py -> segmentation.py -> churn_model.py first,
or use the Makefile / run_pipeline.py to do all steps at once).
"""

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

DATA_PATH = "data/customers_with_risk_scores.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

st.title("📊 Customer Churn & Segmentation Analytics")
st.caption("Behavioral segmentation and churn-risk scoring for retention strategy")

# ---------- KPI ROW ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Overall Churn Rate", f"{df['Exited'].mean():.1%}")
col3.metric("Avg. Balance", f"${df['Balance'].mean():,.0f}")
col4.metric("High-Risk Customers", f"{(df['ChurnRiskBand'] == 'High').sum():,}")

st.divider()

# ---------- SEGMENT BREAKDOWN ----------
left, right = st.columns(2)

with left:
    st.subheader("Segment Sizes")
    seg_counts = df["ClusterLabel"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig = px.bar(seg_counts, x="Segment", y="Count", color="Segment")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn Rate by Segment")
    seg_churn = df.groupby("ClusterLabel")["Exited"].mean().reset_index()
    seg_churn.columns = ["Segment", "Churn Rate"]
    fig2 = px.bar(seg_churn, x="Segment", y="Churn Rate", color="Segment")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------- TREND CHARTS ----------
st.subheader("Churn Drivers")
t1, t2, t3 = st.columns(3)

with t1:
    rate = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
    fig = px.bar(rate, x="NumOfProducts", y="Exited", title="Churn by # of Products")
    fig.update_yaxes(tickformat=".0%", title="Churn Rate")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    rate = df.groupby("Geography")["Exited"].mean().reset_index()
    fig = px.bar(rate, x="Geography", y="Exited", title="Churn by Geography")
    fig.update_yaxes(tickformat=".0%", title="Churn Rate")
    st.plotly_chart(fig, use_container_width=True)

with t3:
    rate = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
    rate["IsActiveMember"] = rate["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    fig = px.bar(rate, x="IsActiveMember", y="Exited", title="Churn by Activity Status")
    fig.update_yaxes(tickformat=".0%", title="Churn Rate")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- AT-RISK CUSTOMER TABLE ----------
st.subheader("At-Risk Customer Explorer")

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    risk_filter = st.multiselect(
        "Risk Band", options=["Low", "Medium", "High"], default=["High"]
    )
with filter_col2:
    segment_filter = st.multiselect(
        "Segment", options=sorted(df["ClusterLabel"].unique()),
        default=sorted(df["ClusterLabel"].unique())
    )
with filter_col3:
    geo_filter = st.multiselect(
        "Geography", options=sorted(df["Geography"].unique()),
        default=sorted(df["Geography"].unique())
    )

filtered = df[
    df["ChurnRiskBand"].isin(risk_filter)
    & df["ClusterLabel"].isin(segment_filter)
    & df["Geography"].isin(geo_filter)
].sort_values("ChurnRiskScore", ascending=False)

st.write(f"Showing {len(filtered):,} customers")
st.dataframe(
    filtered[[
        "CustomerId", "ClusterLabel", "Geography", "Age", "Tenure",
        "Balance", "NumOfProducts", "IsActiveMember", "ChurnRiskScore", "ChurnRiskBand"
    ]].reset_index(drop=True),
    use_container_width=True,
    height=400,
)

st.divider()

# ---------- BUSINESS RECOMMENDATIONS ----------
st.subheader("💡 Recommended Actions")
st.markdown("""
- **Customers with 3+ products show sharply higher churn** — review whether cross-sell/bundling is
  overwhelming customers rather than retaining them; consider a simplification or loyalty check-in
  for this group.
- **Inactive members churn at roughly 2x the rate of active members** — a re-engagement campaign
  (app nudges, relationship manager outreach) targeted at the "Dormant / Flight Risk" segment
  could meaningfully reduce losses.
- **Germany shows a higher churn rate than France or Spain** — worth investigating whether this is
  service-related, competitive, or a product-fit issue specific to that market.
- **At-Risk High-Balance customers** represent disproportionate revenue risk — prioritize this
  segment for proactive retention outreach even though their raw count is smaller than other
  segments.
""")
