# CustoraIQ — Customer & CX Intelligence Platform

**Customer Segmentation · Churn Prediction · Health Scoring · Revenue Action Planning**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?logo=scikit-learn&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-1.1.3-F4C519?logo=duckdb&logoColor=black)
![SHAP](https://img.shields.io/badge/SHAP-0.45.1-8A2BE2)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

**[Live Dashboard](https://custoraiq-4r8ryklwgjkwcdhrpzmf3v.streamlit.app)** · **[GitHub Repo](https://github.com/tejasjd69/custoraiq)**

---

## What It Does

CustoraIQ turns 1M+ raw retail transactions into a customer-level intelligence layer — segmenting customers with RFM analysis, predicting churn risk, scoring customer health, and generating prioritized retention and growth actions, all through a 7-tab interactive dashboard.

**Business problem it solves:** Given raw transaction history, which customers are most valuable, which are at risk of churning, and what action should be taken for each?

## Key Results

| Metric | Result |
|---|---|
| Transactions processed | 1,067,371 raw → 779,425 clean |
| Customers analyzed | 5,878 |
| Total revenue | £17.37M |
| Churn model performance | ROC-AUC **0.81** (Logistic Regression, beat RF & XGBoost) |
| Revenue concentration | Champions = 22% of customers, **68% of revenue** |
| Retention targets flagged | 2,952 customers · 874 Critical Risk |
| SQL/Python reconciliation | £0.00 difference (fully validated) |

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, pandas, NumPy |
| Storage | Parquet, CSV |
| Machine Learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| SQL Analytics | DuckDB |
| Dashboard | Streamlit, Plotly |

## Pipeline

Raw Excel Data
↓
Data Cleaning & Quality Audit → 779,425 clean transactions
↓
Feature Engineering → 27 features × 5,878 customers
↓
RFM Segmentation → 10 business segments
↓
Churn Labeling (time-window split) → 5,041 labeled customers
↓
Model Training & SHAP Explainability → LR selected (ROC-AUC 0.81)
↓
Customer Health Scoring → customer_360.parquet (46 columns)
↓
DuckDB SQL Analytics Layer → 10 validated reporting tables
↓
Streamlit Dashboard → 7-tab interactive app


## Key Findings

**RFM Segments (top 3 by revenue):**

| Segment | Customers | Revenue Share |
|---|---|---|
| Champions | 1,297 (22%) | 68.3% |
| Loyal Customers | 650 (11%) | 10.6% |
| Cannot Lose | 223 (4%) | 5.7% |

**Churn Risk Tiers:**

| Tier | Probability | Customers | Action |
|---|---|---|---|
| Critical | ≥80% | 874 | Immediate outreach |
| High | 60–79% | 1,256 | Win-back email within 7 days |
| Medium | 40–59% | 1,387 | Monitor & nurture |
| Low | <40% | 1,524 | Loyalty & cross-sell |

**Top churn drivers (SHAP):** longer inactivity (recency) increases churn risk; more purchase days and higher RFM/monetary scores reduce it.

## Getting Started

```bash
git clone https://github.com/tejasjd69/custoraiq.git
cd custoraiq
pip install -r requirements.txt
```

Download the UCI Online Retail II dataset (https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it at `data/raw/online_retail_II.xlsx`, then run the pipeline in order:

```bash
python src/01_load_and_inspect.py
python src/02_data_quality_audit.py
python src/03_clean_data.py
python src/04_verify_clean_outputs.py
python src/05_eda.py
python src/06_build_customer_features.py
python src/07_rfm_segmentation.py
python src/08_churn_labeling.py
python src/09_train_churn_model.py
python src/10_shap_explainability.py
python src/11_customer_health_score.py
python src/12_sql_analytics_layer.py
```

Then launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Folder Structure

custoraiq/
├── app/streamlit_app.py # 7-tab dashboard
├── data/{raw,interim,processed}
├── models/ # Trained model + metrics
├── reports/{figures,sql_outputs}
├── sql/ # 10 DuckDB queries
├── src/ # 12-step pipeline scripts
└── requirements.txt


## Business Recommendations

1. **Protect Champions & Loyal Customers** — they drive ~79% of revenue; prioritize loyalty rewards and early access.
2. **Urgent outreach for Critical Risk** — 874 customers at ≥80% churn probability need contact within 48 hours.
3. **Win-back High Risk customers** — 1,256 customers at 60–79% probability; targeted email within 7 days.
4. **Convert one-time buyers** — this group churns at 75.5%; a timely second-purchase incentive helps retention.
5. **Automate low-value nurture** — Hibernating/Low Value segments don't justify manual outreach cost.

## Limitations

- Historical dataset (Dec 2009–Dec 2011); would need recalibration on current data for production use.
- Churn is an engineered label (purchase inactivity), not ground truth — seasonal effects could create false positives.
- No demographic or campaign-response data available.
- Portfolio/educational project — not connected to live data or production systems.

## Author

**Tejas Jadhav**
Data Analyst · Business Analytics · Financial Data · AI Applications

GitHub: https://github.com/tejasjd69 · LinkedIn: https://linkedin.com/in/tejas-jadhav
