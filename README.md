# 🐾 DogNap — Pet Care Market Analytics Dashboard

A production-grade Streamlit analytics dashboard analyzing Indian dog owner survey data (800 respondents) using Machine Learning, Clustering, Association Rules, and Regression to predict pet care app adoption likelihood.

## 📸 Features

- **🏠 Home & Overview** — Key metrics, distribution analysis, spend breakdowns
- **📊 Dataset Exploration** — Data quality, transformation pipeline, correlation heatmaps
- **📉 EDA & Statistics** — Distribution tests, cross-tabulations, chi-square, heatmaps
- **🎯 Classification Models** — 9 algorithms with accuracy, precision, recall, F1, AUC-ROC
- **🔮 Clustering Analysis** — K-Means + Hierarchical with elbow method, silhouette, PCA
- **🔗 Association Rules** — Apriori algorithm with support, confidence, lift
- **📈 Regression Analysis** — Linear, Ridge, Lasso, ElasticNet, Gradient Boosting with R²
- **⚔️ Model Comparison** — Head-to-head across all algorithms
- **📋 Summary & Takeaways** — Business insights and academic rubric coverage
- **📥 Download Center** — Export all datasets and results

## 🚀 Quick Start

### Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/dognap-dashboard.git
cd dognap-dashboard
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy!

## 📊 Dataset

Indian Dog Owner Survey — 800 respondents across 5 regions:
- `age_group`: 18-24, 25-34, 35-44, 45-54, 55+
- `region`: North, South, East, West, Central
- `monthly_spend_inr`: Monthly pet care spending (₹)
- `num_dogs`: Number of dogs owned (1-4)
- `num_services_used`: Pet services used (1-6)
- `ownership_years`: Dog ownership duration
- `app_use_likelihood`: Target — Yes/Maybe/No

## 🎓 Academic Rubric Coverage

| Deliverable | Marks | Status |
|---|---|---|
| 4a — Classification algorithms | 10 | ✅ 9 algorithms with full metrics |
| 4b — Clustering | 10 | ✅ K-Means + Hierarchical |
| 4c — Association Rules / Regression | 10 | ✅ Both included |
| 5 — Report | 10 | ✅ Interactive dashboard |
| 6 — Presentation | 20 | ✅ Dashboard flow |

## ⚠️ Disclaimer

Academic project only. Not business advice.
