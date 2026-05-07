# Customer Lifetime Value Prediction — Probabilistic Modeling

Predicts 6-month and 12-month Customer Lifetime Value (CLV) for an online retailer using probabilistic models, followed by value-based customer segmentation to enable targeted marketing strategy.

---

## 🎯 Business Problem

Not all customers are equal. This project answers: **how much is each customer worth over the next 6–12 months?** — enabling smarter decisions on acquisition spend, retention investment, and customer prioritization.

---

## 🧠 Approach

Rather than using simple historical averages, this project uses **probabilistic models** that account for the stochastic nature of customer purchase behavior:

### BG/NBD Model (Purchase Frequency)
The **Beta-Geometric / Negative Binomial Distribution** model predicts the expected number of future transactions for each customer, given their historical recency, frequency, and tenure (T).

- Models two latent processes: *how often a customer buys* and *when they churn*
- Handles customers who appear inactive but may still return

### Gamma-Gamma Model (Monetary Value)
Estimates the **expected average transaction value** for future purchases.

- Assumes no correlation between purchase frequency and spend (validated via correlation check)
- Combined with BG/NBD output to yield CLV

### CLV Formula
```
CLV = BG/NBD predicted purchases × Gamma-Gamma expected spend × discount factor
```

---

## 📊 Segmentation

Customers are bucketed into four tiers based on predicted 6-month CLV:

| Segment | Description | Action |
|---|---|---|
| **Champions** | Highest CLV quartile | Reward & retain |
| **Loyal Customers** | High-value, consistent | Upsell opportunities |
| **Need Attention** | Mid-tier, at risk | Re-engagement campaigns |
| **Hibernating** | Lowest CLV quartile | Win-back or deprioritize |

---

## ⚙️ Pipeline

```
Raw Transactions (Online_Retail.csv)
        │
        ▼
Data Cleaning (returns, nulls, outlier capping)
        │
        ▼
RFM Feature Engineering (recency, frequency, monetary, T)
        │
        ├──▶ BetaGeoFitter → P(future purchases)
        └──▶ GammaGammaFitter → E(avg transaction value)
                │
                ▼
        CLV = purchases × value × discount rate
                │
                ▼
        Customer Segmentation (quartile-based)
```

---

## 🛠 Stack

- **lifetimes** — BG/NBD and Gamma-Gamma model implementations
- **pandas, NumPy** — data processing and RFM computation
- **matplotlib, seaborn** — frequency/recency matrix visualization and segment analysis
- **datetime** — cohort date calculations

---

## 📁 Structure

```
├── Customer_Lifetime_Value.ipynb    # Full analysis and modeling notebook
├── Online_Retail.csv                # Transaction dataset
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
pip install lifetimes pandas numpy matplotlib seaborn
jupyter notebook Customer_Lifetime_Value.ipynb
```

---

## 📌 Related Projects

- [A/B Testing](https://github.com/Ajay-Deshpande/A-B-testing) — product experimentation and retention analysis
- [Expected Credit Loss](https://github.com/Ajay-Deshpande/Expected-Credit-Loss) — probabilistic financial risk modeling
