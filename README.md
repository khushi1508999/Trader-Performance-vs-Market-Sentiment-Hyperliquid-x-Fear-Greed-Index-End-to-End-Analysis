# Trader-Performance-vs-Market-Sentiment-Hyperliquid-x-Fear-Greed-Index-End-to-End-Analysis
# Trader Performance vs Market Sentiment Analysis
### Primetrade.ai — Data Science Intern Assignment (Round 0)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project investigates how **Bitcoin market sentiment** (Fear & Greed Index) influences **trader behavior and performance** on the Hyperliquid decentralized exchange. Using 211,224 trade records across 32 unique accounts spanning May 2023 – May 2025, the analysis uncovers actionable patterns that can inform smarter, sentiment-aware trading strategies.

**Key questions answered:**
- Do traders earn more on Fear days or Greed days?
- Does sentiment shift trader behavior — leverage, frequency, long/short bias?
- Which trader segments are most vulnerable to adverse sentiment?
- Can we predict next-day profitability from sentiment + behavioral signals?

---

## Project Structure

```
├── Project_0.ipynb              # Main analysis notebook (end-to-end)
├── Report.pdf                   # Detailed explanation about the methodology, insights and strategic recommendations
├── README.md                    # How to run?
└── charts/                      
    ├── fig1_eda_distributions.png
    ├── fig2_coins_directions.png
    ├── fig3_performance_fear_greed.png
    ├── fig4_behavior_sentiment.png
    ├── fig5_leverage_segments.png
    ├── fig6_frequency_segments.png
    ├── fig7_strategies.png
    ├── fig8_model.png
    └── fig9_clusters.png
```

## Datasets

| Dataset | Source | Rows | Columns | Period |
|---------|--------|------|---------|--------|
| `historical_data.csv` | Hyperliquid DEX | 211,224 | 16 | May 2023 – May 2025 |
| `fear_greed_index.csv` | Alternative.me | ~730 | 4 | Overlapping period |

**Trader Data Columns:** `Account`, `Coin`, `Execution Price`, `Size Tokens`, `Size USD`, `Side`, `Timestamp IST`, `Start Position`, `Direction`, `Closed PnL`, `Transaction Hash`, `Order ID`, `Crossed`, `Fee`, `Trade ID`, `Timestamp`

**Fear & Greed Columns:** `timestamp`, `value`, `classification`, `date`

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/primetrade-sentiment-analysis.git
cd primetrade-sentiment-analysis
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

Or install all at once:

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
scikit-learn>=1.1.0
jupyter>=1.0.0
```

### 4. Place Data Files

Ensure both CSV files are in the **same directory** as the notebook:

```
project-root/
├── Project_0.ipynb
├── historical_data.csv          ← required
└── fear_greed_index.csv         ← required
```

> **Note:** The notebook originally mounted Google Drive (for Colab). For local execution, the data-loading cell has been updated to read CSVs directly from the working directory. If running on Colab, re-mount Drive and update the paths accordingly.

---

## How to Run

### Option A — Jupyter Notebook (Recommended)

```bash
jupyter notebook Project_0.ipynb
```

Then: **Kernel → Restart & Run All**

The notebook runs sequentially from top to bottom. All 9 figures are saved as PNG files in the working directory automatically.

### Option B — JupyterLab

```bash
jupyter lab
```

Open `Project_0.ipynb` from the file explorer and run all cells.

### Option C — Google Colab

1. Upload `Project_0.ipynb`, `historical_data.csv`, and `fear_greed_index.csv` to your Google Drive
2. Open the notebook in Colab
3. In the data-loading cell, restore the `drive.mount` block and update paths to match your Drive structure
4. Run all cells

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| **1. Environment Setup** | Library imports, color palette, plot configuration |
| **2. Data Loading & Quality Audit** | Load CSVs, document shape, missing values, duplicates, dtype inspection |
| **3. Preprocessing & Feature Engineering** | Timestamp parsing, sentiment consolidation (5-class → binary), leverage proxy derivation, long/short flags |
| **4. EDA** | PnL distribution, trade size, leverage, trader concentration, coin/direction breakdown |
| **5. Part A — Daily Metrics** | Account-day aggregates: PnL, win rate, trade count, leverage, long ratio, fees |
| **6. Part B — Sentiment vs Performance** | Fear vs Greed PnL comparison (t-test + Mann-Whitney U), drawdown proxy, win rate |
| **7. Part B — Behavioral Analysis** | Leverage, trade frequency, long/short ratio, position size across sentiment regimes |
| **8. Part B — Trader Segmentation** | High/low leverage segments × sentiment; frequent/infrequent × sentiment; consistent/inconsistent winners |
| **9. Part C — Strategy Recommendations** | Two evidence-backed trading rules with visualizations |
| **10. Bonus — Predictive Model** | GBM classifier (5-fold CV ROC-AUC) predicting next-day profitability |
| **11. Bonus — Clustering** | K-Means (k=4) behavioral archetypes, PCA visualization |
| **12. Summary** | All insights in one place, strategy recap |

**Expected runtime:** ~2–4 minutes on a standard laptop.

---

## Key Results at a Glance

| Finding | Result |
|---------|--------|
| Avg daily PnL — Greed days | Higher than Fear (statistically significant) |
| Avg daily PnL — Fear days | Lower; heavy-tailed losses in high-leverage traders |
| Leverage behavior | Traders do **not** reduce leverage during Fear (risk gap) |
| Long bias | Increases during Greed (momentum following) |
| Trade frequency | Higher in Greed; momentum traders suffer on Fear days |
| Predictive model ROC-AUC | Meaningfully above 0.5 (5-fold CV) |
| Dominant predictive features | Lagged PnL, rolling win-rate, `fg_score` (sentiment) |
| Behavioral clusters | 4 archetypes: High-Lev/HF Profitable, Low-Lev/LF Steady, High-Lev Struggling, Mixed |

---

## Strategy Rules (Summary)

**Rule 1 — Dynamic Leverage Control**
> When Fear & Greed Index < 47: cap leverage at **50% of baseline**. When FG > 55: allow up to **1.2× baseline**. Protects against fear-regime stop-outs while maximizing Greed-momentum upside.

**Rule 2 — Activity-Sentiment Filter**
> Frequent/momentum traders: increase activity **only when FG > 55** (confirmed Greed). Infrequent traders: prefer **Extreme Fear (FG < 25)** entries with tight sizing for mean-reversion. Do not increase trade frequency during Fear — data shows it destroys PnL for momentum-style traders.

---

## Reproducibility

- **No random seeds assumed** — all stochastic operations (KMeans, GBM, StratifiedKFold) use `random_state=42`
- **No external data** — everything is derived from the two provided CSVs
- All outputs (charts, tables) regenerate identically on re-run

---

## Author

**Khushi**
Data Science Intern Candidate — Primetrade.ai Round 0
