# 5G Dense Deployment Throughput Prediction: Centralized vs. Simulated Vertical Federated Learning

[![Dataset](https://img.shields.io/badge/Dataset-Nature%20Scientific%20Data-blue.svg)](https://www.nature.com/articles/s41597-025-06282-0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)

---

## 📌 Project Overview

_Part of Network Data Analysis course exam_<br>

Predicting user throughput in dense 5G network environments (such as stadiums or event arenas) is critical for dynamic radio resource management, quality-of-service (QoS) guarantees, and network slice allocation. However, wireless telemetry data is heavily skewed, highly dynamic, and fragmented across different operational entities—specifically the **User Equipment (UE)** side and the **Radio Unit (RU)** side.

This repository presents an end-to-end machine learning framework that compares **Centralized Machine Learning** against **Simulated Vertical Federated Learning (VFL)** for user-level throughput prediction in the **ACC Arena** dense 5G deployment.

### 🎯 Key Question

> **Can user throughput be predicted accurately by partitioning features between UE-side and RU-side views, and can a Vertical Federated Learning (VFL) architecture preserve predictive power compared to centralized learning without sharing raw feature vectors?**

**Key Finding:** The **VFL-stacked LSTM+MLP** model achieves $R^2_{\text{log}} = 0.7106$ on unseen held-out test users, retaining **$\approx 98.6\%$** of the predictive performance of the Centralized LSTM+MLP ($R^2_{\text{log}} = 0.7210$), proving that vertical feature partitioning is highly viable for privacy-preserving wireless analytics.

---

## 📊 Dataset & Scenario Context

The dataset utilized in this project is sourced from the **Liverpool City Region High Density Demand (LCR HDD)** project, published in _Nature Scientific Data_:

- **Publication:** _5G High Density Demand Dataset in Liverpool City Region, UK_ ([Nature Scientific Data, DOI: 10.1038/s41597-025-06282-0](https://www.nature.com/articles/s41597-025-06282-0))
- **Target Venue:** **ACC Arena** (Dense 5G Event Venue Scenario)
- **User Scale:** 12,000 total simulated/measured user equipments (UEs)
- **Sampling Resolution:** Raw telemetry sampled every 3 seconds

```
      +-----------------------------------------------------------------------+
      |               ACC Arena Dense 5G Telemetry (3-sec)                     |
      +-----------------------------------------------------------------------+
                                          |
                                          v
      +-----------------------------------------------------------------------+
      |              Chunked Loading & 60-Second Aggregation                  |
      +-----------------------------------------------------------------------+
                     |                                         |
                     v                                         v
        UE-Side Feature View                      RU-Side Feature View
     [SINR, BLER, MobileState, Pos]           [RU_Load, Crowd_Density_5m]
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
                      Two-Stage VFL / Centralized Pipeline
             Stage 1: Active/Idle Classifier (Logistic / RF / MLP)
             Stage 2: Log-Throughput Regressor (Ridge / RF / LSTM)
```

---

## 🛠️ Data Preprocessing & Aggregation Strategy

To address computational constraints and temporal noise while preserving extreme radio conditions:

1. **Window Aggregation:** Raw 3-second samples are merged into 60-second non-overlapping user-level temporal windows across the full population of 12,000 users.
2. **Aggregation Functions:**
   - **Throughput:** $95^{\text{th}}$ percentile ($T_{q95}$) to capture sustainable peak rate.
   - **PRB Allocation:** $95^{\text{th}}$ percentile ($\text{PRB}_{q95}$).
   - **SINR (Uplink):** $90^{\text{th}}$ percentile ($\text{SINR}_{q90}$) to represent strong link quality.
   - **BLER:** $10^{\text{th}}$ percentile ($\text{BLER}_{q10}$) to capture minimal error baseline conditions.
   - **Position & Slower Metrics:** Temporal mean ($\mu_{\text{X}}, \mu_{\text{Y}}$).

---

## 🔍 Exploratory Data Analysis & Feature Engineering

### 1. Data Distribution Diagnostics

- **Target Skewness:** Throughput is heavily zero-inflated (idle periods) with long-tailed peaks. This necessitates evaluating models in both log-scale ($\log(1 + \text{Throughput})$) and raw Mbps scale.
- **Redundancy Reduction:** High collinearity between Uplink and Downlink SINR resulted in dropping `SINR_DL` to eliminate redundancy.
- **Infrastructure Imbalance:** Connection distributions across 33 Radio Units were highly asymmetric, prompting the engineered feature `RU_Current_Load` (concurrent connected users per timestamp).

### 2. Feature Taxonomy

| Domain              | Feature Name                       | Description                                                            | VFL Partition |
| :------------------ | :--------------------------------- | :--------------------------------------------------------------------- | :------------ |
| **Radio Quality**   | `SINR_UL`, `SINR_UL_q90`           | Uplink Signal-to-Interference-plus-Noise Ratio                         | UE-Side       |
|                     | `BLER`, `BLER_q10`                 | Block Error Rate                                                       | UE-Side       |
|                     | `Log_Effective_Signal_Robust`      | $\log(1 + \text{SINR}_{\text{linear}} / (\text{BLER} + \epsilon))$     | UE-Side       |
| **Infrastructure**  | `RU_Current_Load`                  | Concurrent active user connections per Radio Unit                      | RU-Side       |
| **Spatial / Crowd** | `X_meters`, `Y_meters`, `Velocity` | Spatial coordinates and movement speed                                 | UE-Side       |
|                     | `Crowd_Density_5m`                 | Count of UEs within a 5-meter spatial radius                           | RU-Side       |
| **Application**     | `MobileState`                      | One-hot encoded application activity (Idle, Video, Gaming, HTTP, etc.) | UE-Side       |

### 3. Special Feature Analysis & Leakage Checks

- **Crowd Density ($X$-meter Feature):** Evaluated $X \in \{2\text{m}, 5\text{m}, 10\text{m}\}$. $5\text{m}$ yielded the optimal validation score, balancing local granularity without spatial redundancy. Assigned to the RU-side view because local density represents network environment state.
- **PRB Leakage Risk Check:** Physical Resource Block (`PRB`) allocation showed near-direct correlation with throughput ($r = 0.529$). Including `PRB` improved validation $R^2_{\text{log}}$ from $0.7967$ to $0.8175$. However, because PRB is assigned by the scheduler post-demand, main clean baseline models **exclude PRB** to prevent data shortcutting/leakage.

---

## ⚙️ Experimental Design & Architecture

### User-Level Holdout Split

To prevent data leakage across temporal samples of the same user, samples are partitioned strictly by `User_ID` using `GroupKFold`:

- **Training / Cross-Validation Pool:** Users $0 \to 9,999$
- **Held-out Test Set:** Users $10,000 \to 11,999$ (2,000 completely unseen users)

### Two-Stage Prediction Pipeline

Because idle traffic dominates total time windows, all model families deploy a two-stage setup:
$$\text{Final Prediction} = \begin{cases} 0 & \text{if Stage 1 (Classifier) predicts Idle} \\ \exp(\hat{y}_{\text{log}}) - 1 & \text{if Stage 1 predicts Active} \end{cases}$$

```
                          +-------------------------+
                          |   Input Features (X)    |
                          +-------------------------+
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
        +-------------------------+         +-------------------------+
        |   Stage 1 Classifier    |         |   Stage 2 Regressor     |
        |   (Predicts Active/Idle)|         | (Predicts log(1+Thpt))  |
        +-------------------------+         +-------------------------+
                     |                                   |
                     v                                   v
             [ Is Active? ]                          [ log(1+Y) ]
                /        \                               |
          No   /          \  Yes                         v
              v            \---------------------> [* Expm1 *]
          [ 0 Mbps ]                                     |
                                                         v
                                                 [ Final Mbps ]
```

---

## 🤖 Model Families Explored

1. **Centralized Random Forest:** Two-stage RF (Classifier + Regressor) trained on combined UE + RU clean features.
2. **VFL-stacked Random Forest:** Independent UE-RF and RU-RF models combined by a Ridge/Logistic meta-coordinator receiving only local prediction vectors.
3. **Linear VFL Baseline:** Two-stage Logistic Regression + Ridge Regression under vertical partitioning.
4. **Centralized LSTM+MLP Neural Network:** Hybrid neural model with an LSTM branch processing 16-step temporal sequences of dynamic radio signals, fused with a static contextual MLP branch.
5. **VFL-stacked LSTM+MLP Neural Network:** Independent UE-side and RU-side LSTM+MLP models; outputs are fed to a coordinator neural network trained on meta-user splits.

---

## 📈 Final Results & Performance Evaluation

All models were evaluated on the aligned held-out test set of 2,000 unseen users.

### Unified Test Results Table

| Model Architecture            | $R^2_{\text{log}}$ ↑ | $\text{RMSE}_{\text{log}}$ ↓ | $\text{MAE}_{\text{log}}$ ↓ | $R^2_{\text{raw}}$ ↑ | $\text{RMSE}_{\text{raw}}$ (Mbps) ↓ | $\text{MAE}_{\text{raw}}$ (Mbps) ↓ |
| :---------------------------- | :------------------: | :--------------------------: | :-------------------------: | :------------------: | :---------------------------------: | :--------------------------------: |
| **Centralized LSTM+MLP**      |      **0.7210**      |          **0.3522**          |         **0.1214**          |        0.1863        |               7.1290                |             **0.9440**             |
| **VFL-Stacked LSTM+MLP**      |      **0.7106**      |          **0.3587**          |         **0.1225**          |      **0.2183**      |             **6.9873**              |               0.9278               |
| **Linear VFL Baseline**       |        0.6695        |            0.3833            |           0.1273            |        0.1153        |               7.4334                |               1.0006               |
| **UE-Only LSTM+MLP**          |        0.6437        |            0.3980            |           0.1383            |        0.1261        |               7.3877                |               1.0252               |
| **UE-Only Random Forest**     |        0.6172        |            0.4125            |           0.1452            |        0.1302        |               7.3705                |               1.0531               |
| **Centralized Random Forest** |        0.5849        |            0.4296            |           0.1528            |        0.1945        |               7.0927                |               1.1017               |
| **VFL-Stacked Random Forest** |        0.5817        |            0.4313            |           0.1530            |        0.1897        |               7.1141                |               1.1039               |
| **RU-Only Random Forest**     |        0.5572        |            0.4437            |           0.1575            |        0.1890        |               7.1172                |               1.1431               |
| **RU-Only LSTM+MLP**          |        0.4515        |            0.4939            |           0.1975            |        0.1964        |               7.0847                |               1.1258               |

### Key Trade-Offs & Takeaways

1. **Vertical Federated Learning Retention:**
   $$\text{Retention Ratio} = \frac{R^2_{\text{log}}(\text{VFL-LSTM})}{R^2_{\text{log}}(\text{Central-LSTM})} = \frac{0.7106}{0.7210} \approx \mathbf{98.56\%}$$
   The VFL-stacked neural network retains nearly all predictive accuracy while isolating UE radio metrics from RU load and spatial congestion data.

2. **Temporal Dynamics Matter:** The hybrid LSTM+MLP architectures significantly outperform Random Forest models ($R^2_{\text{log}} \approx 0.7210$ vs $0.5849$), demonstrating that tracking time-series variations in link quality (SINR/BLER) is crucial for accurate throughput estimation.

3. **Log vs. Raw Scale Performance:** While $R^2_{\text{log}}$ reaches $>0.72$, raw scale $R^2_{\text{raw}}$ remains around $\sim 0.20$. This occurs because rare, massive throughput spikes in dense networks introduce high variance on the raw scale, whereas log-scale metrics effectively capture generalized user experience across all operational states.

---

## 🛠️ Repository Structure

```
.
├── data/
│   ├── raw/                 # ACC Arena raw 3-second telemetry chunks
│   └── processed/           # 60-second aggregated user-time feature tables
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb    # Aggregation & distribution checks
│   ├── 02_feature_engineering.ipynb     # Spatial crowd density & log signals
│   ├── 03_rf_and_linear_vfl.ipynb        # Random Forest & Linear VFL models
│   └── 04_lstm_mlp_vfl_pipeline.ipynb   # Centralized & VFL Neural Networks
├── src/
│   ├── features.py          # Spatial radius search and signal robust math
│   ├── models_vfl.py        # Vertical FL coordinator models & local views
│   └── metrics.py           # Two-stage evaluation metrics (Log & Raw)
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

---
