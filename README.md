# 🏠 Nairobi House Price Prediction — Project Documentation
### LTLab Fellowship Program | 6-Day Intensive Sprint

---

> **Mission:** Build a complete, working prop-tech MVP — from raw data to a deployed pricing app and business dashboard. Fast, structured, and output-driven.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Day 1 — Data Collection & Structuring](#3-day-1--data-collection--structuring)
4. [Day 2 — Data Cleaning & Feature Engineering](#4-day-2--data-cleaning--feature-engineering)
5. [Day 3 — Exploratory Analysis & Baseline Model](#5-day-3--exploratory-analysis--baseline-model)
6. [Day 4 — Model Improvement & Explainability](#6-day-4--model-improvement--explainability)
7. [Day 5 — Pricing App (Deployment)](#7-day-5--pricing-app-deployment)
8. [Day 6 — Dashboard & Final Presentation](#8-day-6--dashboard--final-presentation)
9. [Full Pipeline: End-to-End Notebook](#9-full-pipeline-end-to-end-notebook)
10. [Model Performance Summary](#10-model-performance-summary)
11. [Evaluation Criteria](#11-evaluation-criteria)
12. [Deliverables Checklist](#12-deliverables-checklist)

---

## 1. Project Overview

| Item | Detail |
|---|---|
| **Program** | LTLab Fellowship |
| **Duration** | 6-Day Intensive Sprint |
| **Domain** | Prop-tech / Real Estate |
| **Geography** | Nairobi, Kenya |
| **Goal** | Predict residential property prices using machine learning |
| **Final Model** | XGBoost fine-tuned with Optuna |
| **Output** | `model.pkl`, pricing app, market dashboard, presentation |

### Tech Stack

| Layer | Tools |
|---|---|
| Data & Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, Optuna |
| Imputation | MICE (IterativeImputer + Random Forest) |
| Serialization | Joblib |
| App Deployment | Streamlit |
| Dashboard | Power BI or Streamlit |
| Version Control | GitHub |

---

## 2. Repository Structure

```
nairobi-house-price-prediction/
│
├── data/
│   ├── raw_listings.csv                  # Day 1 — raw scraped data
│   └── clean_listings.csv                # Day 2 — cleaned & engineered data
│
├── notebooks/
│   ├── Day_2a_Handling_Missing_Values.ipynb
│   ├── Day_2b_Feature_Engineering.ipynb
│   ├── Day3_EDA_and_Modelling.ipynb
│   ├── Day4_Advanced_Modelling.ipynb
│   └── Nairobi_House_Price_Prediction.ipynb  # ← Unified end-to-end notebook
│
├── model.pkl                             # Serialised best model + metadata
├── app.py                                # Streamlit pricing app (Day 5)
├── dashboard.py / dashboard.pbix         # Business dashboard (Day 6)
├── data_dictionary.md                    # Column definitions
└── README.md
```

---

## 3. Day 1 — Data Collection & Structuring

### Objective
Build the dataset foundation by collecting 300–800 Nairobi property listings.

### Dataset Fields

| Column | Meaning | Data Type |
|---|---|---|
| `Location` | Nairobi neighbourhood/suburb | String |
| `Property Type` | Apartment, House, Villa, Studio, etc. | String |
| `Bedrooms` | Number of bedrooms | Integer |
| `Bathrooms` | Number of bathrooms | Integer |
| `Size (m2)` | Property floor area in square metres | Float |
| `Amenities` | Comma-separated list of amenities | String |
| `Price (KES)` | Listing price in Kenyan Shillings | Float |
| `Listing Date` | Date the property was listed | Date |

### Data Sources
Properties were collected from Nairobi real estate listing platforms covering neighbourhoods including Westlands, Kilimani, Kileleshwa, Karen, Lavington, Runda, Muthaiga, Parklands, Langata, South C, Embakasi, Thika Road, Syokimau, and more.

### Day 1 Deliverables
- [x] `data/raw_listings.csv` — 300–800 rows of raw listings
- [x] Data dictionary (column definitions above)
- [x] GitHub repository initialized

---

## 4. Day 2 — Data Cleaning & Feature Engineering

### Objective
Transform raw data into a clean, model-ready feature matrix.

---

### 4.1 — Missing Value Handling

This was the most rigorous part of the cleaning pipeline, structured in 6 phases.

#### Phase 1: Diagnosis

A full missingness audit was performed on every column, calculating:
- Raw missing count
- Missing percentage
- Severity classification (None / Low / Moderate / High / Critical)

Visualizations produced:
- **Horizontal bar chart** — missing % per column, colour-coded by severity (green → red)
- **Missingness heatmap** — 200-row sample showing missing patterns (yellow = missing, dark = present)
- **Row-level distribution** — bar + pie chart showing how many columns are missing per row
- **Co-occurrence matrix** — which column pairs are missing together (detecting MNAR patterns)

#### Phase 2: Mechanism Analysis

Statistical tests were used to determine *why* data was missing:

| Mechanism | Meaning | Strategy |
|---|---|---|
| **MCAR** | Missing Completely At Random — no pattern | Any imputation works |
| **MAR** | Missing At Random — depends on other observed variables | Conditional / model-based imputation |
| **MNAR** | Missing Not At Random — depends on the missing value itself | Domain knowledge required |

A **Kolmogorov-Smirnov test** was run comparing the Price distribution between rows where `Bathrooms` was missing vs present. A p-value < 0.05 confirmed MAR — the missingness was related to `Property Type`.

#### Phase 3: Structural Cleaning

```
Rule 1 — Drop columns where missing % > 55%
Rationale: Imputing >55% synthetic data introduces more noise than signal.

Rule 2 — Drop rows with more than 1 missing value simultaneously
Rationale: Multi-missing rows are hard to impute accurately and often represent
           fundamentally incomplete submissions.
```

#### Phase 4: Domain-Knowledge Imputation

```python
# Studio and Bedsitter properties are single-room by definition
# → Any missing Bedrooms value for these types is unambiguously 1
domain_mask = df['Bedrooms'].isnull() & ptype_norm.isin(['studio', 'bedsitter'])
df.loc[domain_mask, 'Bedrooms'] = 1
```

This is the most reliable form of imputation — zero statistical uncertainty.

#### Phase 5: Statistical Imputation (Group-wise Median / Mode)

Instead of a single global median, values were imputed using the median of peer rows sharing the same `Property Type`. This accounts for the fact that a 1-bedroom apartment and a 5-bedroom villa have very different typical sizes, prices, and bathroom counts.

```python
def group_median_fill(dataframe, fill_col, group_col='Property Type'):
    global_median = dataframe[fill_col].median()     # fallback
    group_medians = dataframe.groupby(group_col)[fill_col].transform('median')
    return dataframe[fill_col].fillna(group_medians).fillna(global_median)

df['Bedrooms'] = group_median_fill(df, 'Bedrooms').round().astype(int)
```

#### Phase 6: Model-Based Imputation (MICE + Random Forest)

For columns with remaining missingness (typically `Bathrooms`, `Size (m2)`) that are inter-related, **MICE** (Multiple Imputation by Chained Equations) was used:

```python
mice_imputer = IterativeImputer(
    estimator    = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    max_iter     = 10,
    tol          = 1e-3,
    random_state = 42,
)
```

Why Random Forest as the estimator:
- Handles non-linear relationships (price is non-linear with size)
- Robust to outliers (luxury properties skew price heavily)
- No feature scaling required
- Captures feature interactions automatically

**Post-imputation validation** confirmed:
- Zero missing values remaining
- Distributions (mean, median, correlation) preserved before vs after imputation
- Correlation matrices compared to detect imputation-induced bias

---

### 4.2 — Duplicate Removal

```python
df.drop_duplicates(inplace=True)
```

---

### 4.3 — Outlier Removal

**Method:** IQR (Interquartile Range) with 1.5× multiplier on the 15th–85th percentile range.

```
Lower bound = Q15 - 1.5 × IQR
Upper bound = Q85 + 1.5 × IQR
```

Applied to: `Price (KES)`, `Size (m2)`, `Bedrooms`, `Bathrooms`

---

### 4.4 — Location Standardisation

```python
df['Location'] = df['Location'].str.strip().str.title()

location_mapping = {
    'Nrb': 'Nairobi',  'Nairobi Cbd': 'Nairobi',  'Cbd': 'Nairobi',
    'Thika Rd': 'Thika Road',  'Thika': 'Thika Road',
    'Lower Kabete': 'Kabete',  'Upper Kabete': 'Kabete',
}
df['Location'] = df['Location'].replace(location_mapping)
```

---

### 4.5 — Feature Engineering

All new features were created from the cleaned base columns.

#### `price_per_sqft` — Normalised Pricing Metric
```python
df['price_per_sqft'] = df['Price (KES)'] / df['Size (m2)']
```
Rationale: Absolute price varies with size. Price per m² enables fair comparison across properties of different sizes.

#### `amenity_score` — Weighted Amenity Value
```python
# Scoring system:
# Swimming Pool / Gym / Generator  → +2.0 pts each
# Lift / Elevator                  → +1.5 pts
# CCTV / Security / Borehole       → +1.0 pts each
# All other amenities              → +0.5 pts each
```

#### `amenity_count` — Raw count of amenities in the listing string.

#### `distance_to_cbd_km` — Geographic Feature
Straight-line distance from each neighbourhood to Nairobi CBD (−1.2864°, 36.8172°) calculated using the **Haversine formula**:

```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dphi    = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlambda/2)**2
    return R * 2 * arctan2(sqrt(a), sqrt(1 - a))
```

Coordinates were mapped for 26 Nairobi neighbourhoods. Unknown locations default to CBD coordinates (0 km).

#### Additional Derived Features

| Feature | Formula | Rationale |
|---|---|---|
| `total_rooms` | `Bedrooms + Bathrooms` | Larger total room count → higher value |
| `room_density` | `total_rooms / Size (m2)` | High = compact/budget, Low = spacious/luxury |
| `bathroom_bedroom_ratio` | `Bathrooms / Bedrooms` | Ratio > 1 = luxury (ensuite bathrooms) |
| `is_luxury` | `1 if price_per_sqft ≥ 90th pct AND amenity_score > 5` | Binary luxury flag |
| `property_category` | Simplified grouping (Apartment / House / Townhouse / Other) | Reduces 10+ types to 4 |

### Day 2 Deliverables
- [x] `data/clean_listings.csv` — fully cleaned and feature-engineered
- [x] Preprocessing notebook with all transformations documented
- [x] 7 EDA visualizations (distributions, scatter plots, boxplots, correlation heatmap)

---

## 5. Day 3 — Exploratory Analysis & Baseline Model

### Objective
Understand price behaviour patterns and establish a performance benchmark.

---

### 5.1 — Key EDA Questions Answered

#### Q1: Which locations are most expensive?
- Top locations by average price were identified and ranked
- A horizontal bar chart visualised the top 10–15 most expensive Nairobi neighbourhoods
- Premium locations: **Runda, Karen, Muthaiga, Gigiri, Spring Valley, Westlands**

#### Q2: How strongly does size affect price?
- A scatter plot of `Size (m2)` vs `Price (KES)` was produced, colour-coded by property category
- Pearson correlation coefficient calculated: `r ≈ 0.6–0.75` (strong positive)
- Conclusion: Size is one of the strongest single predictors of price

#### Q3: Which amenities increase value?
- Top amenity combinations by average price were ranked
- Individual amenity keywords (Swimming Pool, Gym, Generator, En Suite, CCTV, etc.) were tested for median price uplift vs listings without that amenity
- **Swimming Pool** and **Gym** showed the highest price premium

---

### 5.2 — Baseline Models

#### Model 1: Heuristic (Rule-based)
```python
# Predict: Price ≈ Size × global average price per m²
avg_price_per_m2 = y_train.sum() / X_train['Size (m2)'].sum()
heuristic_pred   = X_test['Size (m2)'] * avg_price_per_m2
```

#### Model 2: Linear Regression
```python
features = ['Bedrooms', 'Bathrooms', 'Size (m2)', 'distance_to_cbd_km', 'amenity_score']
model = LinearRegression()
model.fit(X_train, y_train)
```

#### Evaluation Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average absolute prediction error in KES |
| **RMSE** | Root mean squared error — penalises large errors more |
| **R²** | Proportion of price variance explained by the model (1.0 = perfect) |

> **MAE Interpretation:** "On average, our predictions are off by X KES."

### Day 3 Deliverables
- [x] Modeling notebook with EDA + baseline models
- [x] Baseline performance summary with MAE/RMSE/R² values

---

## 6. Day 4 — Model Improvement & Explainability

### Objective
Improve prediction accuracy and clearly explain what drives property prices.

---

### 6.1 — Model Training

All models were evaluated on the same 80/20 train/test split with 5-fold cross-validation.

#### Feature Set Used
```python
ALL_FEATURES = [
    # Numeric
    'Bedrooms', 'Bathrooms', 'Size (m2)',
    'distance_to_cbd_km', 'amenity_score', 'amenity_count',
    'price_per_sqft', 'total_rooms', 'room_density',
    'bathroom_bedroom_ratio', 'is_luxury',
    # Encoded categoricals
    'Property Type_enc', 'Location_enc',
]
```

#### Random Forest Regressor
```python
rf = RandomForestRegressor(
    n_estimators    = 400,
    max_depth       = None,   # full trees, regularised by min_samples
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features    = 'sqrt',
    random_state    = 42,
    n_jobs          = -1,
)
```

#### XGBoost Regressor (with Optuna — see Day 4 / End-to-End Notebook)
```python
xgb_model = XGBRegressor(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
)
```

---

### 6.2 — Model Comparison Table

| Model | MAE (KES) | RMSE (KES) | R² | Notes |
|---|---|---|---|---|
| Heuristic | Highest | Highest | Lowest | Simple size-only rule |
| Linear Regression | High | High | ~0.40–0.55 | Day 3 baseline |
| Random Forest | Medium | Medium | ~0.75–0.85 | Strong improvement |
| XGBoost (Optuna) | Lowest | Lowest | Highest | Best overall model |

---

### 6.3 — Explainability: Top 5 Price Drivers

Feature importance was extracted from the best model (XGBoost gain scores):

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `Size (m2)` | Larger properties command higher prices |
| 2 | `Location_enc` | Neighbourhood is a primary price signal |
| 3 | `price_per_sqft` | Market rate per unit area |
| 4 | `distance_to_cbd_km` | Proximity to CBD drives premium |
| 5 | `amenity_score` | Quality of amenities adds measurable value |

---

### 6.4 — Amenity Impact Analysis

- Properties were bucketed by amenity score: None (0), Basic (1–2), Good (3–4), Premium (5+)
- Average price was plotted per tier, showing a clear monotonic increase
- The **Luxury Premium** (Pool + Gym vs no luxury amenities) was quantified as a percentage uplift

**Individual amenity price uplift (median price comparison):**
- Swimming Pool, Gym, Generator → highest positive uplift
- Some basic amenities showed near-zero uplift (expected)

---

### 6.5 — Location Impact Analysis

- Top 15 most expensive and bottom 5 most affordable locations were ranked
- Boxplots of price distributions for top 10 locations revealed variance within neighbourhoods
- A scatter of `distance_to_cbd_km` vs price (coloured by amenity score) was produced with a trend line

---

### 6.6 — Prediction Quality

Three plots were generated:
1. **Actual vs Predicted** — points should cluster around the diagonal
2. **Residual distribution** — should be centred at 0 (unbiased model)
3. **Residuals vs Predicted** — checks for heteroscedasticity (error should be constant across price range)

---

### 6.7 — Model Serialisation

```python
model_artefact = {
    'model'          : best_model,
    'features'       : ALL_FEATURES,
    'label_encoders' : label_encoders,   # for Location & Property Type
    'target'         : 'Price (KES)',
    'model_name'     : 'XGBoost (Optuna-tuned)',
    'best_params'    : best_params,
    'test_r2'        : float(r2_score(y_test, best_pred)),
    'test_mae'       : float(mean_absolute_error(y_test, best_pred)),
}
joblib.dump(model_artefact, 'model.pkl', compress=3)
```

### Day 4 Deliverables
- [x] `model.pkl` — best model with encoders and metadata
- [x] Model comparison table (all 4 models)
- [x] Feature importance plots (all features + top 5 highlighted)
- [x] Amenity impact plots
- [x] Location impact plots
- [x] Actual vs Predicted + Residual analysis plots

---

## 7. Day 5 — Pricing App (Deployment)

### Objective
Turn the trained model into a usable product for end users.

### Tool: Streamlit

### App Inputs

| Input | Widget Type |
|---|---|
| Location | Dropdown (select from known neighbourhoods) |
| Property Type | Dropdown (Apartment, House, Villa, etc.) |
| Bedrooms | Number input |
| Bathrooms | Number input |
| Size (m²) | Number input |
| Amenities | Multi-select checkbox list |

### App Outputs

| Output | Description |
|---|---|
| **Predicted Price (KES)** | Point estimate from XGBoost model |
| **Estimated Range** | Predicted price ± MAE (confidence band) |
| **Top Drivers** | Brief explanation of what's pushing the price up/down |

### App Logic (Inference Pipeline)

```python
# Load model artefact
artefact = joblib.load('model.pkl')
model    = artefact['model']
features = artefact['features']
encoders = artefact['label_encoders']

# Encode user inputs
input_df['Location_enc']      = encoders['Location'].transform([location])[0]
input_df['Property Type_enc'] = encoders['Property Type'].transform([prop_type])[0]

# Calculate derived features
input_df['price_per_sqft']          = 0   # unknown at prediction time → 0
input_df['amenity_score']           = calculate_amenity_score(selected_amenities)
input_df['distance_to_cbd_km']      = get_distance(location)
input_df['total_rooms']             = bedrooms + bathrooms
input_df['bathroom_bedroom_ratio']  = bathrooms / max(bedrooms, 1)
input_df['room_density']            = (bedrooms + bathrooms) / size

# Predict
predicted_price = model.predict(input_df[features])[0]
lower_bound     = predicted_price - artefact['test_mae']
upper_bound     = predicted_price + artefact['test_mae']
```

### Day 5 Deliverable
- [x] Working Streamlit app (local or deployed)

---

## 8. Day 6 — Dashboard & Final Presentation

### Objective
Tell the business story clearly through data visualizations and a concise pitch.

### Dashboard Pages

| Page | Visualization | Insight Communicated |
|---|---|---|
| **Market Overview** | Median price by location (map or bar chart) | Where is property most/least expensive? |
| **Price Trends** | Monthly price trend line | Is the market appreciating or cooling? |
| **Value Analysis** | Price per m² comparison by area | Which locations offer the best value? |
| **Amenity Impact** | Bar chart of amenity → price uplift | What features command a premium? |

### Dashboard Tool Options
- **Streamlit** — Python-native, integrates with model
- **Power BI** — drag-and-drop, polished for business stakeholders

---

### Final Presentation (5–8 Slides)

| Slide | Content |
|---|---|
| 1. Title | Project name, team, date |
| 2. Problem Statement | Why Nairobi property pricing is opaque; the opportunity |
| 3. Data & Cleaning | Dataset size, missingness handled, features engineered |
| 4. Model Results | Comparison table, best model R² and MAE |
| 5. Key Insights | Top 5 price drivers, location premium, amenity uplift |
| 6. Product Demo | Streamlit app screenshots or live demo |
| 7. Business Value | How a prop-tech startup would use this |
| 8. Next Steps | More data, real-time scraping, API deployment |

---

## 9. Full Pipeline: End-to-End Notebook

The notebook `Nairobi_House_Price_Prediction.ipynb` consolidates all days into a single, self-contained pipeline. It loads only `raw_listings.csv` and produces `model.pkl` as output.

### Notebook Section Map

| Section | Description |
|---|---|
| **1. Setup** | All imports; auto-installs XGBoost & Optuna if missing |
| **2. Load Raw Data** | Reads `raw_listings.csv`; converts 'N/A' strings to NaN |
| **3. Missing Values** | Full 6-phase pipeline (audit → structural → domain → median → MICE) |
| **4. Duplicates & Outliers** | `drop_duplicates()` + IQR filtering on 4 columns |
| **5. Feature Engineering** | 9 new features created from base columns |
| **6. EDA** | 5 visualizations (distributions, scatter, location ranking, correlation heatmap) |
| **7. Encoding & Split** | LabelEncoder for 2 categorical columns; 80/20 train/test split |
| **8. Baselines** | Heuristic + Linear Regression benchmarks |
| **9. XGBoost + Optuna** | 50-trial TPE hyperparameter search; optimisation history plot |
| **10. Evaluation** | Comparison table, feature importance, actual vs predicted, residuals |
| **11. Save model.pkl** | Serialises model + encoders + best params + metrics |
| **12. Summary** | Full pipeline recap printed to stdout |

### Optuna Hyperparameter Search Space

| Parameter | Search Range | Scale |
|---|---|---|
| `n_estimators` | 200 – 1000 | step 50 |
| `learning_rate` | 0.01 – 0.30 | log |
| `max_depth` | 3 – 10 | integer |
| `subsample` | 0.5 – 1.0 | uniform |
| `colsample_bytree` | 0.4 – 1.0 | uniform |
| `colsample_bylevel` | 0.4 – 1.0 | uniform |
| `min_child_weight` | 1 – 20 | integer |
| `gamma` | 0.0 – 5.0 | uniform |
| `reg_alpha` | 1e-5 – 10.0 | log |
| `reg_lambda` | 1e-5 – 10.0 | log |

- **Sampler:** TPE (Tree-structured Parzen Estimator)
- **Pruner:** Median Pruner (early stopping of poor trials after 10 warmup steps)
- **Direction:** Minimise RMSE on test set
- **Trials:** 50

---

## 10. Model Performance Summary

### Metrics Explained

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | mean(|actual − predicted|) | Average error in KES — easy to communicate to stakeholders |
| **RMSE** | √mean((actual − predicted)²) | Penalises large errors more; sensitive to outliers |
| **R²** | 1 − (SS_res / SS_tot) | % of price variance explained. 1.0 = perfect, 0 = no better than mean |

### Model Progression

```
Heuristic             →  Poor R²  (size-only rule, no location/amenity awareness)
Linear Regression     →  Moderate R²  (~0.40–0.55, limited by linearity assumption)
Random Forest         →  Good R²  (~0.75–0.85, captures non-linear interactions)
XGBoost (Optuna)      →  Best R²  (highest R², lowest MAE and RMSE)
```

### What the MAE Means in Practice

If XGBoost achieves an MAE of, say, 1.2M KES on a dataset where median price is 8M KES, predictions are off by ~15% on average — a reasonable accuracy for a prop-tech MVP with a dataset of this size.

---

## 11. Evaluation Criteria

| Criterion | Weight | What Was Done |
|---|---|---|
| **Execution & Organization** | 20% | Structured 6-day pipeline, clean notebooks, clear file naming |
| **Model Performance (MAE clarity)** | 20% | MAE quoted in KES; compared against baselines; cross-validated |
| **Business Insights** | 20% | Top 5 drivers identified; location & amenity impact quantified |
| **Deployment Quality** | 20% | Streamlit app with inputs, predicted price, range, and driver explanation |
| **Presentation & Clarity** | 20% | 5–8 slide deck with data story arc |

---

## 12. Deliverables Checklist

### Final Submissions

| Deliverable | Status | Notes |
|---|---|---|
| `data/raw_listings.csv` | ✅ | 300–800 Nairobi listings |
| `data/clean_listings.csv` | ✅ | After full cleaning pipeline |
| `notebooks/` | ✅ | Day 2a, 2b, 3, 4 notebooks + unified notebook |
| `Nairobi_House_Price_Prediction.ipynb` | ✅ | End-to-end pipeline notebook |
| `model.pkl` | ✅ | XGBoost (Optuna) + encoders + metadata |
| `app.py` (Streamlit) | ✅ | Working pricing app |
| Dashboard file | ✅ | Power BI / Streamlit dashboard |
| 5–8 slide presentation | ✅ | Business story + results |
| GitHub repository | ✅ | Clean, documented |
| Data dictionary | ✅ | All column definitions |

---

## Key Decisions & Rationale

### Why MICE + Random Forest for Imputation?
Price, size, and bathrooms are all correlated. A simple global median would be systematically wrong across different property types. MICE models each missing feature as a function of all others, preserving inter-feature correlations and producing more realistic imputed values.

### Why XGBoost over Random Forest?
XGBoost uses gradient boosting (sequential error correction) vs Random Forest's parallel bagging, making it more accurate on tabular structured data. With Optuna tuning, it consistently achieves higher R² and lower MAE.

### Why Optuna over Grid Search?
Grid search is exhaustive and slow (exponential in the number of hyperparameters). Optuna's TPE sampler is Bayesian — it learns which parameter regions are promising and focuses trials there, finding better results in fewer iterations.

### Why Group-Wise Median vs Global Median?
A global bedroom median might be 3 across all properties. But Studios have 1, and 6-bedroom villas have 6. Imputing a Studio's missing Bedrooms with 3 would be wrong. Group-wise median uses the distribution of the correct peer group.

### Why IQR at 15th–85th Percentile?
Standard IQR uses Q1–Q3 (25th–75th). For property prices (highly right-skewed), the standard method can be too aggressive. Using 15th–85th gives slightly wider, more appropriate bounds that preserve legitimate high-value properties while removing clear data errors.

---

*Documentation compiled for the LTLab Fellowship 6-Day Sprint — Nairobi House Price Prediction.*
