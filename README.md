# Diabetes-Risk-Prediction-BRFSS-2015
 
A binary classification project that predicts whether a person has **diabetes or prediabetes** from 21 self-reported health, lifestyle, and socioeconomic indicators. The notebook covers data cleaning, exploratory analysis, four classification models built for an imbalanced target, and a feature-importance study showing that **5 features capture almost all of the predictive signal**.
 
 
---
 
## 📑 Table of Contents
 
1. [Overview](#-overview)
2. [Dataset](#-dataset)
3. [Project Workflow](#-project-workflow)
4. [Key EDA Findings](#-key-eda-findings)
5. [Modelling Approach](#-modelling-approach)
6. [Results](#-results)
7. [Feature Importance](#-feature-importance)

---
 
## 🎯 Overview
 
| | |
| :--- | :--- |
| **Problem** | Binary classification: diabetes / prediabetes (1) vs. no diabetes (0) |
| **Data** | CDC BRFSS 2015 survey, 253,680 responses, 21 features |
| **Challenge** | Class imbalance (about 85% vs. 15%), so plain accuracy is misleading |
| **Models** | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| **Main finding** | `GenHlth`, `BMI`, `Age`, `HighBP`, and `HighChol` give a ROC-AUC of 0.81, against 0.82 for all 21 features |
 

```
```
 
## 🗂️ Dataset
 
The data comes from the CDC's **Behavioral Risk Factor Surveillance System (BRFSS) 2015**, in its binary "Diabetes Health Indicators" form (`diabetes_binary_health_indicators_BRFSS2015.csv`).
 
- **Rows:** 253,680 survey responses (229,474 after removing 24,206 duplicates)
- **Target:** `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes)
- **Missing values:** none
| Group | Features |
| :--- | :--- |
| **Medical history** | `HighBP`, `HighChol`, `CholCheck`, `Stroke`, `HeartDiseaseorAttack` |
| **Body & health status** | `BMI`, `GenHlth` (1–5), `MentHlth` (0–30 days), `PhysHlth` (0–30 days), `DiffWalk` |
| **Lifestyle** | `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump` |
| **Healthcare access** | `AnyHealthcare`, `NoDocbcCost` |
| **Demographics** | `Sex`, `Age` (14 age bands), `Education` (1–6), `Income` (1–8) |
 
```
```

## 🔄 Project Workflow
 
1. **Preprocessing:** All columns are cast to integers. No values are missing. 24,206 duplicate rows are dropped, leaving a class split of about **84.7% / 15.3%**.
2. **EDA:** A separate copy of the data (`df_eda`) maps codes to readable labels and ordered categories for plotting. Modelling uses the original numeric data.
3. **Split:** 80% train, 20% test (45,895 rows), stratified on the target so both sets keep the same class ratio.
4. **Modelling:** Each model is handled for imbalance with class weights or `scale_pos_weight`.
5. **Feature selection:** Permutation importance is computed on validation data, then models are refit on the top-k features.
---
```
```
## 🔍 Key EDA Findings
 
**Strongest separators (diabetes group vs. non-diabetes group)**
 
| Feature | Diabetes group | Non-diabetes group |
| :--- | :---: | :---: |
| High blood pressure | ~75% | ~40% |
| High cholesterol | ~67% | ~40% |
| Difficulty walking | ~37% | ~15% |
| Heart disease / attack | ~22% | ~8% |
| Stroke | ~9% | ~4% |
```
```
 
**Other patterns**
- **General health:** "Fair" and "Poor" self-ratings are far more common with diabetes (about 28% vs. 11% for Fair, and about 13% vs. 4% for Poor).
- **Age:** Cases concentrate from age 60 onwards, peaking in the 65–69 band.
- **Income and education:** There is a clear gradient. Lower income and education go with higher diabetes rates.
- **Weak features:** `CholCheck`, `AnyHealthcare`, and `NoDocbcCost` look almost identical across both classes.
- **Counterintuitive result:** Heavy drinking is *less* common in the diabetes group. This likely reflects medical advice and self-reporting effects, so it should not be read as alcohol being protective.
**Spearman correlation with the target**
- Strongest positive: `GenHlth` (0.27), `HighBP` (0.25), `BMI` (0.22), `DiffWalk` (0.21), `HighChol` (0.19), `Age` (0.18).
- Strongest negative: `Income` (−0.14), `PhysActivity` (−0.10), `Education` (−0.10).
- No single feature exceeds about 0.27, so prediction depends on combinations of features.
- No pair of features is correlated above about 0.45, so multicollinearity is not a concern.
---
```
```
 
## 🤖 Modelling Approach
 
| Model | Key settings | Imbalance handling |
| :--- | :--- | :--- |
| Logistic Regression | `max_iter=1500` | `class_weight='balanced'` |
| Decision Tree | `max_depth=12` | `class_weight='balanced'` |
| Random Forest | `n_estimators=200`, `max_depth=12` | `class_weight='balanced_subsample'` |
| XGBoost | `n_estimators=300`, `max_depth=4`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='aucpr'` | `scale_pos_weight` = (negatives / positives) |
 
Because the positive class is only about 15% of the data, the models are tuned to **catch diabetes cases (recall)**, which lowers accuracy and precision. The notebook therefore reports precision, recall, F1, ROC-AUC, and PR-AUC rather than accuracy alone.
 
---
```
```
 
## 📊 Results
 
Test set: 45,895 rows (38,876 negative, 7,019 positive). Metrics for the positive class (diabetes / prediabetes) at a 0.5 threshold:
 
| Model | Train Acc. | Test Acc. | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7174 | 0.7142 | 0.32 | 0.76 | 0.45 |
| Decision Tree | 0.7243 | 0.6972 | 0.30 | 0.75 | 0.43 |
| Random Forest | 0.7525 | 0.7339 | 0.33 | 0.74 | 0.46 |
| XGBoost | 0.7120 | 0.7077 | 0.32 | **0.79** | 0.45 |
 
- All four models perform similarly, with an F1 around 0.45 and recall between 0.74 and 0.79.
- **XGBoost has the highest recall (0.79)**, which suits a screening use case where missing a case costs more than a false alarm.
- **Random Forest has the best accuracy and F1**, with a small train–test gap.
- Precision is low (about 0.30–0.33) for every model. This is expected when the positive class is a small minority and recall is prioritised.
---
```
``` 
## ⭐ Feature Importance
 
Permutation importance (drop in ROC-AUC when a feature is shuffled) with XGBoost:
 
| Rank | Feature | Drop in ROC-AUC |
| :---: | :--- | :---: |
| 1 | `GenHlth` | 0.0492 |
| 2 | `BMI` | 0.0402 |
| 3 | `Age` | 0.0367 |
| 4 | `HighBP` | 0.0173 |
| 5 | `HighChol` | 0.0126 |
| 6–10 | `CholCheck`, `HvyAlcoholConsump`, `Sex`, `Income`, `MentHlth` | each below 0.01 |
 
Importance drops sharply after the top 5.
```
``` 
### How many features are really needed?
 
| Features used | ROC-AUC | PR-AUC |
| :---: | :---: | :---: |
| 3 (`GenHlth`, `BMI`, `Age`) | 0.792 | 0.394 |
| **5 (+ `HighBP`, `HighChol`)** | **0.809** | **0.426** |
| 8 | 0.815 | 0.434 |
| 10 | 0.816 | 0.437 |
| 15 | 0.819 | 0.443 |
| 21 (all) | 0.819 | 0.445 |
```
 ```
### Final comparison: top 5 vs. all 21 features (XGBoost)
 
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Top 5 features | 0.81 | 0.430 | 0.307 | 0.781 | 0.441 |
| All 21 features | 0.82 | 0.448 | 0.317 | 0.791 | 0.453 |
```
 

