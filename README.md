# Car Fuel Efficiency Prediction using Decision Tree, Random Forest & XGBoost

This project predicts **car fuel efficiency (in MPG)** using various regression models:  
Decision Tree • Random Forest • XGBoost  

It demonstrates a complete **end-to-end machine learning pipeline** — from data preprocessing and model training to tuning, feature importance analysis, and evaluation.

---

## Table of Contents
1. Overview  
2. Dataset  
3. Setup  
4. Data Preprocessing  
5. Model Training  
6. Model Tuning  
7. Feature Importance  
8. XGBoost Experiments  
9. Results  
10. Conclusion  
11. Author  

---

## Overview
The goal is to build a regression model that predicts a car’s **fuel efficiency (MPG)** based on features such as weight, engine size, horsepower, and cylinders.  
The workflow includes:
- Data cleaning and vectorization  
- Model training and comparison  
- Hyperparameter tuning  
- Feature importance analysis  
- Learning rate experiments with XGBoost  

---

## Dataset
The dataset used is `car_fuel_efficiency.csv`, containing:
- Vehicle weight  
- Engine size  
- Cylinders  
- Horsepower  
- Model year  
- Other numerical and categorical attributes  

**Target variable:** `fuel_efficiency_mpg`  
which represents the car’s fuel efficiency in miles per gallon (MPG).  
Missing values were replaced with zeros for simplicity.

---

## Setup

1. Create environment → `conda create -n ml-zoomcamp python=3.13`  
Activate it → `conda activate ml-zoomcamp`

2. Install dependencies → `pip install numpy pandas scikit-learn matplotlib xgboost`

3. Launch Jupyter → `jupyter notebook`

---

## Data Preprocessing

- Split dataset → 60% train, 20% validation, 20% test  
- Convert to dictionary format using `DictVectorizer(sparse=True)`  
- Fill missing values with `df = df.fillna(0)`  
- Prepare matrices for XGBoost with `xgb.DMatrix(X_train, label=y_train)`

---

## Model Training

**Decision Tree Regressor:**  
`dt = DecisionTreeRegressor(max_depth=1)`  
`dt.fit(X_train, y_train)`  
`y_pred = dt.predict(X_val)`  
`rmse(y_val, y_pred)`  

Baseline model with relatively high RMSE.  

**Random Forest Regressor:**  
`rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)`  
`rf.fit(X_train, y_train)`  
`rmse(y_val, rf.predict(X_val))`  

Performs much better than Decision Tree.

---

## Model Tuning

**Tuning n_estimators:**  
Loop through values from 10 to 200:  
`for n in range(10, 201, 10): RandomForestRegressor(n_estimators=n)`  

RMSE decreases sharply and stabilizes after `n_estimators ≈ 80`.  

**Tuning max_depth:**  
Tested values `[10, 15, 20, 25]`  
Best configuration → `max_depth = 10`, `n_estimators = 80`

---

## Feature Importance

Final tuned model → `rf = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=1)`  
`rf.fit(X_train, y_train)`  

Top features ranked by importance:  
- `vehicle_weight`  
- `engine_size`  
- `horsepower`  
- `cylinders`  

Vehicle weight has the highest impact on predicting fuel efficiency.

---

## XGBoost Experiments

Train using → `dtrain = xgb.DMatrix(X_train, label=y_train)` and `dval = xgb.DMatrix(X_val, label=y_val)`  

Parameters → `eta=0.3, max_depth=6, min_child_weight=1, objective='reg:squarederror', nthread=8`  

Training → `xgb.train(params, dtrain, evals=[(dtrain,'train'),(dval,'val')], num_boost_round=100, verbose_eval=5)`  

**Learning rate comparison:**  
| eta | Description | Validation RMSE |
|------|--------------|----------------|
| 0.3 | Fast learning, slightly higher RMSE | Faster |
| 0.1 | Slower, smoother learning | Best RMSE |

`eta = 0.1` achieves the lowest RMSE on the validation dataset.

---

## Results

| Model | RMSE (Validation) | Remarks |
|--------|------------------|----------|
| Decision Tree | ~1.61 | Baseline |
| Random Forest | ~0.45 | Strong improvement |
| XGBoost (η=0.1) | ~0.43 | Best performance |

---

## Conclusion
- Random Forest and XGBoost outperform Decision Trees.  
- Optimal configuration:  
  - Random Forest → `n_estimators=80`, `max_depth=10`  
  - XGBoost → `eta=0.1`, `max_depth=6`  
- `vehicle_weight` is the most influential feature.  
- Lower learning rates (`eta=0.1`) improve stability and generalization.

---

## Project Structure
`data/car_fuel_efficiency.csv`  
`notebooks/fuel_efficiency.ipynb`  
`README.md` 

---

## Future Work
- Use `early_stopping_rounds` in XGBoost  
- Apply `GridSearchCV` for hyperparameter tuning  
- Try Gradient Boosting and LightGBM  
- Add visualizations for feature importance and learning curves  

---

## Author
**Purushothama D (Codevalhalla)**  
Machine Learning Engineer  
[GitHub: Codevalhalla](https://github.com/Codevalhalla)

