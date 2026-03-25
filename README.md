# 🏠 Real Estate Price Prediction (Machine Learning)

## 📌 Overview
This project builds a machine learning model to predict **house prices (MEDV)** using a housing dataset.

- 🎯 Target: `MEDV` (Median house value in $1000s)
- 📊 Project Type: Regression
- 📁 Dataset: Kaggle Housing Dataset (Boston Housing-style)

---

## 📊 Dataset Description

The dataset contains 14 features describing neighborhoods:

| Feature | Description |
|--------|------------|
| CRIM | Crime rate by town |
| ZN | Residential land zoning |
| INDUS | Non-retail business area |
| CHAS | River proximity (1 = yes, 0 = no) |
| NOX | Pollution level |
| RM | Average number of rooms |
| AGE | Old houses proportion |
| DIS | Distance to job centers |
| RAD | Highway accessibility |
| TAX | Property tax rate |
| PTRATIO | Student-teacher ratio |
| B | Population-related feature |
| LSTAT | Lower-income population (%) |
| **MEDV** | 🎯 Target variable |

---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## 🔄 Project Workflow

### 1. Data Exploration
- Checked structure using `.info()` and `.describe()`
- Identified missing values in `RM`
- Visualized distributions using histograms

### 2. Feature Engineering
Created a new feature:

```python
housing['TPR'] = housing['TAX'] / housing['RM']
    ```