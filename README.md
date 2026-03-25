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

### 3. Data Splitting

The dataset was split into:

- 80% training data  
- 20% testing data  

---

### 4. Correlation Analysis

Key findings:

- 📈 `RM` has strong positive correlation with price  
- 📉 `LSTAT` has strong negative correlation  
- 📉 `TAX` and `INDUS` also negatively affect price  

---

### 5. Data Preprocessing

#### Missing Values

Handled using **median imputation**:

- Keeps dataset size intact  
- Works well for skewed data  

#### Feature Scaling

Used **StandardScaler**:

- Centers data around mean = 0  
- Improves model performance  

---

### 6. Pipeline

A pipeline was used to combine preprocessing and model training:

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

## 🤖 Models Used

- Linear Regression  
- Decision Tree  
- ✅ Random Forest (final model)  

---

## 📏 Model Performance

| Metric | Value |
|--------|------|
| Train RMSE | ~1.40 |
| CV RMSE (mean) | ~3.67 |
| Test RMSE | ~5.14 |

📌 The difference between training and testing error suggests slight **overfitting**.

---

## 📉 Insights

- More rooms → Higher house price  
- Higher lower-income population → Lower price  
- Feature engineering improved predictions  
- Random Forest performed best overall  

---

## 💾 Model Export

The trained model was saved using:

```python
dump(pipeline, "Real_Estate_Data.joblib")

## 📂 Project Structure


Real-Estate-ML/
│── data.csv
│── notebook.ipynb
│── Real_Estate_Data.joblib
│── README.md´

---

## ⚠️ Limitations

- Small dataset size  
- Possible bias in some features  
- Overfitting not fully eliminated  

---

## 🚀 Future Improvements

- Hyperparameter tuning  
- Try advanced models (XGBoost)  
- Deploy as a web app (Streamlit)  
- Use larger, real-world datasets  

---

## 🙌 Conclusion

This project demonstrates a complete **end-to-end machine learning workflow**, including:

- Data cleaning  
- Feature engineering  
- Model training  
- Evaluation  

A solid foundation for building real-world ML projects 🚀  

---

## 👤 Author

**Roshan Karki**