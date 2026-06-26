# Housing Market Prediction

## Project Overview

This project predicts housing prices using machine learning regression models. The objective is to compare the performance of different regression algorithms and determine which model provides the most accurate predictions.

The project includes the following machine learning models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The models are trained on a housing dataset, evaluated using regression metrics, and compared based on their performance.

---

# Dataset

**Dataset:** `Housing.csv`

The dataset contains various features that influence the selling price of a house.

### Target Variable

- `price`

### Input Features

All remaining columns are used as independent variables (features).

Categorical variables are converted into numerical values using One-Hot Encoding.

```python
housing = pd.read_csv("Housing.csv")
housing = pd.get_dummies(housing, drop_first=True)
```



---

# Data Preprocessing

The preprocessing steps include:

1. Load the dataset.
2. Convert categorical variables into numerical variables using One-Hot Encoding.
3. Separate the input features and target variable.
4. Split the dataset into training and testing sets.

```python
x = housing.drop('price', axis=1)
y = housing['price']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Machine Learning Models

1. Linear Regression


2. Random Forest Regressor


3. Gradient Boosting Regressor

---


# Performance Results

| Model | Training R² | Test R² |
|-------|-------------|----------|
| Linear Regression | 0.6859 | 0.6529 |
| Random Forest Regressor | 0.5404 | 0.4168 |
| Gradient Boosting Regressor | 0.8625 | 0.6660 |

---

# Model Comparison

### Linear Regression

- Simple and fast.
- Performs well for linear relationships.
- Good baseline model.

### Random Forest Regressor

- Handles nonlinear relationships.
- Reduces overfitting compared to a single decision tree.
- The shallow tree depth (`max_depth=2`) resulted in lower prediction accuracy.

### Gradient Boosting Regressor

- Builds trees sequentially.
- Corrects previous prediction errors.
- Achieved the best prediction accuracy among all three models.

---

# Data Visualization

Scatter plots are generated to compare the actual housing prices with the predicted housing prices.

Example:

```python
plt.figure(figsize=(5,5))

plt.scatter(
    x=y_train,
    y=y_gbr_train_pred,
    alpha=0.3
)

plt.xlabel("Actual Housing Price")
plt.ylabel("Predicted Housing Price")

plt.show()
```

A regression line is also added to visualize how closely the predictions match the actual values.

---

# Project Workflow

1. Load Housing.csv dataset.
2. Perform One-Hot Encoding.
3. Separate features and target.
4. Split data into training and testing sets.
5. Train Linear Regression model.
6. Train Random Forest Regressor.
7. Train Gradient Boosting Regressor.
8. Predict housing prices.
9. Evaluate models using MSE and R² Score.
10. Compare model performances.
11. Visualize predictions.

---

# Conclusion

Among the three machine learning algorithms tested:

- **Gradient Boosting Regressor** achieved the highest prediction accuracy with a **Test R² Score of approximately 0.6660**, making it the best-performing model for this dataset.
- **Linear Regression** also performed well and provided a strong baseline.
- **Random Forest Regressor**, with `max_depth=2`, underperformed compared to the other models due to its shallow tree depth.

Overall, Gradient Boosting Regressor was the most effective model for predicting housing prices in this project.


---
