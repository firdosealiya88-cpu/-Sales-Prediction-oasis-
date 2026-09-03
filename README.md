# -Sales-Prediction-oasis-
# Oasis Infobyte Data Science Internship
## Task 5: Sales Prediction Using Python

### 📌 Project Overview
This project builds a predictive machine learning model to forecast product sales based on advertising expenditures across three main budget channels: TV, Radio, and Newspaper. 

### 📊 Model Performance & Evaluation
We trained and evaluated two separate algorithms on the dataset to find the most accurate predictor:

1. **Linear Regression (Baseline)**
   - Mean Absolute Error (MAE): 1.2379
   - Root Mean Squared Error (RMSE): 1.6068
   - R-Squared Accuracy Score: 0.8955 (89.55%)

2. **Random Forest Regressor (Selected Model)**
   - Mean Absolute Error (MAE): 0.8014
   - Root Mean Squared Error (RMSE): 0.9910
   - R-Squared Accuracy Score: 0.9603 (96.03%)

**Conclusion:** The Random Forest Regressor outperformed the Linear Regression model, capturing 96.03% of the variance in sales.

### 💡 Business Insights & Data Analysis
By analyzing feature importances from our best-performing model, we discovered how each marketing channel impacts product sales:
- **TV Advertising:** 68.6% impact (The highest driving factor for sales growth).
- **Radio Advertising:** 29.3% impact (A strong secondary supporting channel).
- **Newspaper Advertising:** 2.2% impact (Negligible effect on overall sales variance).

**Strategic Recommendation:** To optimize return on investment (ROI), the company should prioritize its marketing budget heavily toward TV campaigns, followed by Radio, while scaling back spend on Newspaper prints.
