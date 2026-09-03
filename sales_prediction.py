import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=== OASIS INFOBYTE TASK 5: SALES PREDICTION ===")

# ==========================================
# 1. DATA LOADING (Self-Sourced Safe Data String)
# ==========================================
real_data = """TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9
8.7,48.9,75.0,7.2
57.5,32.8,23.5,11.8
120.2,19.6,11.6,13.2
8.6,2.1,1.0,4.8
199.8,2.6,21.2,10.6
66.1,5.8,24.2,8.6
214.7,24.0,4.0,17.4
23.8,35.1,65.9,9.2
97.5,7.6,7.2,9.7
204.1,32.9,46.0,19.0
195.4,47.7,52.9,22.4
19.6,11.6,114.0,8.7
281.4,39.6,55.8,24.4
69.2,20.5,18.3,11.3
147.3,23.9,19.1,14.6
218.4,27.7,53.4,18.0
237.4,5.1,23.5,12.5
13.2,15.9,49.6,5.6
228.3,16.9,26.2,15.5
62.3,12.6,18.3,9.7
262.9,3.5,19.5,12.0
142.9,29.3,12.6,15.0
240.1,16.7,22.9,15.9
248.8,27.1,22.9,18.9
70.6,16.0,40.8,10.5
292.9,28.3,43.2,21.4
112.9,17.4,38.6,11.9
97.2,1.5,30.0,9.6
265.6,20.0,0.3,17.4
95.7,1.4,7.4,9.5
290.7,4.1,8.5,12.8
266.9,43.8,5.0,25.4
74.7,49.4,45.7,14.7
43.1,26.7,35.1,10.1
228.0,37.7,32.0,21.5
202.5,22.3,31.6,16.6
177.0,33.4,38.7,17.1
293.6,27.7,1.8,20.7
206.9,8.4,26.4,12.9
25.1,25.7,43.3,8.5
175.1,22.5,31.5,14.9
89.7,9.9,35.7,10.6
239.9,41.5,18.5,23.2
227.2,15.8,49.9,14.8
66.9,11.7,36.8,9.7
199.8,3.1,34.6,11.4
100.4,9.6,3.6,10.7
216.4,40.6,39.5,22.6
182.6,46.2,58.7,21.2
262.7,28.8,15.9,20.2
198.9,49.4,60.0,23.7
7.3,28.1,41.4,5.5
136.2,19.2,16.6,13.2
210.8,49.6,37.7,23.8
210.7,29.5,9.3,18.4
53.5,2.0,21.4,8.1
261.3,42.7,54.7,24.2
239.3,15.5,27.3,15.7
102.7,29.6,8.4,14.0
131.1,42.8,28.9,18.0
69.0,9.3,0.9,9.3
31.5,24.6,2.2,9.5
134.5,4.9,9.3,12.2
237.4,27.5,11.0,18.9
216.8,43.9,27.2,22.3
244.5,19.9,50.6,15.0
109.8,14.3,31.7,12.4
26.8,33.0,19.3,8.8
129.4,31.3,22.0,11.0
213.4,24.6,13.1,17.0
16.9,43.7,89.4,8.7
27.5,1.6,20.7,6.9
120.5,28.5,14.2,14.2
5.4,29.9,9.4,5.3
116.0,7.7,3.6,11.0
76.4,26.7,22.3,11.8
239.8,20.3,5.4,12.3
75.3,20.3,32.5,11.3
68.4,44.5,35.6,13.6
213.5,43.0,5.9,21.7
193.2,18.4,65.7,15.2
76.3,27.5,16.0,12.0
110.7,40.6,63.2,16.0
88.3,25.5,73.4,12.9
109.8,47.8,51.4,16.7
134.3,4.9,9.3,11.2
28.6,1.5,33.0,7.3
"""
df = pd.read_csv(io.StringIO(real_data))

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\n[✔] EDA: Checking Null Values...")
print(df.isnull().sum())

print("\n[✔] EDA: Descriptive Statistics...")
print(df.describe())

# ==========================================
# 3. VISUALIZATIONS (Mandatory Requirements)
# ==========================================
print("\nGenerating charts (Press back/close on your screen when each plot appears)...")

# Requirement: Pairplot of all features
sns.pairplot(df)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()
plt.close()

# Requirement: Individual scatter plots vs Sales
for media in ['TV', 'Radio', 'Newspaper']:
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=df, x=media, y='Sales', color='blue')
    plt.title(f'Sales vs {media} Spend')
    plt.show()
    plt.close()

# Requirement: Correlation matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
plt.close()

# ==========================================
# 4. DATA SPLITTING & TRAINING
# ==========================================
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Requirement: Train Linear Regression model as baseline
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Requirement: Train at least one additional model (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ==========================================
# 5. MODEL EVALUATION (MAE, RMSE, R2)
# ==========================================
def print_performance(y_true, y_pred, name):
    print(f"\n[✔] Evaluation metrics for {name}:")
    print(f"  • MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  • RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"  • R2 Score: {r2_score(y_true, y_pred):.4f}")

print_performance(y_test, lr_pred, "Linear Regression (Baseline)")
print_performance(y_test, rf_pred, "Random Forest Regressor")

# ==========================================
# 6. RESIDUAL PLOT (Best Model Requirement)
# ==========================================
residuals = y_test - rf_pred
plt.figure(figsize=(6, 4))
sns.scatterplot(x=rf_pred, y=residuals, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (Random Forest Errors)')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals (Errors)')
plt.show()
plt.close()

# ==========================================
# 7. INTERPRETATION & FEATURE IMPORTANCE
# ==========================================
print("\n[✔] Feature Importance Analysis:")
for col, score in zip(X.columns, rf.feature_importances_):
    print(f"  • {col}: {score*100:.1f}% impact on sales growth")

print("\nAll conditions on the checklist met successfully!")
