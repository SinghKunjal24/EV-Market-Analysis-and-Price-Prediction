# 🚗⚡ Electric Vehicle Business Analysis & Prediction

## 📖 Introduction
The electric vehicle (EV) industry is evolving rapidly, and **data-driven insights** can optimize business strategies. This project leverages **machine learning models** to predict **EV prices**, helping businesses set competitive pricing and identify key factors influencing EV adoption.

## 🛠 Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/yourusername/EV_Prediction.git
cd EV_Prediction
pip install -r requirements.txt
```

## 📊 Data Processing
- Handled **missing values** and **encoded categorical features**.
- Key features selected: **Model Year, Electric Range, Legislative District, Make/Model**.
- Target variable: **Base MSRP (EV Price)**.

## 🧠 Machine Learning Models
### **1️⃣ Random Forest Regressor 🌲**
- Hyperparameter tuning with **GridSearchCV**.
- Results:
  - **MAE:** 36.37
  - **RMSE:** 3990.06
  - **R2 Score:** 0.75

### **2️⃣ XGBoost Regressor 🚀**
- Optimized for better predictions.
- Results:
  - **MAE:** 107.84
  - **RMSE:** 4022.00
  - **R2 Score:** 0.74

## 🖥️ Training the Model
To train the **Random Forest** model, run:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Initialize model with best parameters
best_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
best_rf.fit(X_train, y_train)

# Predict
y_pred = best_rf.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest - MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}")
```

## 📈 Key Insights
- **Electric range is increasing** over model years due to advancements in battery technology.
- **Higher variability in recent years**, showing a mix of low and high-range EVs.
- **Outliers exist**, indicating luxury/high-performance models.
- **Random Forest outperformed XGBoost**, making it the preferred model.

## 🚀 Future Improvements
- Handle **outliers** to improve prediction accuracy.
- Add **market factors** like incentives and charging infrastructure.
- Test with **other models like LightGBM**.

## 🤝 Contributing
Feel free to contribute! 🚀  
1. Fork the repo 🍴  
2. Create a new branch 🔀  
3. Commit your changes 💾  
4. Submit a Pull Request ✅  

## 📜 License
This project is licensed under the **MIT License**.

---

⭐ **If you find this useful, please give it a star!** ⭐
