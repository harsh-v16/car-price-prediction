## 🚗 Car Price Prediction using Machine Learning

📌 Project Overview

This project aims to predict the selling price of cars based on various features such as car brand, fuel type, engine specifications, and other technical details.
By applying feature engineering and machine learning models, the project demonstrates how data can be used to create accurate pricing predictions for the automotive market.

---------

## 🛠 Steps Involved
	1.	Data Exploration & Cleaning
	•	Dropped unnecessary ID columns
	•	Extracted car brand from the car name
	•	Corrected spelling errors in brand names for consistency
	2.	Feature Engineering
	•	Handled missing values
	•	Encoded categorical variables (fuel type, aspiration, etc.)
	•	Correlation analysis to find the most important features
	3.	Model Training & Evaluation
	•	Tried multiple regression models (Linear Regression, XGBoost)
	•	Evaluated models using R² Score, MAE, RMSE
	•	Selected the best-performing model
	4.	Final Prediction
	•	Trained the final model on the cleaned dataset
	•	Generated predictions for the given data

 ---

 ## 📂 Project Structure
 
├── carprice.py          # Source code with full comments

├── requirements.txt     # Required Python libraries

├── car_price_predictions.csv  # Final prediction output file

└── README.md            # Project Documentation

-----

## 📊 Technologies Used
	•	Python
	•	Pandas, NumPy (Data Analysis & Cleaning)
	•	Matplotlib, Seaborn (Data Visualization)
	•	Scikit-learn, XGBoost (Machine Learning Models)

-----------

## 🚀 Results
	•	Achieved good accuracy in predicting car prices.
	•	XGBoost Regressor gave the best performance among tested models.

---

## 👨‍💻 Author

 Harsh Chaudhary
 


