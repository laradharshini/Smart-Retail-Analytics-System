# Smart-Retail-Analytics-System
A Flask-based machine learning web application that helps retail businesses make data-driven decisions using predictive analytics.

## Features

The system provides **five** major analytics tools:

### 1. **Stock-Out Prediction**

Predict whether a customer is likely to face product stock-outs using their shopping behavior and preferences.

### 2. **Sales Forecast**

Predict expected monthly grocery spend based on demographic and behavioral factors.

### 3. **Offer Recommendation**

Recommend the most effective offer types (e.g., discounts, cashback, loyalty points) using a multi-label classifier.

### 4. **Customer Segmentation**

Group customers into meaningful clusters using a trained KMeans model and display user-friendly segment names & traits.

### 5. **Category Optimization**

Provide insights into which product categories the user should focus on based on preferences and offer tendencies.

## Tech Stack

* **Backend:** Flask (Python)
* **Database:** MongoDB (User authentication)
* **ML Models:** Scikit-Learn (Pickled pipelines)
* **Frontend:** HTML, CSS, JS (Jinja templates)

## Required Model Files (`models/`)

Place these inside the *models* folder:

* `stockout_model.pkl`
* `sales_rf_model.pkl`
* `offer_rf_model.pkl`
* `offer_mlb.pkl`
* `segmentation_kmeans_model.pkl`

(*Ignore trend/retention models – not used.*)


## User Authentication

* Signup, Login, Logout
* User data stored in MongoDB (`smartretail.users`)
* Sessions handled using Flask `session`

## Running the App

```bash
pip install -r requirements.txt
python app.py
```

App runs at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

## Folder Structure 

```
smart-retail/
├─ app.py
├─ models/
│ ├─ stockout_model.pkl
│ ├─ sales_rf_model.pkl
│ ├─ offer_rf_model.pkl
│ ├─ offer_mlb.pkl
│ ├─ segmentation_kmeans_model.pkl
│ ├─ category_optimizer.py # contains train_category_model, generate_category_insights
│ └─ ...
├─ templates/
│ ├─ home.html
│ ├─ signup.html
│ ├─ login.html
│ ├─ dashboard.html
│ ├─ predict_stockout.html
│ ├─ sales_dashboard.html
│ ├─ segmentation.html
│ ├─ recommendation.html
│ ├─ category.html
│ ├─ category_result.html
│ └─ ...
├─ static/
│ └─ (images, css, js)
├─ Customer Experience Dataset.csv # optional used to calculate average spend
├─ requirements.txt
```

## Prediction Endpoints (JSON)

* `/predict_stockout`
* `/predict_sales`
* `/predict_recommendation`
* `/predict-segment`
* `/category` (form-based)

## Notes

* Passwords are stored as plain text in this version → **use hashing in production**.
* Missing models will disable specific tools but the app will still run.
Here is a clean, concise **To-Do section** you can directly paste into your README:

## To Do

* Implement password hashing (bcrypt/argon2) instead of storing plaintext passwords.
* Improve preprocessing pipeline consistency across models.
* Add more explainability/insight charts for predictions.
* Add unit tests for prediction endpoints.
* Add integration tests for signup, login, and session flow.
