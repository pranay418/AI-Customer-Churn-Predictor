# AI Customer Churn Predictor (SaaS/Startups)

An end-to-end Machine Learning web application designed to predict and analyze subscription churn for SaaS and subscriber-based businesses. Built with **Streamlit**, **scikit-learn**, and **Plotly**, this project identifies high-risk customers, provides personalized customer retention recommendations, and presents global dashboard insights into structural churn drivers.

---

## 🚀 Key Features

### 1. 🎯 Churn Risk Predictor (Local Explainer)
- **Interactive Profile Builder**: Build custom subscriber profiles via demographic inputs, contracted services, and billing configurations.
- **Probability Meter**: A color-coded gauge visualizes customer churn likelihood dynamically.
- **KPI Metrics Comparison**: Contrast the selected customer's profile metrics (Tenure, Monthly Charges, Total Charges) directly with the averages of churned and retained customers.
- **Retention Recommendations**: Automated alerts identify specific warning factors (e.g., month-to-month contracts, missing security features) and propose business recommendations.

### 2. 📊 Global Churn Analytics Dashboard (Why Users Churn)
- **Feature Importances**: Highlights top drivers of churn derived directly from the machine learning model.
- **Contract Impact Analysis**: Visualizes the volatility of Month-to-Month contracts.
- **Billing & Tenure Density Scatter**: Shows the concentration of churned users across different payment plans and subscription lifespans.
- **ISP Service Breakdown**: Illustrates churn rates relative to internet service types (DSL vs. Fiber Optic vs. No Internet).

---

## 📈 Model Performance
We trained a **Random Forest Classifier** with class weighting on the Telco Customer Churn dataset (7,043 samples).
- **ROC AUC**: `84.3%`
- **Recall (Sensitivity)**: `78.3%` (Successfully captures ~78% of all customers who will churn)
- **Overall Accuracy**: `76.1%`

---

## 🛠️ Tech Stack
- **Core App**: Python, Streamlit
- **Machine Learning**: Scikit-Learn, Joblib
- **Data Engineering**: Pandas, NumPy
- **Visualizations**: Plotly Express, Plotly Graph Objects

---

## ⚙️ Installation & Usage

To run the application locally:

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd customer-churn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Fetch the dataset
```bash
python download_data.py
```

### 4. Train the ML Model
```bash
python train.py
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

---

## 📂 Project Structure
- `download_data.py`: Fetches the Kaggle dataset programmatically.
- `train.py`: Preprocesses features, runs the ML pipeline, evaluates the classifier, and saves it.
- `app.py`: Streamlit frontend with Plotly visualizations.
- `model.joblib`: Serialized random forest model pipeline.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Configured to exclude data and bin files.
