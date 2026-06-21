import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score
import joblib
import os

def load_and_preprocess_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Drop customerID as it is not predictive
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # TotalCharges is read as object because of some spaces. Let's fix it.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    
    # Fill missing values in TotalCharges with 0 (since tenure is 0 for these records)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert SeniorCitizen to string category (Yes/No) for uniform encoding
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Map target Churn to binary 0/1
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    return df

def train_model():
    file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run download_data.py first.")
        return
    
    df = load_and_preprocess_data(file_path)
    
    # Define features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Identify numerical and categorical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in X.columns if col not in num_cols]
    
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols}")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Define full training pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=4, class_weight='balanced', random_state=42))
        ]
    )
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Extract feature importances
    preprocessor_obj = pipeline.named_steps['preprocessor']
    cat_encoder = preprocessor_obj.named_transformers_['cat']
    
    # Get feature names after one-hot encoding
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + encoded_cat_cols
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance_df.head(10))
    
    # Save the pipeline and feature list
    model_data = {
        'pipeline': pipeline,
        'feature_names': feature_names,
        'categorical_columns': cat_cols,
        'numerical_columns': num_cols,
        'feature_importance': feature_importance_df.to_dict(orient='records'),
        'categorical_options': {col: df[col].unique().tolist() for col in cat_cols}
    }
    
    model_file = "model.joblib"
    print(f"\nSaving model data to {model_file}...")
    joblib.dump(model_data, model_file)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
