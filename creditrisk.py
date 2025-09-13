# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# -------------------------
# 2. Load Data
# -------------------------
df = pd.read_csv("loan.csv")

# Strip column spaces (just in case)
df.columns = df.columns.str.strip()

# Features and target
feature_cols = ['customer_id', 'credit_lines_outstanding', 'loan_amt_outstanding',
                'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
X = df[feature_cols]
y = df["Defaulted"]

# -------------------------
# 3. Preprocess Data
# -------------------------
# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------
# 4. Train Random Forest Model
# -------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred_prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_prob)
print("ROC AUC Score:", round(auc, 3))

# Optional: classification report
y_pred_class = (y_pred_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred_class))

# -------------------------
# 5. Expected Loss Function
# -------------------------
def expected_loss(loan_features, recovery_rate=0.1):
    """
    Calculate Probability of Default and Expected Loss for a single loan.
    
    loan_features: dict with keys:
      'customer_id', 'credit_lines_outstanding', 'loan_amt_outstanding',
      'total_debt_outstanding', 'income', 'years_employed', 'fico_score', 'LoanAmount'
      
    recovery_rate: fraction of loan recovered in default
    """
    # Ensure all features exist
    input_df = pd.DataFrame([loan_features])
    
    # Fill missing columns with 0 if needed
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Scale features
    input_scaled = scaler.transform(input_df[feature_cols])
    
    # Predict Probability of Default
    pd_pred = model.predict_proba(input_scaled)[:,1][0]
    
    # Exposure at Default
    ead = loan_features.get("LoanAmount", 0)
    
    # Expected Loss
    el = pd_pred * (1 - recovery_rate) * ead
    
    return pd_pred, el

# -------------------------
# 6. Test Example
# -------------------------
sample_loan = {
    'customer_id': 101,
    'credit_lines_outstanding': 5,
    'loan_amt_outstanding': 10000,
    'total_debt_outstanding': 15000,
    'income': 60000,
    'years_employed': 5,
    'fico_score': 720,
    'LoanAmount': 10000
}

pd_estimate, expected_loss_value = expected_loss(sample_loan)
print(f"Estimated PD: {pd_estimate:.3f}")
print(f"Expected Loss: {expected_loss_value:.2f}")
