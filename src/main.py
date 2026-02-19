import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/heart_cleveland_upload.csv")

# -----------------------------
# Separate Features and Target
# -----------------------------
X = df.drop("condition", axis=1)
y = df["condition"]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train/Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Logistic Regression (Baseline)
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_train)

# -----------------------------
# Random Forest (Ensemble Model)
# -----------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)

# -----------------------------
# Model Evaluation (Random Forest)
# -----------------------------
y_pred_rf = rf_model.predict(x_test)

print("Random Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
