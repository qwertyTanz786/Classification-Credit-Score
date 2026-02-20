import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
df = pd.read_csv(r"C:\Users\panch\Downloads\cleaned credit score.csv")
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.replace("_", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="ignore")
df = df.drop(["Type_of_Loan"], axis=1)
X = df.drop("Credit_Score", axis=1)
Y = df["Credit_Score"]
X = pd.get_dummies(X, columns=["Month", "Occupation", "Credit_Mix", "Payment_Behaviour"])
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
cv_rf=GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2,7],
    }
)
cv_rf.fit(X_train, Y_train)
model = cv_rf.best_estimator_
Y_pred = model.predict(X_test)

print(classification_report(Y_test, Y_pred))
