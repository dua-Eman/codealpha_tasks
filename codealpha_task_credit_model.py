import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Sample Dataset (you can replace this with real financial data)
data = {
    'age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 28],
    'income': [50000, 100000, 75000, 120000, 40000, 90000, 150000, 110000, 72000, 55000],
    'loan_amount': [10000, 20000, 15000, 30000, 5000, 25000, 40000, 27000, 13000, 9000],
    'credit_score': [700, 800, 680, 790, 620, 750, 810, 770, 690, 710],
    'defaulted': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]  # 0 = Good, 1 = Defaulted
}

df = pd.DataFrame(data)

# 2. Feature & Target
X = df[['age', 'income', 'loan_amount', 'credit_score']]
y = df['defaulted']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Plot Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
