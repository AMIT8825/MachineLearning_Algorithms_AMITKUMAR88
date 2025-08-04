# K-Nearest Neighbors on Digits Dataset
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predict & report
pred = model.predict(X_test)
print(classification_report(y_test, pred))
