import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# loading data
train = pd.read_csv("trainst4.csv")
test = pd.read_csv("testst4.csv")

# adding labels and training
le = LabelEncoder()
y_train = le.fit_transform(train['color'])  # Converts 'red'/'blue' to 0/1
y_test = le.transform(test['color'])

X_train = train[['x', 'y']]
X_test = test[['x', 'y']]

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# final results and display
print("Accuracy:", accuracy_score(y_test, y_pred)*100, "%")
plt.figure(figsize=(6, 6))
plt.scatter(X_test['x'], X_test['y'], c=le.inverse_transform(y_pred), cmap='bwr', s=1)
plt.title('Predicted Colors using Logistic Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()



