import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# loading data 
train = pd.read_csv("trainst3.csv")
test = pd.read_csv("testst3.csv")

# data cleaning 
def parse_labels(labels):
    return labels.apply(lambda x: x.split(','))

y_train = parse_labels(train['color'])
y_test = parse_labels(test['color'])

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(y_train)
Y_test = mlb.transform(y_test)

# conditions and training
X_train = train[['x', 'y']]
X_test = test[['x', 'y']]

logreg = LogisticRegression(max_iter=1000, random_state=42)
multi_model = MultiOutputClassifier(logreg)
multi_model.fit(X_train, Y_train)

Y_pred = multi_model.predict(X_test)

# final result 
print("Accuracy", (Y_pred == Y_test).mean()*100, "%")


