import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np


with open("./cwe_label_map.pkl", "rb") as f:
    cwe_label_map = pickle.load(f)

# load train, val data
train = pd.read_csv('../../data/train.csv')
val = pd.read_csv('../../data/val.csv')
# use train + val data to fit the model
train_data = pd.concat([train, val])
# load test data
test_data = pd.read_csv('../../data/test.csv')
# textual code data
X_train = train_data["func_before"]
X_test = test_data["func_before"]
# labels
y_train = train_data["CWE ID"].tolist()
for i in range(len(y_train)):
    # transform to one-hot labels
    y_train[i] = cwe_label_map[y_train[i]][1]

y_test = test_data["CWE ID"].tolist()
for i in range(len(y_test)):
    # transform to one-hot labels
    y_test[i] = cwe_label_map[y_test[i]][1]

# apply BoW feature extraction
vectorizer = TfidfVectorizer(norm='l2', max_features=1000)
vectorizer = vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train).todense()
X_test = vectorizer.transform(X_test).todense()
# train the model
rf = RandomForestClassifier(n_estimators=1000,
                            n_jobs=10,
                            verbose=1)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc}")

with open('./saved_models/best_acc_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("done")
