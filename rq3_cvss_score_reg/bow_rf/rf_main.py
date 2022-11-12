from email import message
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

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
y_train = train_data["Score"].tolist()
y_test = test_data["Score"].tolist()
# apply BoW feature extraction
vectorizer = TfidfVectorizer(norm='l2', max_features=1000)
vectorizer = vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train).todense()
X_test = vectorizer.transform(X_test).todense()
# train the model
rf = RandomForestRegressor(n_estimators=1000,
                            n_jobs=10,
                            verbose=1)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"MSE: {mse}")
print(f"MAE: {mae}")

with open('./saved_models/best_acc_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("done")
