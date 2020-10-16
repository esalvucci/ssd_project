import pandas as pd
from datetime import datetime
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas_datareader as web
import sys
import mlflow
import mlflow.sklearn
import logging

ticker = ['MSFT']
end = datetime.now()
start = datetime(end.year-1, end.month, end.day)

df1 = web.DataReader(ticker, 'yahoo', start, end)['Adj Close']
df1 = df1.reset_index()
df1.rename(columns={'MSFT': 'Adj Close'}, inplace=True)

df1.insert(0, 'id', range(0, len(df1)))
df1.sort_values(by='Date', inplace=True, ascending=True)

X = df1[['id']].values.reshape(-1, 1)
y = df1[['Adj Close']].values.reshape(-1, 1).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

kernel = sys.argv[1]
C = int(sys.argv[2])
gamma = sys.argv[3]
epsilon = float(sys.argv[4])


with mlflow.start_run():

    mlflow.log_param("kernel", kernel)
    mlflow.log_param("C", C)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("epsilon", epsilon)

    regressor = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    print(df)

    print('SVR performance evaluation')
    print(explained_variance_score(y_test, y_pred))
    mlflow.log_metric("explained variance score - should be near 1", explained_variance_score(y_test, y_pred))
    mlflow.sklearn.log_model(regressor, "svr regressor")
