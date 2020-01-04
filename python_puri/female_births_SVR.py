import pandas as pd
import plotly.express as px
from datetime import datetime 
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score

url = 'https://raw.githubusercontent.com/okHotel/ssd_project/master/data_set/female_birth.csv'
df1 = pd.read_csv(url)
df1 = df1[['Date', 'Births']]
df1.insert(0, 'id', range(0,len(df1)))
df1.sort_values(by='Date', inplace=True, ascending=True)

plt.plot(df1.Date, df1.Births)
plt.xlabel('Date')
plt.ylabel('Births')
plt.title('Number of daily female births in California in 1959 and 1960')
plt.show()

X = df1[['id']].values.reshape(-1,1)
y = df1[['Births']].values.reshape(-1,1).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = SVR(kernel='rbf', C=100, gamma='scale', epsilon=.1)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

plt.scatter(X, y, color = 'magenta', label='data')
plt.plot(X, regressor.predict(X), color = 'green', label='regression function')
plt.title('Support Vector Regression Model')
plt.xlabel('id')
plt.ylabel('Births')
plt.legend()
plt.show()

plt.scatter(X_test.flatten(), y_pred.flatten(), color = 'orange', label='predicted data')
plt.scatter(X_test.flatten(), y_test.flatten(), color = 'royalblue', label='test data')
plt.title('Comparison between predicted values and test values')
plt.xlabel('id')
plt.ylabel('Births')
plt.legend() 
plt.show()

print('SVR performance evaluation')
print(explained_variance_score(y_test, y_pred))