import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score

url = 'https://raw.githubusercontent.com/okHotel/ssd_project/master/data_set/daily-min-temperatures.csv'
df = pd.read_csv(url, names=['Date', 'Temp'], skiprows=2556) #carico solo i dati degli ultimi 3 anni
df = df.sort_values(by=['Date']) #ordino i record in base alla data
df.insert(0, 'id', range(0,len(df)))

#print(df.head())
plt.plot(df.Temp)
plt.title('Temperature trend over 3 years')
plt.ylabel('Temperature')
plt.show()

#  SVR prevision
# Spilt dataset into train and test
X = df[['id']].values.reshape(-1,1)
Y = df[['Temp']].values.reshape(-1,1).ravel()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = SVR(kernel='rbf', C=1, gamma=0.001, epsilon=.01) # regressor definition
regressor.fit(X_train,Y_train) # regressor training
y_pred = regressor.predict(X_test) # regressor prediction

plt.title('Support Vector Regression Model')
plt.scatter(X, Y, color='magenta', label='data')
plt.plot(X, regressor.predict(X), color='green', label='regression fun')
plt.ylabel('Temperature')
plt.legend()
plt.show()

df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

plt.title('Test results') 
plt.scatter(X_test.flatten(), Y_test.flatten(), color='orange', label='test data')
plt.scatter(X_test.flatten(), y_pred.flatten(), color='royalblue', label='predicted data')
plt.legend()
plt.show()

# Evaluation
# 1 - ( Var(y' - y)/Var(y) )
print("SVR performance:", explained_variance_score(Y_test, y_pred))


