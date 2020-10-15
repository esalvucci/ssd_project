#2
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pyEX as p

#carico il dataset
url = "https://raw.githubusercontent.com/okHotel/ssd_project/master/data_set/monthly-sunspots.csv"
df = pd.read_csv(url)
df = df.sort_values(by=['Month'])

#faccio grafico
plt.figure(figsize=(19,15))
plt.plot(df.Sunspots)
plt.ylabel('Number of Sunspots')
plt.xlabel('Record ID (Month)')
plt.title('Monthly-sunspots over 350 years')
plt.show()

#vairanza -> qunato ogni valore si discosta dalla sua media
#deviazione standard
#Set number of days and standard deviations to use for rolling lookback period for Bollinger band calculation
window = 20

#Calculate (entrambi mobili) rolling mean and standard deviation using number of days set above
rolling_mean = df.Sunspots.rolling(window, min_periods=1).mean()
rolling_std = df.Sunspots.rolling(window, min_periods=1).std()

#create two new DataFrame columns to hold values of upper and lower Bollinger bands
#media mobile e dev standard sono mobili
plt.figure(figsize=(19,15))
plt.plot(df.Sunspots, label='Dataset', color='blue')
plt.plot(rolling_mean, label='Rolling mean (w = 20)', color='orange') 
plt.plot(rolling_std, label='Rolling standard deviation (w = 20)', color='green')
plt.legend()
plt.show()

df['rolling_mean'] = rolling_mean
df['rolling_std'] = rolling_std
df[:10]

df1 = pd.DataFrame()
df1['Month'] = df.Month.copy()
df1['Sunspots'] = df.Sunspots.copy()

rolling_mean = np.concatenate(([0], df.rolling_mean.values.copy()))
rolling_mean = np.resize(rolling_mean, rolling_mean.size -1)
df1['Prevision'] = rolling_mean

rolling_std = np.nan_to_num(np.concatenate(([0], df.rolling_std.values.copy())))
rolling_std = np.resize(rolling_std, rolling_std.size -1)
df1['rolling_std'] = rolling_std

df1['MinRange'] = df1['Prevision'] - df1['rolling_std']
df1['MaxRange'] = df1['Prevision'] + df1['rolling_std']

predicate = (df1['Sunspots'] <= df1['MaxRange']) & (df1['Sunspots'] >= df1['MinRange'])
df1['Class'] = np.where(predicate, '1', '0')
df1[:50]

plt.figure(figsize=(19,15))
plt.plot(df1.Prevision, label='Prevision', color='orange')
plt.plot(df1.MinRange, label='MinRange', color='blue')
plt.plot(df1.MaxRange, label='MaxRange', color='blue')
plt.scatter(df1.Month, df1.Sunspots, label='x = Month, y = Sunspots', s = 10, color='black')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
X = df1[['Sunspots']].values.reshape(-1,1)
Y = df1[['Class']].values.reshape(-1,1).ravel()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.66,test_size=0.33)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', C=100, gamma=0.01)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

# nella matrice in pos[0,0] ho gli 0 giusti, 
# nella matrice in pos[1,1] ho gli 1 giusti
from sklearn.metrics import classification_report, confusion_matrix
print('\nConfusion matrix')
print(confusion_matrix(Y_test, Y_pred))
print('\nClassification report')
print(classification_report(Y_test, Y_pred))

#precision = true positive / true positive + false positive
#recall = true pos / true positive + false negative