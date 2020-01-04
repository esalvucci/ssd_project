import pandas as pd
import plotly.express as px
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/okHotel/ssd_project/master/data_set/female_birth.csv'
df1 = pd.read_csv(url)
df1 = df1[['Date', 'Births']]

plt.plot(df1.Date, df1.Births)
plt.xlabel('Date')
plt.ylabel('Births')
plt.title('Number of daily female births in California in 1959 and 1960')
plt.show()

rmean = df1.Births.rolling(window=20, min_periods=1).mean() #calcolo la media mobile
rstd = df1.Births.rolling(window=20, min_periods=1).std() #calcolo la deviazione standard mobile

plt.plot(df1.Births, color='blue', label='Dataset')
plt.plot(rmean, color='orange', label='Rolling Mean (20)')
plt.plot(rstd, color='green', label='Rolling Standard Deviation (20)')
plt.legend(loc='upper left')
plt.show()

df1['rmean'] = rmean
df1['rstd'] = rstd
df1[:10]

means =np.concatenate(([0], df1.rmean.values.copy()))
means = np.resize(means, means.size - 1)
df1['Prev'] = means #ogni dato è la media mobile calcolata sul dato precedente

# Create the standard deviations array and delete the last value (to predict) The rolling mean values are considered our predictions.
stds = np.nan_to_num(np.concatenate(([0], df1.rstd.copy()))) #creo l'array traslato di 1 delle deviazioni standard
stds = np.resize(stds, stds.size-1) #elimino l'utlimo elemento
df1['Dev Standard'] = stds 

# Set the lower and upper bound in which the prevision must be included
df1['LowerBound'] = df1['Prev'] - df1['Dev Standard']
df1['UpperBound'] = df1['Prev'] + df1['Dev Standard']
df1.head()

# Set a predicate to establish whether the prevision is between the lower and the upper bound
# Apply the predicate to the whole dataset and create the class column. In the class column the value "1" means the predict value is satisfied, 0 otherwise
pred = (df1['Births'] >= df1['LowerBound']) & (df1['Births'] <= df1['UpperBound'])
df1['class'] = np.where(pred, '1', '0')
df1

# Plot the predicted values, the Lower Bound and the Upper Bound
plt.plot(df1.Prev, color='orange', label='Prevision')
plt.plot(df1.LowerBound, color='red', label='Lower Bound')
plt.plot(df1.UpperBound, color='blue', label='Upper Bound')
plt.scatter(df1.Date, df1.Births, color='black', s=2)
plt.legend(loc='upper left')
plt.show()

# Split the data set into training set and test set
from sklearn.model_selection import train_test_split
X = df1[['Births']].values.reshape(-1,1)
y = df1[['class']].values.reshape(-1, 1).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, test_size=0.33)

# Run the classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', C=10,  gamma=0.001)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate classifier performance with the Confusion Matrix and a classification report
from sklearn.metrics import classification_report, confusion_matrix
print('\nConfusion matrix')
print(confusion_matrix(y_test,y_pred))
print('\nClassification report')
print(classification_report(y_test,y_pred))