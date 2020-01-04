import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

url = 'https://raw.githubusercontent.com/okHotel/ssd_project/master/data_set/daily-min-temperatures.csv'
df = pd.read_csv(url, names=['Date', 'Temp'], skiprows=2556) #carico solo i dati degli ultimi 3 anni
df = df.sort_values(by=['Date']) #ordino i record in base alla data

#print(df.head())
plt.plot(df.Temp)
plt.title('Temperature trend over 3 years')
plt.ylabel('Temperature')
plt.show()

w = 20
rmean = df.Temp.rolling(window=w, min_periods=1).mean() #calcolo la media mobile
rstd = df.Temp.rolling(window=w, min_periods=1).std() #calcolo la deviazione standard mobile
df['rmean'] = rmean
df['rstd'] = rstd

plt.plot(df.Temp, color='blue', label='Dataset')
plt.plot(df.rmean, color='orange', label='Rolling Mean (20)')
plt.plot(df.rstd, color='green', label='Rolling Standard Deviation (20)')
plt.legend(loc='upper left')
plt.show()

# SVM prevision
# Riformatto i Dati in un nuovo DataFrame
df1 = pd.DataFrame()
df1['Date'] = df.Date.copy()
df1['Temp'] = df.Temp.copy()

means = np.concatenate(([0], df.rmean.values.copy())) # creo l'array delle prevsioni con la prima = 0 e scifto tutti gli altri di 1
means = np.resize(means, means.size - 1) # elimino l'ultimo elemento
df1['Prev'] = means # ogni dato è la media mobile calcolata sul dato precedente

stds = np.nan_to_num(np.concatenate(([0], df.rstd.values.copy()))) # creo l'array traslato di 1 delle deviazioni standard
stds = np.resize(stds, stds.size-1) # elimino l'utlimo elemento
df1['Dev Standard'] = stds 

df1['LowerBound'] = df1['Prev'] - df1['Dev Standard'] # lower bound del range entro il quale il dato reale dovrebbe cadere
df1['UpperBound'] = df1['Prev'] + df1['Dev Standard'] # upper bound del range entro il quale il dato reale dovrebbe cadere

pred = (df1['Temp'] >= df1['LowerBound']) & (df1['Temp'] <= df1['UpperBound']) # predicato per verificare se il dato è dentro al range
df1['Class'] = np.where(pred, '1', '0') # predicato soddisfatto mette 1 altrimenti 0
print(df1.head())

plt.plot(df1.Prev, color='orange', label='Prev')
plt.plot(df1.LowerBound, color='blue')
plt.plot(df1.UpperBound, color='blue')
plt.scatter(df1.Date, df1.Temp, color='black', s=2, label='Temp')
plt.legend(loc='upper left')
plt.show()

# Spilt dataset into train and test
X = df1[['Temp']].values.reshape(-1,1)
Y = df1[['Class']].values.reshape(-1,1).ravel()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.66, test_size=0.33)

classifier = SVC(kernel='rbf', C=100, gamma=0.2) # classifier definition
classifier.fit(X_train, Y_train) # classifier training
y_pred = classifier.predict(X_test) # classifier prevision

print('\n')
print('Confisuin matrix\n ', confusion_matrix(Y_test,y_pred)) # confusion matrix 
print('Classification report\n', classification_report(Y_test,y_pred)) # classification properties
# precision = TP / TP + FP
# recall = TP / TP + FN
# f1 = 2* (Prec * Rec) / (Prec + Rec)




