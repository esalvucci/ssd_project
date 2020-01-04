import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/okHotel/ssd_project/master/data_set/monthly-sunspots.csv"
df = pd.read_csv(url)
df = df.sort_values(by=['Month'])

#data = pd.Series()
plt.figure(figsize=(19,15))
plt.plot(df.Sunspots)
plt.ylabel('Number of Sunspots')
plt.xlabel('Record ID (Month)')
plt.title('Monthly-sunspots over 350 years')
plt.show()

df.insert(0, 'id', range(0,len(df)))

from sklearn.model_selection import train_test_split
X = df[['id']].values.reshape(-1,1)
y = df[['Sunspots']].values.reshape(-1,1).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


from sklearn.svm import SVR
# !!! Con kernel rbf sbaglia un po' le previsioni.
# !!! Con kernel linear molto meglio !!!
# C quanto penalizza i pattern fuori rangwe
# gamma 
# epsilon passo con cui ottimizzare le scelte, quanto accetti di essere distante dalla funzione trovata con svr
regressor = SVR(kernel='rbf', C=10, gamma=0.001, epsilon=0.1)
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df.head(50))


#GRAFICO SUL TEST SET 6 Visualising the Support Vector Regression results
#fucsia dati del dataset
#verde funzione di regressione
plt.figure(figsize=(19,15))
plt.scatter(X, y, color = 'magenta') 
plt.plot(X, regressor.predict(X), color = 'green')
plt.ylabel('Number of Sunspots')
plt.xlabel('Record ID (Month)')
plt.title('Support Vector Regression model')
plt.show()


import plotly.graph_objects as go
# predicted data = dati predetti con la regressione
# test data = dati reali usati per il test

#FUNZIONE DI REGRESSIONE
# y_pred valori predetti, y_test valori effettivi
plt.figure(figsize=(10,10))
plt.scatter(X_test.flatten(), y_pred.flatten(), color='orange', label='Predicted data')
plt.scatter(X_test.flatten(), y_test.flatten(), color='royalblue', label='Test data')
plt.ylabel('Number of Sunspots')
plt.xlabel('Record ID (Month)')
plt.title('Comparison between predicted and test values')
plt.show()

#per valutare le prestazioni di SVR, valore compreso tra 0 e 1 
from sklearn.metrics import explained_variance_score 
# Evaluation
# 1 - ( Var(y' - y)/Var(y) )
print("\nSVR performance")
print(explained_variance_score(y_test, y_pred))