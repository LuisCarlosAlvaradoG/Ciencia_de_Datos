#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rociocarrasco
"""
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import mglearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#import seaborn as sns

# Cargar datos
data = pd.read_csv('creditcard.csv')
creditcard_transactions = data.iloc[:,1:-2]
y = data.Class

# Correlación entre las variables
cor=creditcard_transactions.corr()
# Forma gráfica (pocas variables)
# plt.figure()
# sns.pairplot(data)
# plt.show()
# Varianza de las variables
print(creditcard_transactions.var())

#%% Estandarizar los datos media 0 y desviación estándar 1
#Normales y sin outliers
scaler=StandardScaler()
scaler.fit(creditcard_transactions)
scaled_data=scaler.transform(creditcard_transactions)

# MinMaxScaler transformará los valores proporcionalmente dentro del rango [0,1]
#Presencia de outliers, preserva la forma de los datos
# scaler=MinMaxScaler()
# scaler.fit(data)
# scaled_data=scaler.transform(data)

#%% Algoritmo pca
pca=PCA()
#pca=PCA(n_components=10) # Indicar el número de componentes principales 
pca.fit(scaled_data)

# Ponderación de los componentes principales (vectores propios)
pca_score=pd.DataFrame(data    = pca.components_, columns = creditcard_transactions.columns,)

# Mapa de calor para visualizar in influencia de las variables
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
componentes = pca.components_
plt.imshow(componentes.T, cmap='plasma', aspect='auto')
plt.yticks(range(len(creditcard_transactions.columns)), creditcard_transactions.columns)
plt.xticks(range(len(creditcard_transactions.columns)), np.arange(pca.n_components_)+ 1)
plt.grid(False)
plt.colorbar();

#%% Gráfica del aporte a cada componente principal
# Aporte al primer componente principal 
matrix_transform = pca.components_.T
plt.bar(np.arange(28),matrix_transform[:,0])
plt.xticks(range(len(creditcard_transactions.columns)), creditcard_transactions.columns,rotation = 90)
plt.ylabel('Loading Score')
plt.show()

#%%Obtener las primeras 10 variables con mayor aporte
# Pesos 
loading_scores = pd.DataFrame(pca.components_[0])
#Nombre de las columnas
loading_scores.index=creditcard_transactions.columns
# Ordena de mayor a menor los pesos
sorted_loading_scores = loading_scores[0].abs().sort_values(ascending=False)
#Selección de las 10 variables que más aportan a cada componente principal
top_10_variables= sorted_loading_scores[0:10].index.values
print(top_10_variables)


# Nuevas variables,components principales
pca_data=pca.transform(scaled_data) 

mglearn.discrete_scatter(pca_data[:,0],pca_data[:,1], y)
plt.legend(creditcard_transactions.columns,loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


## Forma manual de obtener los componentes princiaples
# pcas = np.dot(pca.components_, scaled_data.T)
# pcas = pd.DataFrame(pcas)
# pcas = pcas.transpose()
#%%Porcentaje de varianza explicada por cada componente principal proporciona
#Lambda/suma_Lambda (valor_propio/suma_valores_propios)
per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)

plt.figure(figsize=(10, 5))
bars=plt.bar(np.arange(len(per_var)), per_var, alpha=0.7)
for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)
plt.xlabel('Componente Principal')
plt.ylabel(' % Varianza Explicada')
plt.xticks(np.arange(len(per_var)))
plt.title('Varianza Explicada por Cada Componente Principal')
plt.show()

#%% Porcentaje de varianza acumulado de los componentes
porcent_acum = np.cumsum(per_var) 

plt.figure()
plt.plot(porcent_acum)
plt.xlabel('Número de componentes')
plt.ylabel('Varianza (%)')  # for each component
plt.title('Porcentaje de varianza acumulada')
plt.show()
    
#%% Recuperar los datos originales
# rec = pca.inverse_transform(X=pcas)
# # Forma manual de recuperar los datos scalados
# rec_m = np.dot(pca_data, pca_score)
# rec = pd.DataFrame(rec,columns = data.columns)                
# origin_data = scaler.inverse_transform(rec)
                    
#%% Escalar los datos
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(creditcard_transactions)

#%% PCA
# pca = PCA() 
# pca.fit(scaled_data)
pca_data = pd.DataFrame(pca.transform(scaled_data))

#%% Seleccionar componentes principales para el entrenamiento
X = pca_data.iloc[:, 0:2]  # Asumiendo que queremos usar las primeras 6 componentes

#%% Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#%% Regresión logística
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#%% Evaluar el modelo
accuracy = log_reg.score(X_test, y_test)
print("Accuracy on test set:", accuracy)

#%% Realizar predicciones (opcional)
Y_predict = log_reg.predict(X_test)

#%% Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, Y_predict)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
xticklabels=['no fraude', 'fraude'], yticklabels=['no fraude', 'fraude'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


#%% Regresión logística sin utilizar PCA
X_train2, X_test2, y_train2, y_test2 = train_test_split(pca_data, y, test_size=0.2, random_state=42)
log_reg2 = LogisticRegression(max_iter=10000)  # Aumentar max_iter si es necesario
log_reg2.fit(X_train2, y_train2)

# Evaluar el modelo
accuracy = log_reg2.score(X_test2, y_test2)
print("Accuracy on test set:", accuracy)

# Realizar predicciones (opcional)
Y_predict2 = log_reg2.predict(X_test2)

# Calcular la matriz de confusión
conf_matrix2 = confusion_matrix(y_test2, Y_predict2)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='g', cmap='Blues', 
xticklabels=['no fraude', 'fraude'], yticklabels=['no fraude', 'fraude'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

