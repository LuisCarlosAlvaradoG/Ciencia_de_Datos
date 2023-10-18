#Practica de regresion lineal multiple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#Leer los datos
data_path = "2019.csv"
df = pd.read_csv(data_path)

#La variable dependiente es el score de felicidad
Y = df['Score'].to_numpy()
#Nuestras variables independientes
X = df[['GDP per capita',
 'Social support',
 'Healthy life expectancy',
 'Freedom to make life choices',
 'Generosity',
 'Perceptions of corruption']].to_numpy()


X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2)
X_train = sm.add_constant(X_train)
modelo = sm.OLS(Y_train, X_train).fit()

# Obtener los resultados del modelo
print(modelo.summary())

# Ahora puedes usar el modelo para hacer predicciones en los datos de prueba
X_test = sm.add_constant(X_test)
Y_pred = modelo.predict(X_test)

#%% Visualizar homocedasticidad 
error=Y_test-Y_pred
plt.scatter(Y_pred, error, marker='*', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Ajustados')
plt.ylabel('Residuos')
plt.title('Homocedasticidad')
plt.show()

#%% Forma Estadística de Homocedasticidad
#Breusch-Pagan
#H0: Homocedasticidad (p>0.05)
#H1: No homocedasticidad (p<0.05)
names=['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(modelo.resid, X_train)
lzip(names, test)

#%% Forma gráfica de la  normalidad de los residuos
plt.figure()
plt.hist(modelo.resid)
plt.show()

#%% QQ plot

qq_plot = sm.qqplot(modelo.resid, line='45', fit=True)

# Personalizar la apariencia del QQ plot
plt.title("QQ Plot de los Residuos")
plt.xlabel("Cuantiles Teóricos")
plt.ylabel("Cuantiles de los Residuos")
plt.show()
#%% Forma estadística de la normalidda (Shapiro-Wilk)
#Ho: Normalidad (p>0.05)
#H1: No normalidad (p<0.05)
names=[' Statistic', 'p-value']
test=stats.shapiro(modelo.resid)
lzip(names,test)

## QUITAR VARIABLES NO SIGNIFICATIVAS
#Leer los datos
data_path = "2019.csv"
df = pd.read_csv(data_path)

#La variable dependiente es el score de felicidad
Y = df['Score'].to_numpy()
#Nuestras variables independientes
X = df[['GDP per capita',
 'Social support',
 'Healthy life expectancy',
 'Freedom to make life choices']].to_numpy()


X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2)
X_train = sm.add_constant(X_train)
modelo = sm.OLS(Y_train, X_train).fit()

# Obtener los resultados del modelo
print(modelo.summary())

# Ahora puedes usar el modelo para hacer predicciones en los datos de prueba
X_test = sm.add_constant(X_test)
Y_pred = modelo.predict(X_test)

#%% Visualizar homocedasticidad 
error=Y_test-Y_pred
plt.scatter(Y_pred, error, marker='*', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Ajustados')
plt.ylabel('Residuos')
plt.title('Homocedasticidad')
plt.show()

#%% Forma Estadística de Homocedasticidad
#Breusch-Pagan
#H0: Homocedasticidad (p>0.05)
#H1: No homocedasticidad (p<0.05)
names=['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(modelo.resid, X_train)
lzip(names, test)

#%% Forma gráfica de la  normalidad de los residuos
plt.figure()
plt.hist(modelo.resid)
plt.show()

#%% QQ plot

qq_plot = sm.qqplot(modelo.resid, line='45', fit=True)

# Personalizar la apariencia del QQ plot
plt.title("QQ Plot de los Residuos")
plt.xlabel("Cuantiles Teóricos")
plt.ylabel("Cuantiles de los Residuos")
plt.show()
#%% Forma estadística de la normalidda (Shapiro-Wilk)
#Ho: Normalidad (p>0.05)
#H1: No normalidad (p<0.05)
names=[' Statistic', 'p-value']
test=stats.shapiro(modelo.resid)
lzip(names,test)