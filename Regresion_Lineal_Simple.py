#Practica de regresion lineal simple

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
#Nuestra variable independiente será el GDP per capita
X = df['GDP per capita'].to_numpy().reshape(-1,1)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2)

rls=linear_model.LinearRegression()
modelo=rls.fit(X_train,Y_train)
Y_pred=rls.predict(X_test)

#resultados del ajuste del modelo
print("Coeficientes:", rls.coef_)
print("Intercepto:", rls.intercept_)
print("R^2:", rls.score(X_test, Y_test))


#Datos de la regresión lineal
error=Y_test-Y_pred
ds_error=error.std()
ds_X=X_test.std()
error_st=ds_error/np.sqrt(506)
t1=rls.coef_/(error_st/ds_X)
print(t1)
#%% Parámetros para prueba de hipótesis B0
media_X=X_test.mean()
media_XC=pow(media_X,2)
var_X=X_test.var()
to=rls.intercept_/(error_st*np.sqrt(1+(media_XC/var_X)))
print(to)

#Gráfica del modelo
plt.scatter(X_test[:, 0], Y_test, label='Valores reales')
plt.plot(X_test[:, 0], Y_pred, color='r', label='Predicciones', linewidth=3)
plt.title('Regresión Lineal')
plt.xlabel('Número promedio de habitaciones')
plt.ylabel('Valor Mediano')
plt.legend()
plt.show()

#Gráfica de residuales (Homosedasticidad)
error = Y_test - Y_pred
plt.scatter(Y_pred, error, marker='*', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Ajustados')
plt.ylabel('Residuos')
plt.title('Homocedasticidad')
plt.show()

# Agregar una constante para el término independiente (B0)
X_train = sm.add_constant(X_train)

# Crear un modelo de regresión lineal con los datos de entrenamiento
modelo = sm.OLS(Y_train, X_train).fit()

# Obtener los resultados del modelo
print(modelo.summary())

# Ahora puedes usar el modelo para hacer predicciones en los datos de prueba
X_test = sm.add_constant(X_test)
Y_pred = modelo.predict(X_test)

# Imprimir las predicciones en los datos de prueba
print("Predicciones en los datos de prueba:")
print(Y_pred)

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

# Mostrar el gráfico
plt.show()
#%% Forma estadística de la normalidda (Shapiro-Wilk)
#Ho: Normalidad (p>0.05)
#H1: No normalidad (p<0.05)
names=[' Statistic', 'p-value']
test=stats.shapiro(modelo.resid)
lzip(names,test)