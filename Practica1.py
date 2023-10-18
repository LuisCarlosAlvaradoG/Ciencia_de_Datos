import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import numpy as np
#import missingno as msno

#%%
# Cargar los datos
data = pd.read_csv('bundesliga_player.csv')

# Mostrar información básica sobre el DataFrame
print(data.info())
#%%
selected_columns = ['age', 'height', 'nationality', 'price', 'max_price', 'position', 'foot']
data = data[selected_columns]

#%%

# Medidas de tendencia central y dispersión
summary = data.describe()
data["nationality"].value_counts()
data["position"].value_counts()
data["foot"].value_counts()


# Análisis de Player Price
#Age
plt.figure()
data['age'].value_counts(normalize=True).plot.bar()
plt.title("Distribution of Age")
plt.xlabel("Player Age")
plt.ylabel("Percentage")
plt.show()
#Height
plt.figure()
data['height'].value_counts(normalize=True).plot.bar()
plt.title("Distribution of Height")
plt.xlabel("Player Height")
plt.ylabel("Percentage")
plt.show()
#Nationality
plt.figure()
data['nationality'].value_counts(normalize=True).plot.bar()
plt.title("Distribution of Nationality")
plt.xlabel("Player Nationality")
plt.ylabel("Percentage")
plt.xticks(fontsize = 6.5)
plt.show()
#MaxPrice
plt.figure()
data['max_price'].value_counts(normalize=True).plot.bar()
plt.title("Distribution of MaxPrice")
plt.xlabel("Player MaxPrice")
plt.ylabel("Percentage")
plt.xticks(fontsize = 5)

plt.show()
#Position
plt.figure()
data['position'].value_counts(normalize=True).plot.bar()
plt.title("Distribution of Position")
plt.xlabel("Player Position")
plt.ylabel("Percentage")
plt.show()
#Foot
plt.figure()
data['foot'].value_counts(normalize=True).plot.bar()
plt.title("Distribution of Foot")
plt.xlabel("Player Foot")
plt.ylabel("Percentage")
plt.show()

# Gráfica de distribución de Player Price
plt.figure()
sns.histplot(data['price'], kde=True)
plt.title("Distribution of Price")
plt.xlabel("Player Price (in million dollars)")
plt.ylabel("Frequency")
plt.show()

# Player Price con respecto a Player Age
plt.figure()
sns.boxplot(x='age', y="price", data=data)
plt.title("Player Price vs Player Age")
plt.xlabel("Player Age")
plt.ylabel("Player Price")
plt.show()
# Player Price con respecto a Player Height
plt.figure()
sns.boxplot(x='height', y="price", data=data)
plt.title("Player Price vs Player Height")
plt.xlabel("Player Height")
plt.ylabel("Player Price")
plt.xticks(rotation = 90)
plt.show()
# Player Price con respecto a Player Nationality
plt.figure()
sns.boxplot(x='nationality', y="price", data=data)
plt.title("Player Price vs Player Nationality")
plt.xlabel("Player Nationality")
plt.ylabel("Player Price")
plt.xticks(rotation = 90, fontsize = 6.5)
plt.show()
# Player Price con respecto a Player MaxPrice
plt.figure()
sns.boxplot(x='max_price', y="price", data=data)
plt.title("Player Price vs Player MaxPrice")
plt.xlabel("Player MaxPrice")
plt.ylabel("Player Price")
plt.xticks(rotation = 90, fontsize = 5)
plt.show()
# Player Price con respecto a Player Position
plt.figure()
sns.boxplot(x='position', y="price", data=data)
plt.title("Player Price vs Player Position")
plt.xlabel("Player Position")
plt.ylabel("Player Price")
plt.xticks(rotation = 90)
plt.show()
# Player Price con respecto a Player Foot
plt.figure()
sns.boxplot(x='foot', y="price", data=data)
plt.title("Player Price vs Player Foot")
plt.xlabel("Player Foot")
plt.ylabel("Player Price")
plt.show()
#%%
# Player Height con respecto a Player Position
plt.figure()
sns.boxplot(x='position', y="height", data=data)
plt.title("Player Height vs Player Position")
plt.xlabel("Player Position")
plt.ylabel("Player Height")
plt.xticks(rotation = 90)
plt.show()

# Player Height con respecto a Player Age
plt.figure()
sns.boxplot(x='age', y="height", data=data)
plt.title("Player Height vs Player Age")
plt.xlabel("Player Age")
plt.ylabel("Player Height")
plt.show()

# Player Position con respecto a Player Nationality
plt.figure()
sns.boxplot(x='age', y="nationality", data=data)
plt.title("Player Age vs Player Nationality ")
plt.xlabel("Player Age")
plt.ylabel("Player Nationality")
plt.yticks(fontsize = 5)
plt.show()

# Calcular la matriz de varianzas y covarianzas

selected_columns_cov_corr = ['age', 'height', 'price', 'max_price']
data_cov_corr = data[selected_columns_cov_corr]
cov_matrix = data_cov_corr.cov()
corr_matrix = data_cov_corr.corr()

# Visualizar la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pairplot para análisis bivariado
sns.pairplot(data_cov_corr, kind='reg')
plt.title("Pairplot")
plt.show()


# echo "# Ciencia_de_Datos" >> README.md
# git init
# git add README.md
# git commit -m "first commit"
# git branch -M main
# git remote add origin https://github.com/LuisCarlosAlvaradoG/Ciencia_de_Datos.git
# git push -u origin main

#Existing repository
#git remote add origin https://github.com/LuisCarlosAlvaradoG/Ciencia_de_Datos.git
# git branch -M main
# git push -u origin main