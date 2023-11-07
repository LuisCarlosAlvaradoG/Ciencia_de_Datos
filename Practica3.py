import pandas as pd
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
#Cargar la base de datos
data = pd.read_csv('Diet.csv')
data["diference"] = round(data["weight"] - data["weight6weeks"], 1)
# Filtrar los estudiantes graduados
sns.countplot(data = data, x='Diet')
plt.xticks(rotation = 90)
plt.show()

#Resumen Estadístico
resumen = rp.summary_cont(data['diference'].groupby(data['Diet']))

#BoxPlot
sns.set(style="whitegrid")
sns.boxplot(x='Diet', y='diference', data=data)
plt.xticks(rotation=90)
plt.show()

# Cumplimiento de supuestos
#Normalidad prueba de Shapiro-Wilk
#Ho:Normalidad(p>0.05)
#H1: No normalidad (p<0.05)
normality_test = pg.normality(data, dv = 'diference', group = 'Diet')
print("Prueba de Normalidad (Shapiro-Wilk):\n", normality_test)
         
#Homocedasticidad prueba de Levene (sin normalidad)
#Ho:Homocedasticidad (p>0.05)
#H1: No Homocedasticidad (p<0.05)
print(pg.homoscedasticity(data, dv='diference', 
                    group='diference',method='levene'))

#Homocedasticidad prueba de Bartlett (con normalidad)
#Ho:Homocedasticidad (p>0.05)
#H1: No Homocedasticidad (p<0.05)
print(pg.homoscedasticity(data, dv='diference', 
                    group='Diet',method='bartlett'))
# One way ANOVA
# Typ = 2 calcula las sumas de cuadrados de tipo II, que es apropiado para ANOVAs balanceados y desequilibrados
model =ols('diference ~ Diet', data=data).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
print("\nTabla ANOVA:")
print(anova_table)

#anova(gender)
#anova(religion)

# residuos = model.resid

# plt.figure(figsize=(8, 6))
# sns.regplot(x=model.fittedvalues, y=residuos, lowess=True, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})
# plt.title("Gráfico de Residuos")
# plt.xlabel("Valores Ajustados")
# plt.ylabel("Residuos Estandarizados")
# plt.grid(True)
# plt.show()
  
#Comparación múltiple Prueba de Tukey
comp = mc.MultiComparison(data['diference'],data['Diet'])
post_hoc_res = comp.tukeyhsd()
print("\nComparaciones múltiples (Tukey):")
print(post_hoc_res.summary())

# Two ways ANOVA
#Ho:m1=m2=m3+.... (p>0.05)
#H1: mi dif mj (p<0.05)
model =ols('diference ~ Diet + gender+Diet:gender ', data=data).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
print("\nTabla ANOVA:")
print(anova_table)

# Interacción entre variables
model =ols('diference ~ Diet:gender', data=data).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
print("\nTabla ANOVA:")
print(anova_table)

# Prueba de Tukey (HSD)
interaction_groups = "Diet" + data.Diet.astype(str) + " & " + "gender" + data.gender.astype(str)
comp = mc.MultiComparison(data["diference"], interaction_groups)
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())