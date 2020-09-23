import numpy as np
import pandas as pd

from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

recipes = pd.read_csv("recipes_muffins_cupcakes.csv")

# sns.lmplot(x= 'Flour',y= 'Sugar',data=recipes, hue='Type', palette='Set1',fit_reg=False,scatter_kws={"s":70})
# plt.show()

type_label = np.where(recipes['Type']=='Muffin',0,1)
recipe_feature = recipes.columns.values[1:].tolist()

ingredients = recipes[['Flour','Sugar']].values

model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)

w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30,60)
yy = a * xx - (model.intercept_[0] / w[1])

b = model.support_vectors_[0]
yy_down = a * xx + (b[1]- a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx +(b[1] - a * b[0])

sns.lmplot(x='Flour',y='Sugar',data=recipes, hue='Type', palette='Set1',fit_reg=False,scatter_kws={"s":70})
# plt.plot(xx , yy, linewidth=2, color='black')
# plt.plot(xx , yy_down, 'k--')
# plt.plot(xx , yy_up, 'k--')

# plt.show()

F = int(input("Enter Flour amount:"))
S = int(input("Enter Sugar amount:"))

if(model.predict([[F,S]]) == 0):
    print("Its Muffin recipe")
else:
    print("Its Cupcake recipe")