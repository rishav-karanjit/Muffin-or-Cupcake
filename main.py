import numpy as np
import pandas as pd

from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

recipes = pd.read_csv("Muffin or Cupcake.csv")

sns.lmplot('Flour','Sugar',data=recipes, hue='Type', palette='Set1',fit_reg=False,scatter_kws={"s":70})
# plt.show()

type_label = np.where(recipes['Type']=='Muffin',0,1)
recipe_feature = recipes.columns.values[1:].tolist()

ingredients = recipes[['Flour','Sugar']].values

model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)