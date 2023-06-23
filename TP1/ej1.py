# Import all libraries that we will need
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random as rd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv("https://raw.githubusercontent.com/anyoneai/notebooks/main/datasets/project2_players_df.csv")

data.dropna(subset=["PTS"], inplace=True)
data.head()
features = ['PTS', 'REB','AST','REB', 'STL']
features = ['PTS', 'REB','AST','REB', 'STL']
# graficamos cada una de las caracteristicas
for feature in features:
    sns.scatterplot(data=data, x=feature, y ='SALARY', size='SEASON_EXP', hue='POSITION').set_title(f'{feature} vs SALARY')
    plt.show()
# graficamos cada una de las caracteristicas
#for feature in features:
#    sns.scatterplot(data=data, x=feature, y ='SALARY', size='SEASON_EXP', hue='POSITION').set_title(f'{feature} vs SALARY')
#    plt.show()

X = data.drop(columns=['SALARY'])
# Pon aqui tu variable dependiente
y = data['SALARY']

# Ahora usamos train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

class BaselineModel():
  """
  Un modelo base que siempre retorna siempre el mismo valor,
  la media de los salarios de los jugadores en los datos de entrenamiento
  """
  def fit(self, y_train):
    self.y_train = y_train
    self.mean = round(sum(y_train)/len(y_train))
    """
    Entrena con los datos de entrenamiento

    Parametros
    ----------
    y_train: Union[pd.Series, np.ndarray]
        una serie de pandas o array de numpy conteniendo la informacion del salario
    """


  def predict(self, X):
    length = len(X)
    self.fitted_values = np.array([self.mean] * length)
    return self.fitted_values
    """
    Salarios predichos

    Parameteros
    ----------
    X: Union[pd.DataFrame, pd.Series, np.ndarray]
        una serie de pandas, un dataframe o un array de numpy

    Returns
    -------
    np.ndarray
        un array de numpy del mismo largo que X, con todos los elementos iguales
        a la media calculada en fit()
    """
    baseline = BaselineModel()
    # Entrenamos el modelo
    baseline.fit(y_train)
    # Obtenemos el error medio absoluto
    mae_baseline = mean_absolute_error(y, baseline.predict(X['PTS']))
    print(f'El mae para el modelo base es {round(mae_baseline, 2)}')
    # Pon aqui tus variable independientes

    # Instanciamos el modelo
baseline = BaselineModel()
# Entrenamos el modelo
baseline.fit(y_train)
# Obtenemos el error medio absoluto
mae_baseline = mean_absolute_error(y, baseline.predict(X['PTS']))
print(f'El mae para el modelo base es {round(mae_baseline, 2)}')

# Instanciamos el escalador

Scaler = RobustScaler()
# Fit the scaler and transform the data

pts_scaled_X_train = Scaler.fit_transform(X_train['PTS'].values.reshape(-1,1))
pts_scaled_X_test = Scaler.transform(X_test['PTS'].values.reshape(-1,1))

##!!!!TAREA A REALIZAR!!!!

# Instancien el modelo con un random state (estado aleatorio)
model=SGDRegressor(random_state=42)

# Entrenen el modelo con los datos
model.fit(pts_scaled_X_train, y_train)

# Obtengan una prediccion
predictions = model.predict(pts_scaled_X_test)

# Obtengan el error absolut medio para el modelo
mae_pts = mean_absolute_error(y_test, predictions)

#  Hagan un print de los resultados obtenidos

if mae_baseline > mae_pts:
    print(f'El error de nuestro modelo base es mayor por {round((mae_baseline - mae_pts), 2)}')
else:
    print(f'El error de nuestra regresion simple es mayor por {round((mae_pts - mae_baseline),2)}')
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
import math

def search_best_hyperparameters(max_iter_values, eta0_values):
    result = {
        "hyperparameters": {"max_iter": None, "eta0": None},
        "mae": math.inf
    }

    for max_iter in max_iter_values:
        for eta0 in eta0_values:
                model = SGDRegressor(max_iter=max_iter, eta0=eta0)
                model.fit(pts_scaled_X_train, y_train)
                predictions = model.predict(pts_scaled_X_test)
                mae = mean_absolute_error(y_test, predictions)

                if mae < result["mae"]:
                    result["mae"] = mae
                    result["hyperparameters"]["max_iter"] = max_iter
                    result["hyperparameters"]["eta0"] = eta0

    return result

max_iter = [10_000, 1_000_000]
eta0 = [0.0001, 0.001, 0.01, 0.1]

x= search_best_hyperparameters(max_iter, eta0)
print(x)

print("MAE para cada modelo:")
print(f"Modelo base: {round(mae_baseline,2)}")
print(f"SGDRegressor con hiperparametros por default: {round(mae_pts, 2)}")
print(f"SGDRegressor con los mejores hiperparametros: {round(result['mae'],2)}")