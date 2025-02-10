import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generar datos de ejemplo
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Guardar el modelo en un archivo .pkl
with open("Modelo_Entrenado.pkl", "wb") as f:
    pickle.dump(model, f)


print("Modelo entrenado y guardado en 'modelo.pkl'")