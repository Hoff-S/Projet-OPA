import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta

# Fonction pour inverser la transformation Box-Cox
def invboxcox(y, lmbda):
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lmbda * y + 1) / lmbda)

# Définir l'URL du serveur MLflow
mlflow.set_tracking_uri('http://86.77.27.174:8080/')

'''
# ID run du modèle 
run_id = "6df6abc77def4d67a07bc8b3f6e7837a"
local_model_path = f"C:/Users/apsho/OneDrive/Documents/Formations/DataScientest/Projet/mlruns/594249274040179504/{run_id}/artifacts/sarima_crypto/model.pkl"

# Charger les paramètres du modèle
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
lmbda = float(run.data.params['lambda'])

# Charger le modèle SARIMA depuis le fichier local
with open(local_model_path, "rb") as f:
    loaded_model = pickle.load(f)

# Obtenir la date actuelle
current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# Faire des prédictions pour demain avec le modèle chargé
steps = 1
forecast = loaded_model.get_forecast(steps=steps)
forecast_dates = [current_date + timedelta(days=i) for i in range(steps)]
predictions = forecast.summary_frame()
predictions.index = forecast_dates

# Appliquer l'inverse de la transformation Box-Cox
predictions['mean'] = invboxcox(predictions['mean'], lmbda)
predictions['mean_ci_lower'] = invboxcox(predictions['mean_ci_lower'], lmbda)
predictions['mean_ci_upper'] = invboxcox(predictions['mean_ci_upper'], lmbda)

# Afficher la prédiction pour demain
print("Prédiction pour demain:")
print(predictions[['mean', 'mean_ci_lower', 'mean_ci_upper']])

# Visualisation de la prédiction
fig, ax = plt.subplots(figsize=(10, 5))
predictions['mean'].plot(ax=ax, style='k--', label='Prévision')
ax.fill_between(predictions.index, predictions['mean_ci_lower'], predictions['mean_ci_upper'], color='k', alpha=0.1)
plt.legend()
plt.title('Prédiction pour demain')
plt.xlabel('Date')
plt.ylabel('Valeur prédites')
plt.show()

print(predictions)'''