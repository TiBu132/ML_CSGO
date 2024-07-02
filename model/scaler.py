from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

best_features=['Team','MatchWinner','Survived','RoundKills','RoundAssists','RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
       'EquipmentDifferenceValue', 'DifferenceEquipmentValueLastRound',]

upload='C:\\Users\\jfbad\\Downloads\\dataframe_modificado.csv'
df_scale=pd.read_csv(upload, sep=',')

x_temp=df_scale[best_features]
scaler = StandardScaler()
scaler_fitted = scaler.fit(x_temp)  # Ajustar el scaler con los datos de entrenamiento

# Guardar el scaler ajustado para usarlo posteriormente
filename = 'checkpoints/scaler_fitted.pkl'
pickle.dump(scaler_fitted, open(filename, 'wb'))