from sklearn.decomposition import PCA
import pandas as pd
import pickle

scaler_fitted = pickle.load(open('checkpoints/scaler_fitted.pkl', 'rb'))

upload = 'C:\\Users\\jfbad\\Downloads\\dataframe_modificado.csv'
df_scale = pd.read_csv(upload, sep=',')
best_features = ['Team','MatchWinner','Survived','RoundKills','RoundAssists',
                 'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
                 'EquipmentDifferenceValue', 'DifferenceEquipmentValueLastRound']

x_temp = df_scale[best_features]
x_scaled = scaler_fitted.transform(x_temp)
pca = PCA(n_components=9,svd_solver='full')
pca_fitted = pca.fit(x_scaled)  # Ajustar el PCA con los datos escalados

# Guardar el PCA ajustado para usarlo posteriormente
filename = 'checkpoints/pca_fitted.pkl'
pickle.dump(pca_fitted, open(filename, 'wb'))