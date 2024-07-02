from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

uploaded = 'C:\\Users\\jfbad\\Downloads\\df_pca.csv'
df = pd.read_csv(uploaded, sep=',')

X = df.drop(['Cluster'], axis=1)
y = df['Cluster'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_knn = KNeighborsClassifier(algorithm='ball_tree', metric='manhattan', n_neighbors=7, weights='distance')
modelo_knn.fit(X_train, y_train)

print('La precisión del modelo con los datos de entrenamiento: {:.2%}'.format(modelo_knn.score(X_train, y_train)))
print('La precisión del modelo con los datos de practica: {:.2%}'.format(modelo_knn.score(X_test, y_test)))

filename = 'checkpoints/model.pkl'
pickle.dump(modelo_knn, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)