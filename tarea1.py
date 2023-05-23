# librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from umap import UMAP
from tensorflow.keras.datasets import mnist

# cargar datos entrenamiento y prueba, datos MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# dividir datos entre 255
# para que estén en un rango de 0 a 1 y sean más fáciles de trabajar
x_train = train_X.reshape(train_X.shape[0], -1) / 255.0
y_train = train_y
x_test = test_X.reshape(test_X.shape[0], -1) / 255.0
y_test = test_y

# Definir una función para extraer características HOG de una imagen
def extract_hog_features(img):
    return hog(img.reshape((28,28)), orientations=8, pixels_per_cell=(7,7), cells_per_block=(1,1))

# sacar las características HOG de los datos aplicando la función a continuacion
x_train_hog = np.apply_along_axis(extract_hog_features, 1, x_train)
x_test_hog = np.apply_along_axis(extract_hog_features, 1, x_test)

#crear y entrenar un modelo de regresión softmax  con los los datos de entrenamiento
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
softmax_reg.fit(x_train_hog, y_train)

# r4ealizar predicciones en los datos de prueba
y_pred = softmax_reg.predict(x_test_hog)

#ver  la precisión 
# ver la matriz de confusion
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

#print de resultados
print(f'Precisión del modelo: {accuracy:.4f}')
print('Matriz de confusión:')
print(confusion)

#ver precision por clase para havcer grafico
accuracy_by_class = np.diag(confusion) / np.sum(confusion, axis=1)
plt.bar(np.arange(10), accuracy_by_class)
plt.xticks(np.arange(10), np.arange(10))
plt.xlabel('Clase')
plt.ylabel('Precisión')
plt.title('Precisión por clase')
plt.show()

#hacer grafico de calor
plt.imshow(confusion, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(10), np.arange(10))
plt.yticks(np.arange(10), np.arange(10))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de confusión')
plt.show()

#vizualizar 2D de los vectores de entrada utilizando umap a continuacion
umap = UMAP(n_components=2, random_state=42)
x_umap = umap.fit_transform(x_test_hog)

#Graficar  datos  con UMAP.
plt.scatter(x_umap[:,0], x_umap[:,1], c=y_test, cmap='rainbow', alpha=0.5)
plt.colorbar()
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Visualización 2D con UMAP')
plt.show()



