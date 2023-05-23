Descripción del proyecto
Este proyecto utiliza el conjunto de datos MNIST para entrenar un modelo de regresión softmax que sea capaz de reconocer dígitos escritos a mano. Se utiliza la técnica de extracción de características HOG (Histogram of Oriented Gradients) para transformar las imágenes en vectores de características que puedan ser utilizados por el modelo.

Además, se utiliza la técnica de reducción de la dimensionalidad UMAP para visualizar los vectores de características en un espacio 2D y poder observar cómo se agrupan las diferentes etiquetas.

Se utilizó el conjunto de datos MNIST

pasos:

1. Asegurarse de tener las siguientes librerías instaladas: numpy, pandas, matplotlib, scikit-image, scikit-learn, umap-learn y tensorflow.
2. Ejecutar el archivo de código fuente "tarea1.py" desde la línea de comandos o el IDE de preferencia.
3. Esperar a que el modelo se entrene y realice las predicciones en los datos de prueba.
4. Se mostrará la precisión del modelo y la matriz de confusión. Además, se graficará la precisión por clase y la matriz de confusión.
6. También se mostrará una visualización 2D de los vectores de características utilizando UMAP.


