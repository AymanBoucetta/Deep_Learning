import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
tf.keras.utils.set_random_seed(42)



# Chargement
(x_train_full, y_train_full), (x_test, y_test) = \
tf.keras.datasets.mnist.load_data()


# Normalisation et transformation
x_train_full = x_train_full.reshape(-1, 784).astype("float32") /255
x_test = x_test.reshape(-1, 784).astype("float32") / 255


# Séparation entraînement / validation
x_train = x_train_full[:-10000]

y_train = y_train_full[:-10000]
x_val = x_train_full[-10000:]
y_val = y_train_full[-10000:]
print("Entraînement :", x_train.shape)
print("Validation :", x_val.shape)
print("Test :", x_test.shape)




#Q1. Pourquoi normaliser les pixels avant l'entraînement ?
###Pour accélérer l'apprentissage, stabiliser le gradient et éviter que de grandes valeurs dominent le modèle.

#Q2. Pourquoi transformer les images en vecteurs de 784 valeurs ?
###Pour transformer une image 2D (ex: $28 \times 28$ pixels) en une ligne de données 1D que les couches de neurones classiques peuvent lire.

#Q3. Quelle est la différence entre les ensembles d'entraînement, de validation et de test ?
###Entraînement : Pour que le modèle apprenne.

###Validation : Pour régler les paramètres et éviter le surapprentissage.

###Test : Pour évaluer la performance finale sur des données totalement inédites.


