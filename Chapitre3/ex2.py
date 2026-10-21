import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)
def creer_modele(neurones1=128, neurones2=64,
dropout=0.0, activation="relu",

learning_rate=0.001,
optimiseur="adam"):

    modele = Sequential([
    Input(shape=(784,)),
    Dense(neurones1, activation=activation),
    Dropout(dropout),
    Dense(neurones2, activation=activation),
    Dense(10, activation="softmax")
    ])
    if optimiseur == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimiseur == "sgd":
        opt = SGD(learning_rate=learning_rate)
    elif optimiseur == "rmsprop":
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Optimiseur inconnu")
    modele.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
)

    return modele

modele = creer_modele()
modele.summary()




#Q4. Calculer manuellement le nombre total de paramètres entraînables.
#Q5. Pourquoi utilise-t-on ReLU dans les couches cachées et Softmax dans la couche de sortie ?
#Q6. Pourquoi utiliser sparse_categorical_crossentropy dans cet exemple ?


