import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

tf.keras.utils.set_random_seed(42)

# --- Données ---
(x_train_full, y_train_full), _ = tf.keras.datasets.mnist.load_data()
x_train_full = x_train_full.reshape(-1, 784).astype("float32") / 255
x_train, y_train = x_train_full[:-10000], y_train_full[:-10000]
x_val, y_val = x_train_full[-10000:], y_train_full[-10000:]

# --- Modèle ---
modele = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu"),
    Dropout(0.0),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
modele.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

# --- Exercice 3 : Premier entraînement ---
debut = time.perf_counter()
historique = modele.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)
duree = time.perf_counter() - debut
print(f"Durée : {duree:.2f} secondes")
plt.figure(figsize=(10, 4))
plt.plot(historique.history["loss"], label="Train Loss")
plt.plot(historique.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()