import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

tf.keras.utils.set_random_seed(42)

# --- Données ---
(x_train_full, y_train_full), _ = tf.keras.datasets.mnist.load_data()
x_train_full = x_train_full.reshape(-1, 784).astype("float32") / 255
x_train, y_train = x_train_full[:-10000], y_train_full[:-10000]
x_val, y_val = x_train_full[-10000:], y_train_full[-10000:]

# --- Définition des expériences ---
experiences = [
    {"nom": "E1", "lr": 0.1,    "epochs": 10, "batch_size": 32},
    {"nom": "E2", "lr": 0.01,   "epochs": 10, "batch_size": 32},
    {"nom": "E3", "lr": 0.001,  "epochs": 10, "batch_size": 32},
    {"nom": "E4", "lr": 0.0001, "epochs": 10, "batch_size": 32},
]

plt.figure(figsize=(10, 6))

# --- Boucle d'entraînement pour chaque expérience ---
for exp in experiences:
    print(f"\n--- Lancement de l'expérience {exp['nom']} (lr={exp['lr']}) ---")
    
    # 1. Construction du modèle
    modele = Sequential([
        Input(shape=(784,)),
        Dense(128, activation="relu"),
        Dropout(0.0),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    
    # 1 & 2. Modification du learning rate via l'optimiseur Adam
    modele.compile(optimizer=Adam(learning_rate=exp["lr"]),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    
    # Entraînement avec mesure du temps (4. Durée d'entraînement)
    debut = time.perf_counter()
    historique = modele.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=exp["epochs"],
        batch_size=exp["batch_size"],
        verbose=0  # Mettre à 1 si vous souhaitez voir le détail de chaque époque
    )
    duree = time.perf_counter() - debut
    
    # 4. Relevé de la meilleure validation accuracy
    meilleure_val_acc = max(historique.history["val_accuracy"])
    
    print(f"Durée : {duree:.2f} secondes | Meilleure Validation Accuracy : {meilleure_val_acc * 100:.2f}%")
    
    # 3. Traçage des courbes de validation loss sur la même figure
    plt.plot(historique.history["val_loss"], label=f"{exp['nom']} (lr={exp['lr']})")

# --- Mise en forme du graphique ---
plt.title("Comparaison de la Validation Loss selon le Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.show()