# Reconnaissance des émotions faciales avec CNN

## 1. Présentation du projet

Ce projet consiste à développer un modèle de Deep Learning capable de reconnaître les émotions humaines à partir d'images de visages.

Le modèle utilise un réseau de neurones convolutif (CNN) entraîné sur le dataset FER-2013. Deux architectures sont comparées : un CNN simple et un CNN amélioré visant à limiter le surapprentissage.

## 2. Objectifs

- Préparer et normaliser les images.
- Construire un modèle CNN de classification.
- Comparer deux architectures.
- Évaluer les performances sur un ensemble de test.
- Analyser les limites de la reconnaissance des émotions.

## 3. Technologies utilisées

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 4. Dataset

Le projet utilise le dataset FER-2013. Les images sont en niveaux de gris et ont une résolution de 48 × 48 pixels.

Le dataset contient sept classes :

- Angry : colère
- Disgust : dégoût
- Fear : peur
- Happy : joie
- Neutral : neutralité
- Sad : tristesse
- Surprise : surprise

### Répartition des données

| Ensemble | Nombre d'images |
|---|---:|
| Entraînement | 22 967 |
| Validation | 5 742 |
| Test | 7 178 |
| **Total** | **35 887** |

Les données d'entraînement ont été divisées en ensembles d'entraînement et de validation. La séparation utilise une graine aléatoire fixe et une stratification des classes.

### Prétraitement

Les images sont normalisées en divisant les valeurs des pixels par 255. Les pixels sont ainsi représentés entre 0 et 1.

La forme de chaque image est `(48, 48, 1)`.

## 5. Première architecture : CNN simple

Le premier modèle utilise une architecture convolutive classique.

Elle comprend :
- Trois couches convolutionnelles avec 32, 64 et 128 filtres.
- Des couches MaxPooling pour réduire les dimensions.
- Une couche Flatten.
- Une couche Dense de 128 neurones avec activation ReLU.
- Une couche de sortie de sept neurones avec Softmax.

L'entraînement utilise l'optimiseur Adam et la fonction de perte `sparse_categorical_crossentropy`.

### Résultats

Le modèle atteint une accuracy d'entraînement de 76,07 % à la quatrième époque. Cependant, son accuracy de validation atteint au maximum 55,66 % avant de diminuer.

Cette différence indique un surapprentissage : le modèle apprend bien les données d'entraînement, mais généralise moins bien sur des images non utilisées pendant cet entraînement.

## 6. Deuxième architecture : CNN amélioré

Une deuxième architecture est proposée pour améliorer la généralisation.

Elle comprend :
- Une augmentation des données : retournement horizontal, rotation, translation et zoom.
- Trois blocs convolutionnels avec 32, 64 et 128 filtres.
- Des couches Batch Normalization.
- Des couches MaxPooling.
- Plusieurs couches Dropout.
- Une couche Dense de 128 neurones avec régularisation L2.
- Une couche de sortie Softmax pour les sept émotions.

### Stratégie d'entraînement

Le modèle utilise Adam avec un taux d'apprentissage initial de 0,001.

Deux mécanismes complètent l'entraînement :
- `EarlyStopping` permet d'arrêter l'entraînement si la perte de validation ne s'améliore plus.
- `ReduceLROnPlateau` réduit le taux d'apprentissage lorsque la progression ralentit.

L'entraînement est configuré pour un maximum de 30 époques, avec une taille de batch de 64.

### Résultats

Après 30 époques, le modèle atteint :
- Accuracy d'entraînement : 52,26 %.
- Meilleure accuracy de validation : 56,93 %.
- Meilleure perte de validation : 1,1861.

Le modèle amélioré obtient une meilleure accuracy de validation et une perte de validation plus faible que le CNN simple.

## 7. Comparaison des modèles

| Critère | CNN simple | CNN amélioré |
|---|---|---|
| Architecture | CNN classique | CNN avec régularisation |
| Augmentation des données | Non | Oui |
| Batch Normalization | Non | Oui |
| Dropout | Non | Oui |
| Régularisation L2 | Non | Oui |
| Surapprentissage | Important | Réduit |
| Accuracy maximale de validation | 55,66 % | 56,93 % |

Le CNN amélioré constitue la meilleure des deux architectures selon les résultats de validation observés. Son avantage reste toutefois modéré.

## 8. Limites et améliorations possibles

Les performances peuvent être limitées par plusieurs facteurs :
- Le déséquilibre entre les classes.
- La difficulté à distinguer certaines émotions visuellement proches.
- La faible résolution des images.
- Les variations d'éclairage, d'orientation et d'expression faciale.

Des améliorations possibles consistent à utiliser des poids de classes, à analyser la matrice de confusion, à tester d'autres architectures et à optimiser les hyperparamètres.

## 9. Évaluation sur le jeu de test

Le modèle CNN amélioré a été évalué sur un ensemble de test indépendant contenant 7 178 images.

### Résultats globaux

| Métrique | Résultat |
|---|---:|
| Accuracy | 56,59 % |
| Loss | 1,1883 |
| Macro F1-score | 0,49 |
| F1-score pondéré | 0,55 |

L'accuracy mesure la proportion totale de prédictions correctes. Le macro F1-score calcule la moyenne des F1-scores des classes en donnant le même poids à chaque émotion. Le F1-score pondéré tient compte du nombre d'images de chaque classe.

### Analyse par classe

La classe `happy` obtient les meilleurs résultats, avec un F1-score de 0,80 et un recall de 83 %. La classe `surprise` est également bien reconnue, avec un F1-score de 0,69.

En revanche, la classe `disgust` présente un recall de seulement 11 %. Le modèle identifie correctement très peu d'images de cette émotion. La classe `fear` présente également des difficultés, avec un recall de 25 %.

Ces résultats montrent que les performances varient fortement selon les émotions. Le déséquilibre des classes et la similarité visuelle entre certaines expressions peuvent expliquer une partie de ces difficultés.

### Matrice de confusion

La matrice de confusion permet d'observer les émotions réelles en lignes et les émotions prédites en colonnes. Les valeurs de la diagonale correspondent aux prédictions correctes.

Elle montre notamment des confusions entre `sad` et `neutral`, ainsi qu'entre `fear` et plusieurs autres émotions. Ces erreurs expliquent en partie les performances limitées de certaines classes.

## 10. Conclusion

Le CNN amélioré atteint une accuracy de 56,59 % sur le jeu de test. L'utilisation de l'augmentation des données et de techniques de régularisation permet d'obtenir des performances de validation légèrement meilleures que celles du CNN simple.

Cependant, les résultats restent inégaux entre les classes. Le modèle reconnaît mieux les émotions `happy` et `surprise`, tandis que `disgust` et `fear` restent difficiles à identifier.

Pour améliorer le modèle, plusieurs pistes peuvent être explorées : utiliser des poids de classes, augmenter les données des émotions minoritaires, analyser les erreurs de classification et tester des architectures plus adaptées à la reconnaissance des expressions faciales.


11. Exécution

1. Installer les dépendances nécessaires.
2. Télécharger et organiser le dataset FER-2013.
3. Ouvrir le notebook Jupyter.
4. Exécuter les cellules dans l'ordre.
5. Comparer les courbes d'apprentissage et les résultats des modèles.
