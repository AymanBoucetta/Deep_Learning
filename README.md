# Deep Learning

Ce dépôt rassemble des exercices de deep learning et de traitement d’images, organisés par chapitre.

## Contenu

- `Chapitre3/` : exercices Python autour d’un réseau dense entraîné sur MNIST (préparation des données, création du modèle et expériences sur l’entraînement).
- `Chapitre4/CNN.ipynb` : notebook d’introduction aux filtres de convolution, avec l’image `Chapitre4/noir_et_blanc.jpg` utilisée comme entrée.
- `chapitre5/Tp5 atelier CNN local.ipynb` : TP de classification des images CIFAR-10 avec un CNN entraîné sur un jeu de données local. Il couvre le chargement et la normalisation des images, l’augmentation des données, l’entraînement avec arrêt anticipé, l’évaluation et l’analyse des erreurs (matrice de confusion et exemples mal classés).
- `src/deeplearning/` : point d’entrée minimal du paquet Python.

## Prérequis et installation

Python 3.11 ou une version ultérieure et [uv](https://docs.astral.sh/uv/) sont nécessaires.

Depuis la racine du dépôt, synchronisez l’environnement et les dépendances :

```bash
uv sync --all-groups
```

## Exécution

Les scripts du chapitre 3 peuvent être lancés depuis la racine du dépôt :

```bash
uv run python Chapitre3/ex1.py
uv run python Chapitre3/ex2.py
uv run python Chapitre3/ex3.py
uv run python Chapitre3/ex4.py
```

Ouvrez `Chapitre4/CNN.ipynb` dans VS Code avec l’extension Jupyter et sélectionnez l’environnement `.venv` comme noyau. Pour que le chargement de l’image fonctionne tel quel, exécutez le notebook avec `Chapitre4/` comme répertoire courant.

Pour le chapitre 5, le notebook attend les images CIFAR-10 réparties par classe dans `cifar10/cifar10/train/` et `cifar10/cifar10/test/` (par exemple `train/airplane/`). Ouvrez `chapitre5/Tp5 atelier CNN local.ipynb` avec Jupyter. Le chemin du jeu de données est actuellement défini en dur dans le notebook (`DATA_DIR`) et doit être adapté à votre installation avant l’exécution.

## Dépendances principales

Les exercices utilisent TensorFlow/Keras, NumPy, Matplotlib, OpenCV et scikit-learn. Les versions sont déclarées dans `pyproject.toml` et verrouillées dans `uv.lock`.
