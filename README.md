# Deep Learning

Ce dépôt rassemble des exercices de deep learning et de traitement d’images, organisés par chapitre.

## Contenu

- `Chapitre3/` : exercices Python autour d’un réseau dense entraîné sur MNIST (préparation des données, création du modèle et expériences sur l’entraînement).
- `Chapitre4/CNN.ipynb` : notebook d’introduction aux filtres de convolution, avec l’image `Chapitre4/noir_et_blanc.jpg` utilisée comme entrée.
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

## Dépendances principales

Les exercices utilisent TensorFlow/Keras, NumPy, Matplotlib et OpenCV. Les versions sont déclarées dans `pyproject.toml` et verrouillées dans `uv.lock`.
