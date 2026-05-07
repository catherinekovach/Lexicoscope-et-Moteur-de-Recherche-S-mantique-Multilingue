# Lexicoscope-et-Moteur-de-Recherche-S-mantique-Multilingue
Il s'agit d'un moteur de recherche sémantique multilingue. Le modèle actuel prend en charge l'anglais et le français et s'appuie sur un corpus issu du Guardian et du Monde.

## 1. Mini-Cahier des Charges
**Objectif du projet :** Concevoir et développer un concordancier (lexicoscope) basé sur la recherche sémantique cross-lingue. Contrairement à une recherche par mots-clés classique, cet outil utilise des modèles d'intelligence artificielle (Sentence Embeddings) pour trouver des phrases partageant le même sens, indépendamment du vocabulaire utilisé ou de la langue (anglais/français).

Le fichier build_index.py regroupe tous les fichiers .txt du corpus et les indexe. C'est l'étape la plus longue : l'indexation de deux corpus d'environ 260 Mo a pris environ 6 heures. 

Une fois l'index créé, les deux fichiers générés peuvent être intégrés au fichier search.py. 

**Corpus utilisés :** Le Monde (français) et The Guardian 2017 (anglais).

## 2. Dépendances et Notes d'Installation
Pour exécuter ce projet localement, les bibliothèques suivantes sont requises.

```bash
pip install sentence-transformers faiss-cpu nltk numpy
```

Sites web qui m'ont aidé :

https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

https://github.com/facebookresearch/faiss/wiki/Faiss-indexes





