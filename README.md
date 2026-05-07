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


## 3. Script 1 : build_index.py

Fonction=> clean_and_extract_sentences(folder_path)

Entrées (Arguments) => folder_path (str) - Chemin vers le dossier contenant les fichiers .txt.

Sorties (Return) => List[str] - Une liste contenant toutes les phrases extraites, nettoyées et taguées.

Fonctionnalité : Parcourt les fichiers, utilise des expressions régulières (Regex) pour supprimer les balises XML (<doc>, <p>) s'il reste dans les documents .txt, utilise nltk.sent_tokenize pour isoler les phrases, supprime les fragments trop courts, et ajoute un tag de provenance ([LE MONDE] ou [GUARDIAN]). Il est utile d'ajouter les tags si vous avez plusier corpus.


## 4. Script 2 : search.py (Moteur de recherche, Search Engine)

Fonction => display_kwic(result_indices, distances, all_sentences, window=1)

Entrées => result_indices (Tableau 2D d'entiers Faiss), distances (Tableau 2D de flottants Faiss), all_sentences (List[str] du corpus entier), window (int, par défaut 1).

Sorties => Affichage standard (Console/Stdout).

Fonctionnalité => Affiche les résultats sous forme de KWIC (Key Word In Context). Récupère l'index du résultat exact et affiche les n phrases précédentes et suivantes pour fournir un contexte linguistique à l'utilisateur.

## 5. Tests et Bogues identifiés

Bogue/Limitation (Gestion de la RAM) => L'encodage de millions de phrases nécessite une grande quantité de mémoire vive. Pour éviter les "Out of Memory", le paramètre batch_size a été réglé sur 64.

Limitation NLTK =. La tokénisation par NLTK peut parfois être imparfaite sur des textes bruts de presse contenant des abréviations complexes non standard.


## 6.  L'Algorithmique

#### A. Évolution des Structures de Données (Data Pipeline)
1.  **Données brutes :** Fichiers textes lus sous forme de longues chaînes de caractères (`str`).
2.  **Données après tokénisation :** Création d'une structure `List[str]`. C'est une liste Python native contenant chaque phrase séparément. Cette structure est sauvegardée sur le disque dur grâce au module `pickle` (`.pkl`).
3.  **Sorties du Modèle (Embeddings) :** Le modèle `paraphrase-multilingual-MiniLM-L12-v2` transforme la `List[str]` en tenseurs mathématiques. Ils sont convertis en tableaux NumPy (`numpy.ndarray`).
    * **Propriétés de la matrice :** C'est une matrice de dimension `(N, 384)`, où `N` est le nombre total de phrases et `384` représente les dimensions du vecteur d'embedding. 
4.  **Index de Recherche :** La matrice NumPy est insérée dans un objet `faiss.IndexFlatIP`. C'est une structure de données optimisée en C++ qui stocke les vecteurs en mémoire de manière contiguë pour des calculs ultra-rapides.

#### B. Évaluation de la Complexité Algorithmique
* **Complexité de l'encodage (Phase de construction) :** L'inférence d'un réseau de neurones Transformer (MiniLM) est très lourde. La complexité en temps est $O(N \times L^2)$ où $N$ est le nombre de phrases et $L$ la longueur de la phrase (à cause du mécanisme de d'attention (Self-Attention)). C'est pour cela que cette phase a pris plusieurs heures.
* **Complexité spatiale (Mémoire) :** Stocker $N$ vecteurs de dimension $d=384$ en `float32` (4 octets par nombre) donne une complexité spatiale $O(N \times d \times 4)$. Pour ~3 millions de phrases, cela représente environ 4,6 Go de RAM.
* **Complexité de la recherche (Phase d'utilisation) :** * Pour calculer la similarité cosinus (Cosine Similarity), nous avons normalisé les vecteurs (L2 Norm) lors de l'encodage.
    * L'algorithme `IndexFlatIP` de Faiss effectue une recherche exacte par force brute calculant le produit scalaire (Inner Product).
    * La complexité temporelle de la recherche est donc **$O(N \times d)$**, où $N$ est la taille du corpus et $d$ la dimension (384). Bien que linéaire, l'implémentation C++ de Faiss permet d'exécuter ce calcul sur des millions d'entrées en quelques millisecondes.


