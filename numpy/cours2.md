# Cours 1 : Introduction à NumPy et Manipulation de Tableaux

## Table des matières
1. [Introduction à NumPy](#introduction)
2. [Types de tableaux NumPy](#types-tableaux)
3. [Constructeurs de tableaux](#constructeurs)
4. [Manipulation de tableaux](#manipulation)
5. [Slicing et Indexing](#slicing)
6. [Exercices](#exercices)
7. [Corrigés](#corriges)

---

## 1. Introduction à NumPy {#introduction}

### Qu'est-ce que NumPy ?

**NumPy** (Numerical Python) est une bibliothèque Python fondamentale pour le calcul scientifique. Elle a été développée pour faciliter la vie des ingénieurs en Machine Learning et des Data Scientists lors de calculs complexes.

**Structure principale** : Le **NumPy array** (tableau NumPy) - une structure de données optimisée pour les opérations mathématiques.

### Pourquoi NumPy en Machine Learning ?

```python
import numpy as np

# Installation si nécessaire
# pip install numpy
```

### Application concrète : Normalisation de données

En Machine Learning, on doit souvent normaliser nos données avant de les donner à un modèle :

```python
# Données brutes : notes d'étudiants
notes = np.array([12, 15, 8, 18, 14, 11, 16, 9])

# Normalisation (mise à l'échelle entre 0 et 1)
notes_min = notes.min()
notes_max = notes.max()
notes_normalisees = (notes - notes_min) / (notes_max - notes_min)

print("Notes originales:", notes)
print("Notes normalisées:", notes_normalisees)
```

**Avantage** : NumPy effectue ces opérations sur des millions de données en quelques millisecondes !

---

## 2. Types de tableaux NumPy {#types-tableaux}

### Tableaux 1D (Vecteurs)

```python
# Un vecteur : liste de valeurs
vecteur = np.array([1, 2, 3, 4, 5])
print("1D:", vecteur)
print("Shape:", vecteur.shape)  # (5,)
```

**Usage ML** : Représenter les caractéristiques d'un seul échantillon (ex: température, humidité, pression)

### Tableaux 2D (Matrices) - LE PLUS UTILISÉ

```python
# Une matrice : tableau de tableaux
matrice = np.array([[1, 2, 3],
                    [4, 5, 6]])
print("2D:", matrice)
print("Shape:", matrice.shape)  # (2, 3) = 2 lignes, 3 colonnes
```

**Usages ML courants** :
- 📊 **Dataset** : Chaque ligne = un échantillon, chaque colonne = une caractéristique
- 🖼️ **Image en noir et blanc** : Chaque valeur = intensité d'un pixel
- 📈 **Feuille Excel/CSV** : Données tabulaires

### Tableaux 3D (Tenseurs)

```python
# Un tenseur : tableau de matrices
tenseur = np.array([[[1, 2], [3, 4]],
                    [[5, 6], [7, 8]]])
print("3D:", tenseur)
print("Shape:", tenseur.shape)  # (2, 2, 2)
```

**Usage ML** : 
- 🎨 **Images couleur** : (hauteur, largeur, 3 canaux RGB)
- 🎬 **Vidéos** : (temps, hauteur, largeur, canaux)

---

## 3. Constructeurs de tableaux {#constructeurs}

### np.array() - Création depuis une liste

```python
# À partir d'une liste Python
arr = np.array([1, 2, 3, 4, 5])
```

### np.zeros() - Tableau de zéros

```python
# Initialiser avec des zéros
zeros = np.zeros((3, 4))  # 3 lignes, 4 colonnes
print(zeros)
```

**Usage ML** : Initialiser des matrices de poids ou de résultats

### np.ones() - Tableau de uns

```python
# Initialiser avec des uns
ones = np.ones((2, 3))
print(ones)
```

### np.random.randn() - Valeurs aléatoires (distribution normale)

```python
# Génère des valeurs selon une distribution normale centrée en 0
random_arr = np.random.randn(3, 3)
print(random_arr)
```

#### Pourquoi une distribution normale centrée en 0 ?

**Raisons en Machine Learning** :

1. **Symétrie** : Valeurs équilibrées entre positif et négatif
2. **Convergence** : Les algorithmes d'optimisation (descente de gradient) convergent plus vite
3. **Éviter la saturation** : Avec des fonctions d'activation comme sigmoid ou tanh, des valeurs trop grandes peuvent bloquer l'apprentissage
4. **Propriétés mathématiques** : La distribution normale est bien comprise et prévisible

```python
# Exemple : Initialisation des poids d'un réseau de neurones
poids = np.random.randn(100, 50) * 0.01  # Petites valeurs autour de 0
```

### Autres constructeurs utiles

```python
# Séquence de nombres
seq = np.arange(0, 10, 2)  # De 0 à 10, pas de 2 : [0, 2, 4, 6, 8]

# Valeurs espacées linéairement
lin = np.linspace(0, 1, 5)  # 5 valeurs entre 0 et 1 : [0, 0.25, 0.5, 0.75, 1]

# Matrice identité
identite = np.eye(3)  # Matrice 3x3 avec des 1 sur la diagonale
```

---

## 4. Manipulation de tableaux {#manipulation}

### np.shape - Voir les dimensions

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3) : 2 lignes, 3 colonnes

# Aussi accessible comme attribut
print(arr.shape)
```

### np.reshape() - Changer la forme

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print("Original:", arr.shape)  # (6,)

# Transformer en matrice 2x3
reshaped = arr.reshape(2, 3)
print("Reshaped:", reshaped.shape)  # (2, 3)
print(reshaped)
```

**⚠️ RÈGLE IMPORTANTE** : Le nombre total d'éléments doit rester constant !
- ✅ (6,) → (2, 3) : 6 éléments = 2×3 ✓
- ❌ (6,) → (2, 4) : 6 éléments ≠ 2×4 = 8 ✗

```python
# Utiliser -1 pour calculer automatiquement une dimension
arr.reshape(3, -1)  # NumPy calcule : (3, 2)
arr.reshape(-1, 1)  # Colonne : (6, 1)
```

### np.concatenate() - Assembler des tableaux

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Axe 0 : empiler verticalement (lignes)
vertical = np.concatenate([arr1, arr2], axis=0)
print("Vertical (axis=0):\n", vertical)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

# Axe 1 : empiler horizontalement (colonnes)
horizontal = np.concatenate([arr1, arr2], axis=1)
print("Horizontal (axis=1):\n", horizontal)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]
```

**Mnémotechnique** : 
- `axis=0` : ↓ (ajouter des lignes)
- `axis=1` : → (ajouter des colonnes)

### np.ravel() - Aplatir en 1D

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
flat = arr.ravel()
print(flat)  # [1 2 3 4 5 6]
```

**Usage ML** : Transformer une image 2D en vecteur pour certains algorithmes

### np.squeeze() - Supprimer les dimensions de taille 1

```python
arr = np.array([[[1], [2], [3]]])
print("Original:", arr.shape)  # (1, 3, 1)

squeezed = np.squeeze(arr)
print("Squeezed:", squeezed.shape)  # (3,)
print(squeezed)  # [1 2 3]
```

**Usage ML** : Nettoyer les dimensions inutiles après des opérations

---

## 5. Slicing et Indexing {#slicing}

### Indexing - Accéder à des éléments

```python
# Tableau 1D
arr = np.array([10, 20, 30, 40, 50])
print(arr[0])    # 10 (premier élément)
print(arr[-1])   # 50 (dernier élément)
print(arr[2])    # 30 (troisième élément)

# Tableau 2D
mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(mat[0, 0])   # 1 (ligne 0, colonne 0)
print(mat[1, 2])   # 6 (ligne 1, colonne 2)
print(mat[-1, -1]) # 9 (dernière ligne, dernière colonne)
```

### Slicing - Extraire des portions

**Syntaxe** : `[début:fin:pas]`

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(arr[2:7])      # [2 3 4 5 6] : de l'index 2 à 6
print(arr[:5])       # [0 1 2 3 4] : du début à 4
print(arr[5:])       # [5 6 7 8 9] : de 5 à la fin
print(arr[::2])      # [0 2 4 6 8] : tous les 2 éléments
print(arr[::-1])     # [9 8 7...0] : inverser le tableau
```

### Slicing 2D

```python
mat = np.array([[1,  2,  3,  4],
                [5,  6,  7,  8],
                [9,  10, 11, 12]])

# Extraire une ligne
print(mat[1, :])     # [5 6 7 8] : toute la ligne 1

# Extraire une colonne
print(mat[:, 2])     # [3 7 11] : toute la colonne 2

# Sous-matrice
print(mat[0:2, 1:3]) # [[2 3]
                     #  [6 7]] : lignes 0-1, colonnes 1-2

# Lignes alternées
print(mat[::2, :])   # Lignes 0 et 2
```

### Indexing booléen (masquage)

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Créer un masque
masque = arr > 3
print(masque)        # [False False False True True True]

# Appliquer le masque
print(arr[masque])   # [4 5 6]

# En une ligne
print(arr[arr > 3])  # [4 5 6]

# Usage ML : Filtrer des données
temperatures = np.array([20, 25, 30, 35, 40])
jours_chauds = temperatures[temperatures > 30]
print(jours_chauds)  # [35 40]
```

### Fancy Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Sélectionner plusieurs indices
indices = [0, 2, 4]
print(arr[indices])  # [10 30 50]

# Pour les matrices
mat = np.array([[1, 2], [3, 4], [5, 6]])
print(mat[[0, 2]])   # Lignes 0 et 2
                     # [[1 2]
                     #  [5 6]]
```

---

## 6. Exercices {#exercices}

### Exercice 1 : Manipulation de tableaux (Machine Learning)

**Contexte** : Vous avez collecté des données de température (°C) de 5 villes sur 7 jours.

```python
temperatures = np.array([
    [22, 24, 23, 25, 26, 24, 23],  # Ville 1
    [18, 19, 20, 19, 21, 20, 19],  # Ville 2
    [30, 31, 29, 32, 33, 31, 30],  # Ville 3
    [15, 16, 15, 17, 18, 16, 15],  # Ville 4
    [25, 26, 27, 26, 28, 27, 26]   # Ville 5
])
```

**Questions** :

1. Quelle est la forme (shape) de ce tableau ? Qu'est-ce que cela représente ?

2. Créez un tableau `moyennes` contenant la température moyenne de chaque ville sur les 7 jours.

3. Quel jour a été le plus chaud pour la ville 3 ? (utilisez `np.argmax()`)

4. Créez un tableau `temp_week` contenant uniquement les températures des jours 2 à 5 (indices 1 à 4) pour toutes les villes.

5. Normalisez le tableau `temperatures` pour que toutes les valeurs soient entre 0 et 1.
   Formule : `(X - X_min) / (X_max - X_min)`

6. Créez un masque booléen pour identifier tous les jours où la température a dépassé 25°C.

7. Ajoutez une nouvelle ville (Ville 6) avec les températures : `[20, 21, 22, 21, 23, 22, 21]` au tableau existant.

8. Transformez le tableau des 5 premières villes en un vecteur 1D (35 éléments) puis remettez-le en forme de matrice 7x5 (jours en lignes, villes en colonnes).

---

### Exercice 2 : Slicing et Indexing avancé

Soit le tableau suivant représentant les notes de 6 étudiants pour 4 matières :

```python
notes = np.array([
    [15, 12, 14, 16],  # Étudiant 1
    [10, 11, 9,  12],  # Étudiant 2
    [18, 17, 19, 18],  # Étudiant 3
    [8,  9,  7,  10],  # Étudiant 4
    [14, 15, 13, 16],  # Étudiant 5
    [12, 13, 11, 14]   # Étudiant 6
])
```

**Questions** :

1. Extrayez les notes de l'étudiant 3 (toutes les matières).

2. Extrayez les notes de la matière 2 (colonne d'indice 1) pour tous les étudiants.

3. Créez un sous-tableau contenant les notes des étudiants 2 à 4 pour les matières 1 à 3.

4. Affichez les notes des étudiants pairs (indices 0, 2, 4) uniquement.

5. Créez un masque pour identifier tous les étudiants ayant au moins une note >= 15.

6. Extrayez toutes les notes supérieures à 12.

7. Remplacez toutes les notes inférieures à 10 par 10 (note plancher).

8. Calculez la moyenne de chaque étudiant et identifiez l'étudiant avec la meilleure moyenne.

9. Triez les étudiants par ordre décroissant de leur moyenne (utilisez `np.argsort()`).

10. Créez un tableau de mentions : 
    - "TB" (Très Bien) pour moyenne >= 16
    - "B" (Bien) pour moyenne >= 14
    - "AB" (Assez Bien) pour moyenne >= 12
    - "P" (Passable) pour le reste

---

## 7. Corrigés {#corriges}

### Corrigé Exercice 1

```python
import numpy as np

temperatures = np.array([
    [22, 24, 23, 25, 26, 24, 23],
    [18, 19, 20, 19, 21, 20, 19],
    [30, 31, 29, 32, 33, 31, 30],
    [15, 16, 15, 17, 18, 16, 15],
    [25, 26, 27, 26, 28, 27, 26]
])

# 1. Shape
print("1. Shape:", temperatures.shape)
print("   Signification: 5 villes (lignes), 7 jours (colonnes)")

# 2. Températures moyennes par ville
moyennes = temperatures.mean(axis=1)  # Moyenne sur l'axe des colonnes
print("\n2. Moyennes par ville:", moyennes)
# Alternative : np.mean(temperatures, axis=1)

# 3. Jour le plus chaud pour ville 3
jour_max_ville3 = np.argmax(temperatures[2])
temp_max_ville3 = temperatures[2, jour_max_ville3]
print(f"\n3. Ville 3: jour {jour_max_ville3} le plus chaud avec {temp_max_ville3}°C")

# 4. Températures jours 2 à 5
temp_week = temperatures[:, 1:5]
print("\n4. Températures jours 2-5:\n", temp_week)

# 5. Normalisation
temp_min = temperatures.min()
temp_max = temperatures.max()
temp_normalisees = (temperatures - temp_min) / (temp_max - temp_min)
print("\n5. Températures normalisées:\n", temp_normalisees)

# 6. Masque > 25°C
masque_chaud = temperatures > 25
print("\n6. Jours > 25°C:\n", masque_chaud)
print("   Nombre de mesures > 25°C:", np.sum(masque_chaud))

# 7. Ajouter ville 6
ville6 = np.array([[20, 21, 22, 21, 23, 22, 21]])
temperatures_extended = np.concatenate([temperatures, ville6], axis=0)
print("\n7. Avec ville 6:", temperatures_extended.shape)

# 8. Transformation
temp_flat = temperatures.ravel()
print("\n8a. Vecteur 1D:", temp_flat.shape)

temp_transposed = temp_flat.reshape(7, 5)
print("8b. Matrice 7x5 (jours x villes):\n", temp_transposed)
```

**Résultats attendus** :
```
1. Shape: (5, 7)
   Signification: 5 villes (lignes), 7 jours (colonnes)

2. Moyennes par ville: [23.86 19.43 30.86 15.86 26.43]

3. Ville 3: jour 4 le plus chaud avec 33°C

...
```

---

### Corrigé Exercice 2

```python
import numpy as np

notes = np.array([
    [15, 12, 14, 16],
    [10, 11, 9,  12],
    [18, 17, 19, 18],
    [8,  9,  7,  10],
    [14, 15, 13, 16],
    [12, 13, 11, 14]
])

# 1. Notes étudiant 3
print("1. Étudiant 3:", notes[2])

# 2. Notes matière 2
print("\n2. Matière 2:", notes[:, 1])

# 3. Sous-tableau étudiants 2-4, matières 1-3
sous_tableau = notes[1:4, 0:3]
print("\n3. Sous-tableau:\n", sous_tableau)

# 4. Étudiants pairs
etudiants_pairs = notes[::2]
print("\n4. Étudiants pairs:\n", etudiants_pairs)

# 5. Étudiants avec note >= 15
masque_15 = np.any(notes >= 15, axis=1)
print("\n5. Étudiants avec note >= 15:", masque_15)
print("   Indices:", np.where(masque_15)[0])

# 6. Notes > 12
notes_sup_12 = notes[notes > 12]
print("\n6. Notes > 12:", notes_sup_12)

# 7. Plancher à 10
notes_corrigees = notes.copy()
notes_corrigees[notes_corrigees < 10] = 10
print("\n7. Notes avec plancher:\n", notes_corrigees)

# 8. Meilleur étudiant
moyennes_etudiants = notes.mean(axis=1)
meilleur_idx = np.argmax(moyennes_etudiants)
print(f"\n8. Meilleur étudiant: {meilleur_idx} avec {moyennes_etudiants[meilleur_idx]:.2f}")
print("   Toutes les moyennes:", moyennes_etudiants)

# 9. Tri par moyenne décroissante
ordre_decroissant = np.argsort(moyennes_etudiants)[::-1]
print("\n9. Classement (indices étudiants):", ordre_decroissant)
print("   Moyennes triées:", moyennes_etudiants[ordre_decroissant])

# 10. Mentions
mentions = []
for moyenne in moyennes_etudiants:
    if moyenne >= 16:
        mentions.append("TB")
    elif moyenne >= 14:
        mentions.append("B")
    elif moyenne >= 12:
        mentions.append("AB")
    else:
        mentions.append("P")

print("\n10. Mentions par étudiant:", mentions)

# Bonus : Affichage avec noms
for i, (moy, mention) in enumerate(zip(moyennes_etudiants, mentions)):
    print(f"    Étudiant {i+1}: {moy:.2f}/20 → {mention}")
```

**Résultats attendus** :
```
1. Étudiant 3: [18 17 19 18]

2. Matière 2: [12 11 17  9 15 13]

8. Meilleur étudiant: 2 avec 18.00
   Toutes les moyennes: [14.25 10.5  18.    8.5  14.5  12.5 ]

10. Mentions par étudiant: ['B', 'P', 'TB', 'P', 'B', 'AB']
```

---

## Ressources complémentaires

- 📚 Documentation NumPy : https://numpy.org/doc/
- 🎥 Tutoriels vidéo recommandés
- 💡 Pour aller plus loin : Broadcasting, algèbre linéaire avec NumPy

---

## À venir dans le prochain cours

- 📊 Statistiques avec NumPy (mean, std, var, percentiles)
- 🔢 Algèbre linéaire (produit matriciel, déterminant, inverse)
- 🔄 Broadcasting (opérations sur tableaux de tailles différentes)

**Bon apprentissage ! 🚀**
