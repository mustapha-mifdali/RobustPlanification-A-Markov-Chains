# 🗺️ Planification Robuste sur Grille 2D
## A\* · UCS · Greedy + Chaînes de Markov · Monte-Carlo

> **Comparaison d'algorithmes de recherche heuristique et analyse de robustesse stochastique sur grilles 2D avec obstacles.**

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Résultats clés](#résultats-clés)
3. [Installation](#installation)
4. [Utilisation](#utilisation)
5. [Structure du projet](#structure-du-projet)
6. [Description des modules](#description-des-modules)
7. [Fondements théoriques](#fondements-théoriques)
8. [Visualisations générées](#visualisations-générées)
9. [API de référence](#api-de-référence)
10. [Références](#références)

---

## Vue d'ensemble

Ce projet implémente et compare trois algorithmes de recherche sur grille 2D :

| Algorithme | Fonction f(n) | Optimal ? | Complet ? |
|-----------|--------------|-----------|-----------|
| **A\***   | g(n) + h(n)  | ✅ Oui    | ✅ Oui    |
| **UCS**   | g(n)         | ✅ Oui    | ✅ Oui    |
| **Greedy**| h(n)         | ❌ Non    | ✅ Oui    |

L'heuristique utilisée est la **distance de Manhattan** — admissible et cohérente pour un déplacement 4-connexe (Nord / Sud / Est / Ouest).

Une fois les chemins trouvés, leur **robustesse sous incertitude d'exécution** est analysée via :
- **Chaînes de Markov** : modèle analytique exact avec matrice de transition P
- **Simulation Monte-Carlo** : validation empirique par N trajectoires stochastiques

---

## Résultats clés

### Performances des algorithmes (grille 20×20, 50 obstacles)

| Algorithme | Coût chemin | Nœuds explorés | Temps (ms) | Efficacité vs UCS |
|-----------|------------|----------------|------------|-------------------|
| **A\***   | **38**     | **281**        | 0.730      | **1.25×**         |
| UCS       | 38         | 350            | 0.866      | 1.00× (baseline)  |
| Greedy    | ⚠️ 42      | **43**         | 0.129      | 8.14×             |

> ⚠️ Greedy trouve un chemin **sous-optimal** (+10.5%) sur la grille difficile.

### Robustesse stochastique (chemin A\* = 28 étapes, grille 15×15)

| ε (incertitude) | P(atteindre GOAL) | Interprétation |
|----------------|-------------------|----------------|
| ε = 0.00 | **100.0%** | Exécution parfaite |
| ε = 0.05 | **99.9%**  | Bruit négligeable |
| ε = 0.10 | **96.4%**  | Seuil opérationnel |
| ε = 0.15 | **80.0%**  | Dégradation notable |
| ε = 0.20 | **52.0%**  | Replanification recommandée |
| ε = 0.30 | **9.8%**   | Plan peu fiable |

> 📌 Les chemins **plus courts** (A\*/UCS optimaux) sont toujours **plus robustes** que les chemins plus longs (Greedy).

---

## Installation

```bash
pip install matplotlib numpy
```

**Prérequis :** Python 3.7+, numpy, matplotlib. Aucune autre dépendance.

---

## Utilisation

### ▶️ Pipeline complet — 3 grilles (Easy / Medium / Hard)

```bash
python run_complete_analysis.py
```

Génère tout dans `analysis_results/` :

```
analysis_results/
├── easy/        → 9 figures PNG
├── medium/      → 9 figures PNG
├── hard/        → 9 figures PNG
├── cross_difficulty_comparison.png
└── ANALYSIS_SUMMARY.txt
```

---

### ▶️ Visualisation grille uniquement

```bash
python grid_visualizer.py
```

Génère 6 figures dans `grid_results/` en ~30 secondes.

---

### ▶️ Analyse Markov uniquement

```bash
python markov_analysis.py
```

Génère 3 figures dans `markov_results/`.

---

### ▶️ Test rapide des algorithmes

```bash
python astar.py
```

Affiche un tableau comparatif A\* / UCS / Greedy + itinéraire détaillé sur une grille 10×10.

---

### ▶️ Test du modèle Markov

```bash
python markov.py
```

Affiche la matrice P, l'évolution de la distribution, P(GOAL) vs ε et les résultats Monte-Carlo.

---

## Structure du projet

```
miniproject/
│
├── run_complete_analysis.py     ← Pipeline principal (lance tout)
│
├── astar.py                     ← Algorithmes A*, UCS, Greedy
├── markov.py                    ← Modèle chaînes de Markov + Monte-Carlo
├── grid_visualizer.py           ← Visualisations grille + métriques
├── markov_analysis.py           ← Analyse Markov détaillée
│
├── README.md                    ← Ce fichier
└── Rapport_Planification_Robuste.docx  ← Rapport technique complet (français)
```

---

## Description des modules

### `astar.py` — Moteur de recherche

Implémente un **moteur de recherche unifié** pour A\*, UCS et Greedy via une file de priorité (tas min). Un seul moteur gère les trois algorithmes — seule la fonction de priorité f(n) change.

```python
from astar import run_all, run_astar, build_itinerary

# Lancer les 3 algorithmes d'un coup
obstacles = {(2,3), (3,3), (4,3), (4,4)}
results = run_all((0,0), (9,9), 10, obstacles)

for algo in ['astar', 'ucs', 'greedy']:
    r = results[algo]
    print(f"{algo}: coût={r['cost']}  nœuds={r['nodes']}  temps={r['time_ms']}ms")

# Itinéraire détaillé étape par étape
itin = build_itinerary(results['astar']['path'], goal=(9,9))
for step in itin:
    print(f"#{step['step']:2d}  {step['cell']}  g={step['g']} h={step['h']} f={step['f']}  {step['dir']}")
```

**Structure du résultat retourné :**

| Clé | Type | Description |
|-----|------|-------------|
| `path` | `list` | Liste ordonnée de cellules `(row, col)` |
| `explored` | `list` | Cellules développées durant la recherche |
| `cost` | `int` | Coût total du chemin trouvé |
| `nodes` | `int` | Nombre de nœuds développés |
| `time_ms` | `float` | Temps d'exécution en millisecondes |

---

### `markov.py` — Modèle stochastique

Construit la matrice de transition **P** et calcule la probabilité d'atteindre le but sous incertitude d'exécution.

```python
from markov import (build_transition_matrix, verify_stochastic,
                    goal_probability_vs_epsilon, monte_carlo, absorption_analysis)

path = results['astar']['path']

# Matrice de transition (représentation creuse)
P = build_transition_matrix(path, grid_n=10, obstacles=obstacles, epsilon=0.10)
print(f"États : {len(P)}  |  Stochastique : {verify_stochastic(P)}")

# P(GOAL) pour plusieurs valeurs d'epsilon (analytique)
probs = goal_probability_vs_epsilon(path, [0.0, 0.10, 0.20, 0.30])
for eps, p in probs.items():
    print(f"ε={eps:.2f}  →  P(GOAL) = {p*100:.1f}%")

# Simulation Monte-Carlo (validation empirique)
mc = monte_carlo(path, grid_n=10, obstacles=obstacles, epsilon=0.10, n_simulations=500)
print(f"P̂(GOAL) = {mc['prob_goal']*100:.1f}%  moy={mc['avg_steps']} pas  σ={mc['std_steps']}")

# Analyse d'absorption
ab = absorption_analysis(path, epsilon=0.10)
print(f"P(GOAL)={ab['prob_goal']}  P(FAIL)={ab['prob_fail']}  E[T]={ab['expected_steps']} pas")
```

**Modèle de transition avec paramètre ε :**

```
P(s'|s, a) = (1 - ε)   si s' = direction prévue
           = ε / 2     si s' = déviation latérale gauche ou droite
           = 0         sinon
```

En cas de collision (obstacle ou bord), l'agent **reste en place**.

---

### `grid_visualizer.py` — Visualisations

Génère toutes les figures comparatives pour les algorithmes sur une grille donnée.

```python
from grid_visualizer import generate_all_visualizations

# Génère 6 figures dans grid_results/
results = generate_all_visualizations(grid_n=15, obstacle_percentage=20)
```

**Figures générées :**

| Fichier | Description |
|---------|-------------|
| `01_grid_comparison.png` | Grille + chemins A\*, UCS, Greedy côte à côte |
| `02_metrics_comparison.png` | Barres : nœuds, coût, temps, efficacité |
| `03_uncertainty_analysis.png` | P(GOAL) vs ε : table colorée + courbe |
| `04_itinerary_tables.png` | Itinéraire étape par étape (g, h, f, direction) |
| `05_uncertainty_table.png` | Tableau Markov vs Monte-Carlo détaillé par algo |
| `06_stochastic_matrix.png` | Heatmap P + vérification sommes lignes + valeurs P[i→j] |

---

### `markov_analysis.py` — Analyse Markov avancée

```python
from markov_analysis import generate_markov_analysis
from astar import run_all

results = run_all((0,0), (14,14), 15, obstacles)
generate_markov_analysis(results, grid_n=15, obstacles=obstacles, save_dir="markov_results")
```

**Figures générées :**

| Fichier | Description |
|---------|-------------|
| `01_transition_matrix.png` | Heatmap de P + graphe de connectivité des états |
| `02_markov_vs_montecarlo.png` | Comparaison analytique vs empirique + classement robustesse |
| `03_absorption_analysis.png` | P(GOAL) / P(FAIL) + longueur chemin vs E[T] |

---

### `run_complete_analysis.py` — Pipeline principal

Lance automatiquement l'analyse complète sur 3 grilles de difficulté croissante.

```python
import random
from run_complete_analysis import AnalysisPipeline

random.seed(42)
pipeline = AnalysisPipeline(base_dir="analysis_results")
pipeline.run_complete_pipeline()
```

**Étapes du pipeline :**

1. **Génération des grilles** — Easy 10×10, Medium 15×15, Hard 20×20
2. **Exécution des algorithmes** — A\*, UCS, Greedy sur chaque grille
3. **Visualisations grille** — 6 figures par niveau de difficulté
4. **Analyse Markov** — matrice P, Markov vs MC, absorption
5. **Comparaison croisée** — évolution des métriques Easy→Medium→Hard
6. **Rapport texte** — `ANALYSIS_SUMMARY.txt`

---

## Fondements théoriques

### Algorithme A\*

```
f(n) = g(n) + h(n)

  g(n) : coût réel depuis le départ jusqu'à n
  h(n) : estimation heuristique de n jusqu'au but (Manhattan)
  f(n) : priorité dans la file — plus f est petit, plus n est prioritaire
```

**Propriétés :**
- **Admissibilité** : h(n) ≤ h\*(n) → garantit l'optimalité
- **Cohérence** : h(n) ≤ c(n,n') + h(n') → pas de ré-évaluation des nœuds fermés

### Heuristique de Manhattan

```
h(n) = |r₁ - r₂| + |c₁ - c₂|
```

Admissible pour déplacement 4-connexe avec coût unitaire.

### Chaîne de Markov

```
π⁽ⁿ⁾ = π⁽⁰⁾ · Pⁿ

  π⁽⁰⁾ : distribution initiale (masse 1 sur le nœud de départ)
  P    : matrice de transition stochastique (chaque ligne somme à 1)
  GOAL et FAIL : états absorbants
```

---

## Visualisations générées

Lors de l'exécution de `python run_complete_analysis.py` :

```
analysis_results/
├── easy/
│   ├── 01_grid_comparison.png       ← Chemins A*, UCS, Greedy + nœuds explorés
│   ├── 02_metrics_comparison.png    ← Nœuds, coût, temps, efficacité
│   ├── 03_uncertainty_analysis.png  ← P(GOAL) vs ε : tableau + courbe
│   ├── 04_itinerary_tables.png      ← Étape par étape : g, h, f, direction
│   ├── 05_uncertainty_table.png     ← Markov vs MC : stats complètes par ε
│   ├── 06_stochastic_matrix.png     ← Heatmap P + sommes lignes + P[i→j]
│   ├── 05_transition_matrix.png     ← Heatmap matrice (analyse Markov)
│   ├── 06_markov_vs_montecarlo.png  ← Analytique vs empirique + ranking
│   └── 07_absorption_analysis.png  ← P(GOAL), P(FAIL), E[T]
│
├── medium/   (même structure)
├── hard/     (même structure)
│
├── cross_difficulty_comparison.png  ← Évolution Easy → Medium → Hard
└── ANALYSIS_SUMMARY.txt             ← Synthèse textuelle
```

**Total : 27 figures PNG + 1 rapport texte**

---

## API de référence

### `astar.py`

| Fonction | Retour |
|---------|--------|
| `run_astar(start, goal, grid_n, obstacles)` | dict résultat |
| `run_ucs(start, goal, grid_n, obstacles)` | dict résultat |
| `run_greedy(start, goal, grid_n, obstacles)` | dict résultat |
| `run_all(start, goal, grid_n, obstacles)` | `{"astar": ..., "ucs": ..., "greedy": ...}` |
| `build_itinerary(path, goal)` | liste de dicts `{step, cell, g, h, f, dir, type}` |

### `markov.py`

| Fonction | Description |
|---------|-------------|
| `build_transition_matrix(path, grid_n, obstacles, epsilon)` | Construit P en dict creux |
| `verify_stochastic(P)` | Vérifie que toutes les lignes somment à 1.0 |
| `evolve_distribution(path, epsilon, n_steps)` | Historique P(GOAL) pour t=0..n_steps |
| `goal_probability(path, epsilon, steps)` | P(GOAL) après `steps` pas |
| `goal_probability_vs_epsilon(path, epsilon_values)` | `{ε: P(GOAL)}` |
| `monte_carlo(path, grid_n, obstacles, epsilon, n_simulations)` | Simulation MC |
| `absorption_analysis(path, epsilon)` | `{prob_goal, prob_fail, expected_steps}` |

### `grid_visualizer.py`

| Fonction | Description |
|---------|-------------|
| `generate_all_visualizations(grid_n, obstacle_percentage, save_dir)` | Génère les 6 figures |
| `create_grid_visualization(grid_n, obstacles, filename)` | Grille comparative |
| `create_metrics_comparison(results, filename)` | Barres de métriques |
| `create_uncertainty_analysis(results, grid_n, obstacles, filename)` | P(GOAL) vs ε |
| `create_itinerary_table(results, goal, filename)` | Tables itinéraire |
| `create_uncertainty_table(results, grid_n, obstacles, filename)` | Table Markov + MC |
| `create_stochastic_matrix_viz(results, grid_n, obstacles, filename)` | Heatmap P |

### `markov_analysis.py`

| Fonction | Description |
|---------|-------------|
| `generate_markov_analysis(results, grid_n, obstacles, save_dir)` | Génère les 3 figures |
| `visualize_transition_matrix(path, grid_n, obstacles, epsilon, filename)` | Heatmap + connectivité |
| `compare_markov_vs_montecarlo(results, grid_n, obstacles, filename)` | Comparaison + ranking |
| `absorption_comparison(results, grid_n, obstacles, epsilon, filename)` | P(GOAL/FAIL) + E[T] |

---

## Références

1. Hart, P.E., Nilsson, N.J., Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths.* IEEE Transactions on Systems Science and Cybernetics.

2. Russell, S.J., Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4e éd.). Pearson.

3. Norris, J.R. (1997). *Markov Chains.* Cambridge University Press.

4. Puterman, M.L. (1994). *Markov Decision Processes.* Wiley.

5. Dijkstra, E.W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*.

---

**Lancer l'analyse complète :** `python run_complete_analysis.py` → résultats dans `analysis_results/`
