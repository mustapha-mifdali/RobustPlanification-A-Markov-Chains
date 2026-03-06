# 🗺️ Recherche A* + Chaînes de Markov — Navigation sur Grille sous Incertitude

**Projet :** Planification Robuste sur Grilles 2D avec Transitions Stochastiques

---

## 🎓 Cadre Académique

| | |
|---|---|
| **Établissement** | Université Hassan II de Casablanca – ENSET Mohammedia |
| **Master** | SDIA (Systèmes de Données & Intelligence Artificielle) |
| **Module** | Les bases de l'Intelligence Artificielle |
| **Encadrant** | Pr. Mohammed MESTARI |
| **Année universitaire** | 2025–2026 |

## 👥 Auteurs

- **Mustapha Elmifdali**
- **Mbarek Etalebi**

---

## 📋 Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Installation & Démarrage rapide](#2-installation--démarrage-rapide)
3. [Structure du projet](#3-structure-du-projet)
4. [Grilles de test](#4-grilles-de-test)
5. [Visualisations générées](#5-visualisations-générées)
6. [Galerie — Niveau Facile (10×10)](#6-galerie--niveau-facile-1010)
7. [Galerie — Niveau Moyen (15×15)](#7-galerie--niveau-moyen-1515)
8. [Galerie — Niveau Difficile (20×20)](#8-galerie--niveau-difficile-2020)
9. [Galerie — Comparaison inter-difficulté](#9-galerie--comparaison-inter-difficulté)
10. [Galerie — Résultats Markov (pipeline rapide)](#10-galerie--résultats-markov-pipeline-rapide)
11. [Exemples d'utilisation](#11-exemples-dutilisation)
12. [Fondements mathématiques](#12-fondements-mathématiques)
13. [Référence de l'API](#13-référence-de-lapi)
14. [Résultats clés](#14-résultats-clés)
15. [Références bibliographiques](#15-références-bibliographiques)

---

## 1. Vue d'ensemble

Ce projet combine deux techniques complémentaires pour la **planification robuste** sur grille :

- **A\* (A-étoile)** — algorithme de recherche heuristique qui trouve des chemins optimaux en combinant le coût réel `g(n)` et l'estimation heuristique `h(n)`. Avec une heuristique admissible, A\* garantit l'optimalité.
- **Chaînes de Markov** — modèles stochastiques qui capturent l'incertitude d'action via un paramètre `ε`. Nous analysons l'évolution `π⁽ⁿ⁾ = π⁽⁰⁾·Pⁿ` et validons par simulation Monte-Carlo.

**Fonctionnalités :**
- Algorithme A\* avec heuristique de Manhattan (optimal + admissible + cohérent)
- UCS — Recherche à coût uniforme (référence optimale, h = 0)
- Recherche Gloutonne (rapide, non-optimale)
- Modélisation stochastique par Chaînes de Markov
- Validation empirique par simulation Monte-Carlo
- Génération automatique de **28 visualisations** haute-qualité

---

## 2. Installation & Démarrage rapide

```bash
pip install matplotlib numpy
```

```bash
# Analyse complète — génère 28 images dans resultats_analyse/
python run_complete_analysis.py

# Test rapide — génère 6 images dans resultats_grille/
python -c "from grid_visualizer import generate_all_visualizations; generate_all_visualizations(15, 20)"

# Notebook interactif
jupyter notebook Robust_Planning_Notebook_FR.ipynb
```

---

## 3. Structure du projet

```
📁 projet/
│
├── astar.py                               # Algorithmes A*, UCS, Glouton
├── markov.py                              # Modèle Markov + Monte-Carlo
├── grid_visualizer.py                     # Visualisations grille et métriques
├── markov_analysis.py                     # Analyse avancée Markov
├── run_complete_analysis.py               # Pipeline complet ← POINT D'ENTRÉE
├── Robust_Planning_Notebook_FR.ipynb      # Notebook reproductible (français)
├── Rapport_Planification_Robuste_FR.docx  # Rapport technique (10 pages)
├── README.md                              # Ce fichier
│
├── 📁 resultats_analyse/                  # ← généré par run_complete_analysis.py
│   ├── 📁 facile/          (grille 10×10 — 9 figures)
│   │   ├── 01_comparaison_grilles.png
│   │   ├── 02_metriques.png
│   │   ├── 03_incertitude.png
│   │   ├── 04_itineraires.png
│   │   ├── 05_tableau_incertitude.png
│   │   ├── 06_matrice_stochastique.png
│   │   ├── 07_matrice_transition.png
│   │   ├── 08_markov_vs_monte_carlo.png
│   │   └── 09_absorption.png
│   ├── 📁 moyen/           (grille 15×15 — 9 figures)
│   │   └── (mêmes 9 fichiers)
│   ├── 📁 difficile/       (grille 20×20 — 9 figures)
│   │   └── (mêmes 9 fichiers)
│   ├── comparaison_difficulte.png
│   └── RAPPORT_SYNTHESE.txt
│
├── 📁 resultats_grille/                   # ← généré par generate_all_visualizations()
│   ├── 01_comparaison_grilles.png
│   ├── 02_metriques.png
│   ├── 03_incertitude.png
│   ├── 04_itineraires.png
│   ├── 05_tableau_incertitude.png
│   └── 06_matrice_stochastique.png
│
└── 📁 resultats_markov/                   # ← généré par generate_markov_analysis()
    ├── 01_matrice_transition.png
    ├── 02_markov_vs_monte_carlo.png
    └── 03_absorption.png
```

---

## 4. Grilles de test

| Niveau | Taille | Obstacles | Départ | But |
|:------:|:------:|:---------:|:------:|:---:|
| Facile | 10×10 | 5 | (0,0) | (9,9) |
| Moyen | 15×15 | 20 | (0,0) | (14,14) |
| Difficile | 20×20 | 50 | (0,0) | (19,19) |

---

## 5. Visualisations générées

| Fichier | Description |
|---------|-------------|
| `01_comparaison_grilles.png` | Les 3 algorithmes côte-à-côte sur la même grille |
| `02_metriques.png` | Nœuds, coût, temps, ratio d'efficacité |
| `03_incertitude.png` | P(But) vs ε — tableau coloré + courbes |
| `04_itineraires.png` | Progression g/h/f pas-à-pas |
| `05_tableau_incertitude.png` | Markov vs Monte-Carlo pour chaque ε |
| `06_matrice_stochastique.png` | Heatmap matrice P + vérification sommes lignes |
| `07_matrice_transition.png` | Connectivité des états Markov (chemin A*) |
| `08_markov_vs_monte_carlo.png` | Convergence analytique vs empirique + classement |
| `09_absorption.png` | P(But)/P(Échec) + temps d'absorption espéré E[T] |
| `comparaison_difficulte.png` | Scalabilité Facile → Moyen → Difficile |

---

## 6. Galerie — Niveau Facile (10×10)

### 6.1 Comparaison des algorithmes sur grille
> 🟣 Cellules violettes = nœuds explorés · Trait coloré = chemin · 🟢 Départ · 🔴 But

![Comparaison grilles — Facile](resultats_analyse/facile/01_comparaison_grilles.png)

---

### 6.2 Métriques de performance
![Métriques — Facile](resultats_analyse/facile/02_metriques.png)

---

### 6.3 Analyse de robustesse sous incertitude
![Incertitude — Facile](resultats_analyse/facile/03_incertitude.png)

---

### 6.4 Itinéraires détaillés pas-à-pas
![Itinéraires — Facile](resultats_analyse/facile/04_itineraires.png)

---

### 6.5 Tableau d'incertitude étendu (Markov vs Monte-Carlo)
![Tableau incertitude — Facile](resultats_analyse/facile/05_tableau_incertitude.png)

---

### 6.6 Matrice de transition stochastique P
![Matrice stochastique — Facile](resultats_analyse/facile/06_matrice_stochastique.png)

---

### 6.7 Matrice de transition — connectivité Markov
![Matrice transition — Facile](resultats_analyse/facile/07_matrice_transition.png)

---

### 6.8 Markov analytique vs Monte-Carlo
![Markov vs MC — Facile](resultats_analyse/facile/08_markov_vs_monte_carlo.png)

---

### 6.9 Analyse d'absorption
![Absorption — Facile](resultats_analyse/facile/09_absorption.png)

---

## 7. Galerie — Niveau Moyen (15×15)

### 7.1 Comparaison des algorithmes sur grille
![Comparaison grilles — Moyen](resultats_analyse/moyen/01_comparaison_grilles.png)

---

### 7.2 Métriques de performance
![Métriques — Moyen](resultats_analyse/moyen/02_metriques.png)

---

### 7.3 Analyse de robustesse sous incertitude
![Incertitude — Moyen](resultats_analyse/moyen/03_incertitude.png)

| ε (déviation) | P(Succès) typique |
|:---:|:---:|
| 0,00 | 100 % |
| 0,10 | ≈ 95 % |
| 0,20 | ≈ 80 % |
| 0,30 | ≈ 60 % |

---

### 7.4 Itinéraires détaillés pas-à-pas
![Itinéraires — Moyen](resultats_analyse/moyen/04_itineraires.png)

---

### 7.5 Tableau d'incertitude étendu (Markov vs Monte-Carlo)
![Tableau incertitude — Moyen](resultats_analyse/moyen/05_tableau_incertitude.png)

---

### 7.6 Matrice de transition stochastique P
![Matrice stochastique — Moyen](resultats_analyse/moyen/06_matrice_stochastique.png)

---

### 7.7 Matrice de transition — connectivité Markov
![Matrice transition — Moyen](resultats_analyse/moyen/07_matrice_transition.png)

---

### 7.8 Markov analytique vs Monte-Carlo
![Markov vs MC — Moyen](resultats_analyse/moyen/08_markov_vs_monte_carlo.png)

---

### 7.9 Analyse d'absorption
![Absorption — Moyen](resultats_analyse/moyen/09_absorption.png)

---

## 8. Galerie — Niveau Difficile (20×20)

### 8.1 Comparaison des algorithmes sur grille
![Comparaison grilles — Difficile](resultats_analyse/difficile/01_comparaison_grilles.png)

---

### 8.2 Métriques de performance
![Métriques — Difficile](resultats_analyse/difficile/02_metriques.png)

---

### 8.3 Analyse de robustesse sous incertitude
![Incertitude — Difficile](resultats_analyse/difficile/03_incertitude.png)

---

### 8.4 Itinéraires détaillés pas-à-pas
![Itinéraires — Difficile](resultats_analyse/difficile/04_itineraires.png)

---

### 8.5 Tableau d'incertitude étendu (Markov vs Monte-Carlo)
![Tableau incertitude — Difficile](resultats_analyse/difficile/05_tableau_incertitude.png)

---

### 8.6 Matrice de transition stochastique P
![Matrice stochastique — Difficile](resultats_analyse/difficile/06_matrice_stochastique.png)

---

### 8.7 Matrice de transition — connectivité Markov
![Matrice transition — Difficile](resultats_analyse/difficile/07_matrice_transition.png)

---

### 8.8 Markov analytique vs Monte-Carlo
![Markov vs MC — Difficile](resultats_analyse/difficile/08_markov_vs_monte_carlo.png)

---

### 8.9 Analyse d'absorption
![Absorption — Difficile](resultats_analyse/difficile/09_absorption.png)

---

## 9. Galerie — Comparaison inter-difficulté

![Comparaison inter-difficulté](resultats_analyse/comparaison_difficulte.png)

Évolution des quatre métriques (nœuds, coût, temps, ratio d'efficacité) de la grille **Facile (10×10)** à la grille **Difficile (20×20)**. Le gain de A\* s'amplifie à mesure que la complexité augmente.

---

## 10. Galerie — Résultats Markov (pipeline rapide)

Ces figures sont produites par `generate_markov_analysis()` sur une grille unique.

### 10.1 Matrice de transition — heatmap et connectivité
![Matrice transition](resultats_markov/01_matrice_transition.png)

---

### 10.2 Markov analytique vs Monte-Carlo
![Markov vs MC](resultats_markov/02_markov_vs_monte_carlo.png)

---

### 10.3 Analyse d'absorption
![Absorption](resultats_markov/03_absorption.png)

---

## 11. Exemples d'utilisation

### Exécuter un seul algorithme
```python
from astar import run_astar

result = run_astar((0,0), (14,14), 15, obstacles={(3,3),(3,4),(3,5)})
print(f"Coût: {result['cost']}, Nœuds: {result['nodes']}, Temps: {result['time_ms']} ms")
```

### Comparer les trois algorithmes
```python
from astar import run_all

results = run_all((0,0), (14,14), 15, obstacles)
for algo in ['astar', 'ucs', 'greedy']:
    r = results[algo]
    print(f"{algo:8s} | coût={r['cost']}  nœuds={r['nodes']}  temps={r['time_ms']} ms")
```

### Analyse d'incertitude Markov
```python
from markov import goal_probability_vs_epsilon

path  = results['astar']['path']
probs = goal_probability_vs_epsilon(path, [0.0, 0.1, 0.2, 0.3])
for eps, prob in probs.items():
    print(f"ε={eps:.1f}  →  P(But) = {prob*100:.1f}%")
```

### Simulation Monte-Carlo
```python
from markov import monte_carlo

mc = monte_carlo(path, grid_n=15, obstacles=obstacles,
                 epsilon=0.10, n_simulations=500)
print(f"P̂(Succès) = {mc['prob_goal']*100:.1f}%")
print(f"Pas moyens = {mc['avg_steps']:.1f} ± {mc['std_steps']:.1f}")
```

### Analyse Markov complète (pipeline rapide)
```python
from astar import run_all
from markov_analysis import generate_markov_analysis

results   = run_all((0,0), (14,14), 15, obstacles)
generate_markov_analysis(results, grid_n=15, obstacles=obstacles,
                         save_dir='resultats_markov')
# → produit resultats_markov/01_matrice_transition.png
#            resultats_markov/02_markov_vs_monte_carlo.png
#            resultats_markov/03_absorption.png
```

---

## 12. Fondements mathématiques

### Fonction d'évaluation A\*
```
f(n) = g(n) + h(n)
  g(n) = coût réel depuis le départ jusqu'à n
  h(n) = estimation heuristique de n jusqu'au but
  f(n) = coût total estimé du chemin passant par n
```

### Heuristique de Manhattan (admissible + cohérente)
```
h(pos, but) = |pos.x − but.x| + |pos.y − but.y|

  Admissible  : h(n) ≤ h*(n)  pour tout n  →  garantit l'optimalité
  Cohérente   : h(n) ≤ c(n,m) + h(m)       →  pas de re-développement
```

### Modèle de transition Markov avec bruit ε
```
P(s' | s, a) =  1 − ε    si s' = action_prévue(s)    [succès]
             =  ε / 2    si s' ∈ {voisins latéraux}   [déviation]
             =  0        sinon
```

### Évolution de la distribution de probabilité
```
π⁽ⁿ⁾ = π⁽⁰⁾ × Pⁿ

  π⁽⁰⁾ = distribution initiale (masse = 1 sur l'état de départ)
  Pⁿ   = matrice de transition à la puissance n
  π⁽ⁿ⁾ = distribution sur les états après n pas de temps
```

---

## 13. Référence de l'API

### `astar.py`
```python
run_astar(start, goal, grid_n, obstacles)    → dict
run_ucs(start, goal, grid_n, obstacles)      → dict
run_greedy(start, goal, grid_n, obstacles)   → dict
run_all(start, goal, grid_n, obstacles)      → dict  # lance les 3 en une fois
build_itinerary(path, goal)                  → list  # décomposition pas-à-pas
```

### `markov.py`
```python
build_transition_matrix(path, grid_n, obstacles, epsilon)      → dict
goal_probability(path, epsilon, steps)                         → float
goal_probability_vs_epsilon(path, epsilon_values)              → dict
monte_carlo(path, grid_n, obstacles, epsilon, n_simulations)   → dict
absorption_analysis(path, epsilon)                             → dict
```

### `grid_visualizer.py`
```python
create_grid_visualization(grid_n, obstacles, filename)
create_metrics_comparison(results, filename)
create_uncertainty_analysis(results, grid_n, obstacles, filename)
create_itinerary_table(results, goal, filename)
create_uncertainty_table(results, grid_n, obstacles, filename)
create_stochastic_matrix_viz(results, grid_n, obstacles, filename)
generate_all_visualizations(grid_n, obstacle_percentage, save_dir)
# → produit resultats_grille/ avec 6 figures
```

### `markov_analysis.py`
```python
visualize_transition_matrix(path, grid_n, obstacles, epsilon, filename)
compare_markov_vs_montecarlo(results, grid_n, obstacles, filename)
absorption_comparison(results, grid_n, obstacles, epsilon, filename)
generate_markov_analysis(results, grid_n, obstacles, save_dir)
# → produit resultats_markov/ avec 3 figures
```

### `run_complete_analysis.py`
```python
class PipelineAnalyse:
    generer_grilles()                  # Étape 1 — création des 3 grilles
    executer_algorithmes()             # Étape 2 — A*, UCS, Glouton
    generer_visualisations_grille()    # Étape 3 — 27 figures (9 × 3 niveaux)
    generer_analyse_markov()           # Étape 4 — analyse stochastique
    generer_comparaison_difficulte()   # Étape 5 — comparaison inter-niveaux
    generer_rapport()                  # Étape 6 — RAPPORT_SYNTHESE.txt
    executer_pipeline_complet()        # ← POINT D'ENTRÉE PRINCIPAL
```

---

## 14. Résultats clés

| Métrique | Résultat |
|----------|----------|
| Efficacité A\* vs UCS | **2,5× à 5,1×** (croît avec la difficulté) |
| Réduction nœuds par heuristique | **15–20 %** vs h = 0 |
| P(Succès) à ε = 0,10 | **≥ 95 %** |
| P(Succès) à ε = 0,30 | **≈ 60 %** |
| Erreur Markov vs Monte-Carlo | **< 5 %** pour N ≥ 300 |

| Tâche | Durée estimée |
|-------|:---:|
| Grille unique 15×15 | ~30 s |
| Pipeline complet (3 grilles) | ~10 min |
| Monte-Carlo seul (500 sim.) | ~1–2 min |

---

## 15. Références bibliographiques

1. Hart, P.E., Nilsson, N.J. & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths.* IEEE Transactions on Systems Science and Cybernetics.
2. Russell, S.J. & Norvig, P. (2009). *Artificial Intelligence: A Modern Approach* (3rd ed.). Prentice Hall.
3. Norris, J.R. (1997). *Markov Chains.* Cambridge University Press.
4. Puterman, M.L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming.* Wiley.
5. Thrun, S., Burgard, W. & Fox, D. (2005). *Probabilistic Robotics.* MIT Press.

---

<div align="center">

**Lancer l'analyse complète :** `python run_complete_analysis.py`  
**Notebook interactif :** `jupyter notebook Robust_Planning_Notebook_FR.ipynb`  
**Rapport complet :** `Rapport_Planification_Robuste_FR.docx`

---

*Université Hassan II de Casablanca – ENSET Mohammedia · Master SDIA · 2025–2026*

</div>