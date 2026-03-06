# 🗺️ Recherche A* + Chaînes de Markov — Navigation sur Grille sous Incertitude

**Projet :** Planification Robuste sur Grilles 2D avec Transitions Stochastiques

---

## 🎓 Cadre Académique
* **Établissement :** Université Hassan II de Casablanca – ENSET Mohammedia
* **Master :** SDIA (Systèmes de Données & Intelligence Artificielle)
* **Module :** Les bases De L'intelligence artificielle
* **Encadrant :** Pr. Mohammed MESTARI
* **Année universitaire :** 2025–2026

## 👥 Auteurs
* **Mustapha Elmifdali**
* **Mbarek Etalebi**

---

## Vue d'ensemble

Ce projet compare trois algorithmes de recherche de chemin (**A\*, UCS (CUC), Glouton**) sur des grilles 2D avec obstacles, puis analyse leur robustesse sous incertitude d'action grâce aux **Chaînes de Markov** et aux simulations **Monte-Carlo**.

---

## Installation

```bash
pip install matplotlib numpy
```

## Démarrage rapide

```bash
# Analyse complète (≈ 10 min, génère 28 images)
python run_complete_analysis.py

# Test rapide (< 1 min, génère 6 images)
python -c "from grid_visualizer import generate_all_visualizations; generate_all_visualizations(15, 20)"

# Notebook interactif
jupyter notebook Robust_Planning_Notebook_FR.ipynb
```

---

## Structure du projet

```
astar.py                          # Algorithmes A*, UCS, Glouton
markov.py                         # Modèle Markov + simulation Monte-Carlo
grid_visualizer.py                # Visualisations grille et métriques
markov_analysis.py                # Analyse avancée Markov
run_complete_analysis.py          # Pipeline complet ← POINT D'ENTRÉE
Robust_Planning_Notebook_FR.ipynb # Notebook reproductible (français)
Rapport_Planification_Robuste_FR.docx  # Rapport technique (10 pages, français)
README.md                         # Ce fichier
```

---

## Visualisations générées

Le pipeline complet crée un dossier `resultats_analyse/` avec 3 sous-dossiers (facile/moyen/difficile), chacun contenant :

| Fichier | Description |
|---------|-------------|
| `01_comparaison_grilles.png` | Les 3 algorithmes côte-à-côte sur la même grille |
| `02_metriques.png` | Nœuds, coût, temps, ratio d'efficacité |
| `03_incertitude.png` | P(But) vs ε — tableau coloré + courbes |
| `04_itineraires.png` | Progression g/h/f pas-à-pas |
| `05_tableau_incertitude.png` | Markov vs Monte-Carlo pour chaque ε |
| `06_matrice_stochastique.png` | Heatmap de la matrice P + vérification |
| `07_matrice_transition.png` | Connectivité des états Markov (A*) |
| `08_markov_vs_monte_carlo.png` | Convergence analytique vs empirique |
| `09_absorption.png` | P(But)/P(Échec) + temps d'absorption espéré |
| `comparaison_difficulte.png` | Scalabilité Facile → Moyen → Difficile |

---

## Galerie des figures clés

### Figure 1 — Comparaison des algorithmes sur grille (15×15, niveau Moyen)

> Cellules violettes = nœuds explorés · Trait coloré = chemin · ⬟ vert = Départ · ■ rouge = But

![Comparaison grilles](exemples_images/01_comparaison_grilles.png)

**A\*** explore nettement moins de cellules que **UCS** pour un coût de chemin identique, grâce à l'heuristique de Manhattan qui guide la recherche vers le but.

---

### Figure 2 — Métriques de performance

![Métriques](exemples_images/02_metriques.png)

Quatre graphiques en barres comparent :
- Le nombre de **nœuds développés** (A\* est 2,5–5× plus efficace que UCS)
- Le **coût du chemin** (A\* et UCS sont toujours optimaux)
- Le **temps d'exécution** en millisecondes
- Le **ratio d'efficacité** normalisé sur UCS = 1,0

---

### Figure 3 — Analyse de robustesse sous incertitude

![Incertitude](exemples_images/03_incertitude.png)

Tableau coloré + courbes P(But) vs ε pour les trois algorithmes. Les chemins courts (A\*/UCS) dégradent moins vite que le Glouton.

| ε (incertitude) | P(Succès) typique |
|:---:|:---:|
| 0,00 | 100 % |
| 0,10 | ≈ 95 % |
| 0,20 | ≈ 80 % |
| 0,30 | ≈ 60 % |

---

### Figure 4 — Itinéraires détaillés pas-à-pas

![Itinéraires](exemples_images/04_itineraires.png)

Pour chaque étape du chemin : cellule, valeurs g(n), h(n), f(n), direction et type (DÉPART / CHEMIN / BUT).

---

### Figure 5 — Tableau d'incertitude étendu (Markov vs Monte-Carlo)

![Tableau incertitude](exemples_images/05_tableau_incertitude.png)

Pour chaque algorithme et chaque valeur de ε : probabilité analytique Markov, estimation Monte-Carlo (N=300), erreur absolue, pas moyens et écart-type.

---

### Figure 6 — Matrice de transition stochastique P

![Matrice stochastique](exemples_images/06_matrice_stochastique.png)

Trois panneaux par algorithme : heatmap P[i,j], vérification que chaque ligne somme à 1,0, et tableau des transitions non-nulles les plus importantes.

---

### Figure 7 — Markov analytique vs Monte-Carlo

![Markov vs MC](exemples_images/08_markov_vs_monte_carlo.png)

Quatre sous-graphiques : résultat analytique, résultat empirique, erreur de convergence (< 5 % pour N ≥ 300), et classement de robustesse des algorithmes.

---

### Figure 8 — Analyse d'absorption

![Absorption](exemples_images/09_absorption.png)

Probabilités P(But) et P(Échec) + longueur du chemin comparée au temps d'absorption espéré E[T] pour chaque algorithme.

---

### Figure 9 — Scalabilité inter-difficulté

![Comparaison difficulté](exemples_images/comparaison_difficulte.png)

Évolution des quatre métriques (nœuds, coût, temps, efficacité) de la grille Facile (10×10) à la grille Difficile (20×20). Montre que le gain de A\* s'amplifie avec la complexité.

---

## Exemples d'utilisation

### Un seul algorithme
```python
from astar import run_astar
result = run_astar((0,0), (14,14), 15, obstacles={(3,3),(3,4),(3,5)})
print(f"Coût: {result['cost']}, Nœuds: {result['nodes']}")
```

### Comparer les trois algorithmes
```python
from astar import run_all
results = run_all((0,0), (14,14), 15, obstacles)
for algo in ['astar', 'ucs', 'greedy']:
    print(f"{algo}: {results[algo]['nodes']} nœuds")
```

### Analyse d'incertitude
```python
from markov import goal_probability_vs_epsilon
path  = results['astar']['path']
probs = goal_probability_vs_epsilon(path, [0.0, 0.1, 0.2, 0.3])
for eps, prob in probs.items():
    print(f"ε={eps:.1f}  P(But)={prob*100:.1f}%")
```

### Simulation Monte-Carlo
```python
from markov import monte_carlo
mc = monte_carlo(path, grid_n=15, obstacles=obstacles,
                 epsilon=0.10, n_simulations=500)
print(f"P̂(Succès) = {mc['prob_goal']*100:.1f}%")
print(f"Pas moyens = {mc['avg_steps']:.1f} ± {mc['std_steps']:.1f}")
```

---

## Fondements mathématiques

### Fonction d'évaluation A\*
```
f(n) = g(n) + h(n)
  g(n) = coût réel depuis le départ
  h(n) = estimation heuristique jusqu'au but
```

### Heuristique de Manhattan (admissible + cohérente)
```
h(pos, but) = |x₁ - x₂| + |y₁ - y₂|
```

### Transition Markov avec bruit
```
P(s'|s, a) = (1 - ε)   si s' = action prévue
           =  ε/2      si s' = déviation latérale
           =  0        sinon
```

### Évolution de la distribution
```
π⁽ⁿ⁾ = π⁽⁰⁾ × Pⁿ
```

---

## Référence de l'API

### `astar.py`
```python
run_astar(start, goal, grid_n, obstacles)    → dict
run_ucs(start, goal, grid_n, obstacles)      → dict
run_greedy(start, goal, grid_n, obstacles)   → dict
run_all(start, goal, grid_n, obstacles)      → dict
build_itinerary(path, goal)                  → list
```

### `markov.py`
```python
build_transition_matrix(path, grid_n, obstacles, epsilon)     → dict
goal_probability(path, epsilon, steps)                        → float
goal_probability_vs_epsilon(path, epsilon_values)             → dict
monte_carlo(path, grid_n, obstacles, epsilon, n_simulations)  → dict
absorption_analysis(path, epsilon)                            → dict
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
```

---

## Résultats clés

- **A\* est 2,5 à 5,1× plus efficace que UCS** (avantage croissant avec la difficulté)
- L'heuristique de Manhattan réduit les nœuds développés de **15–20 %**
- À **ε = 0,1**, le taux de succès reste ≥ 95 % ; à **ε = 0,3**, il chute à ≈ 60 %
- Monte-Carlo converge vers les prédictions analytiques à **< 5 % d'erreur** (N ≥ 300)

---

## Références

1. Hart, Nilsson & Raphael (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths.* IEEE Trans. SSC.
2. Russell & Norvig (2009). *Artificial Intelligence: A Modern Approach* (3rd ed.). Prentice Hall.
3. Norris (1997). *Markov Chains.* Cambridge University Press.
4. Puterman (1994). *Markov Decision Processes.* Wiley.
5. Thrun, Burgard & Fox (2005). *Probabilistic Robotics.* MIT Press.

---

**Lancer l'analyse complète :** `python run_complete_analysis.py`  
**Notebook interactif :** `jupyter notebook Robust_Planning_Notebook_FR.ipynb`  
**Rapport complet :** `Rapport_Planification_Robuste_FR.docx`