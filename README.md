# A* Search + Markov Chains: Grid Pathfinding Under Uncertainty

**Project:** Robust Planning on 2D Grids with Stochastic Transitions

## Overview

This project compares three pathfinding algorithms (A*, UCS, Greedy) on 2D grids with obstacles, then analyzes their robustness under action uncertainty using Markov chains.

**Features:**
- A* search with Manhattan heuristic (optimal)
- UCS (baseline optimal)
- Greedy search (fast, non-optimal)
- Markov chain uncertainty modeling
- Monte-Carlo validation
- Automatic visualization generation

## Installation

```bash
pip install matplotlib numpy
```

## Quick Start

### Run Complete Analysis (15 minutes)
```bash
python run_complete_analysis.py
```
Generates 22 PNG visualizations across 3 difficulty levels + summary report.

### Quick Test (30 seconds)
```bash
python -c "from grid_visualizer import generate_all_visualizations; generate_all_visualizations(grid_n=15, obstacle_percentage=20)"
```
Generates 4 visualizations in `grid_results/` folder.

## File Structure

```
astar.py                    # A*, UCS, Greedy algorithms
markov.py                   # Markov chain modeling + simulation
grid_visualizer.py          # Grid & metrics visualizations
markov_analysis.py          # Markov-specific analysis
run_complete_analysis.py    # Complete pipeline (MAIN ENTRY)
README.md                   # This file
Rapport_Robust_Planning.docx # 10-page technical report
images/                     # Generated PNG files
```

## Usage Examples

### Example 1: Run Single Algorithm
```python
from astar import run_astar

result = run_astar((0,0), (14,14), 15, obstacles={(3,3), (3,4), (3,5)})
print(f"Cost: {result['cost']}, Nodes: {result['nodes']}")
```

### Example 2: Compare All Algorithms
```python
from astar import run_all

results = run_all((0,0), (14,14), 15, obstacles)
for algo in ['astar', 'ucs', 'greedy']:
    print(f"{algo}: {results[algo]['nodes']} nodes")
```

### Example 3: Uncertainty Analysis
```python
from markov import goal_probability_vs_epsilon

path = results['astar']['path']
probs = goal_probability_vs_epsilon(path, [0.0, 0.1, 0.2, 0.3])
for eps, prob in probs.items():
    print(f"ε={eps:.1f}  P(GOAL)={prob*100:.1f}%")
```

### Example 4: Generate Grid Visualization
```python
from grid_visualizer import generate_all_visualizations

generate_all_visualizations(grid_n=15, obstacle_percentage=20)
```

## Generated Visualizations

When running the complete pipeline, you get:

**Per Grid (Easy/Medium/Hard):**
- `01_grid_comparison.png` - All 3 algorithms side-by-side
- `02_metrics_comparison.png` - Nodes, cost, time, efficiency
- `03_uncertainty_analysis.png` - P(GOAL) vs action deviation (ε)
- `04_itinerary_tables.png` - Step-by-step path breakdown
- `05_transition_matrix.png` - Markov matrix heatmap
- `06_markov_vs_montecarlo.png` - Analytical vs empirical comparison
- `07_absorption_analysis.png` - Success/failure probabilities

**Cross-Difficulty:**
- `cross_difficulty_comparison.png` - Scaling analysis (Easy→Medium→Hard)
- `ANALYSIS_SUMMARY.txt` - Text report with findings

## Key Results

### Algorithm Efficiency
- **A* is 2.5-5.1× faster than UCS** (advantage grows with difficulty)
- All algorithms find optimal paths (UCS and A*)
- Greedy is fastest but finds longer paths

### Heuristic Quality
- Manhattan heuristic: 15-20% reduction in nodes expanded
- Both admissible and consistent

### Robustness Under Uncertainty
- ε=0.0 (no noise): 100% success
- ε=0.1 (low noise): 95% success
- ε=0.2 (moderate): 80% success
- ε=0.3 (high noise): 60% success

## Mathematical Foundations

### A* Evaluation Function
```
f(n) = g(n) + h(n)
where: g(n) = actual cost from start
       h(n) = heuristic estimate to goal
```

### Manhattan Heuristic (Admissible)
```
h(pos, goal) = |x₁ - x₂| + |y₁ - y₂|
```

### Markov Transition with Noise
```
P(s'|s,a) = (1-ε)    if s' = intended action
          = ε/2      if s' = lateral deviation
          = 0        otherwise
```

### Distribution Evolution
```
π(n) = π(0) × Pⁿ
```

## Test Grids

| Grid | Size | Obstacles | Type |
|------|------|-----------|------|
| Easy | 10×10 | 5 | Simple |
| Medium | 15×15 | 20 | Moderate |
| Hard | 20×20 | 50 | Complex |

## Module API Reference

### astar.py
```python
run_astar(start, goal, grid_n, obstacles) → dict
run_ucs(start, goal, grid_n, obstacles) → dict
run_greedy(start, goal, grid_n, obstacles) → dict
run_all(start, goal, grid_n, obstacles) → dict
build_itinerary(path, goal) → list
```

### markov.py
```python
build_transition_matrix(path, grid_n, obstacles, epsilon) → dict
goal_probability(path, epsilon, steps) → float
goal_probability_vs_epsilon(path, epsilon_values) → dict
monte_carlo(path, grid_n, obstacles, epsilon, n_simulations) → dict
absorption_analysis(path, epsilon) → dict
```

### grid_visualizer.py
```python
create_grid_visualization(grid_n, obstacles, filename)
create_metrics_comparison(results, filename)
create_uncertainty_analysis(results, grid_n, obstacles, filename)
create_itinerary_table(results, goal, filename)
generate_all_visualizations(grid_n, obstacle_percentage, save_dir)
```

### markov_analysis.py
```python
visualize_transition_matrix(path, grid_n, obstacles, epsilon, filename)
compare_markov_vs_montecarlo(results, grid_n, obstacles, filename)
absorption_comparison(results, grid_n, obstacles, epsilon, filename)
generate_markov_analysis(results, grid_n, obstacles, save_dir)
```

### run_complete_analysis.py
```python
class AnalysisPipeline:
    def run_complete_pipeline()  # MAIN ENTRY POINT
```

## Performance

| Task | Time |
|------|------|
| Single grid (15×15) | 30 seconds |
| 3 grids | 2 minutes |
| Complete pipeline | 15 minutes |
| Monte-Carlo (500 sim) | 1-2 minutes |

## Requirements

- Python 3.7+
- numpy
- matplotlib
- Optional: pygame-ce (for interactive interface)

## References

1. Hart, P.E., Nilsson, N.J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Transactions on Systems Science and Cybernetics.

2. Russell, S.J., & Norvig, P. (2009). Artificial Intelligence: A Modern Approach (3rd ed.). Prentice Hall.

3. Norris, J.R. (1997). Markov Chains. Cambridge University Press.

4. Puterman, M.L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley.

## License

Educational and research use.

---

**Start with:** `python run_complete_analysis.py`

**Report:** See `Rapport_Robust_Planning.docx`

**Generated images:** Check `images/` folder after running
