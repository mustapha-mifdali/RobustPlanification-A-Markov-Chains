"""
astar.py
────────────────────────────────────────────────────────────
Module de recherche heuristique sur grille 2D.
Algorithmes implémentés :
  - A*     : f(n) = g(n) + h(n)
  - UCS    : f(n) = g(n)
  - Greedy : f(n) = h(n)

Heuristique : Distance de Manhattan (admissible + cohérente)
────────────────────────────────────────────────────────────
"""

import heapq
import time


# ─────────────────────────────────────────────
#  HEURISTIQUE
# ─────────────────────────────────────────────

def manhattan(a: tuple, b: tuple) -> int:
    """
    Heuristique de Manhattan entre deux cellules (r1,c1) et (r2,c2).
    Admissible  : h(n) <= h*(n)  pour coût unitaire
    Cohérente   : h(n) <= c(n,n') + h(n')
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ─────────────────────────────────────────────
#  VOISINS
# ─────────────────────────────────────────────

def get_neighbors(pos: tuple, grid_n: int, obstacles: set) -> list:
    """
    Retourne les voisins valides (4-connexité) de la cellule pos.
    Exclut les obstacles et les cellules hors grille.
    """
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_n and 0 <= nc < grid_n:
            if (nr, nc) not in obstacles:
                neighbors.append((nr, nc))
    return neighbors


# ─────────────────────────────────────────────
#  MOTEUR DE RECHERCHE GÉNÉRIQUE
# ─────────────────────────────────────────────

def search(algo: str,
           start: tuple,
           goal: tuple,
           grid_n: int,
           obstacles: set) -> dict:
    """
    Moteur de recherche générique pour A*, UCS et Greedy.

    Paramètres
    ----------
    algo      : 'astar' | 'ucs' | 'greedy'
    start     : cellule de départ  (row, col)
    goal      : cellule cible      (row, col)
    grid_n    : taille de la grille (grid_n × grid_n)
    obstacles : ensemble des cellules bloquées

    Retour
    ------
    dict avec les clés :
        path      : liste ordonnée de cellules ([] si aucun chemin)
        explored  : liste des cellules développées
        cost      : coût du chemin trouvé (0 si aucun)
        nodes     : nombre de nœuds développés
        time_ms   : temps d'exécution en millisecondes
        algo      : nom de l'algorithme
    """
    t0     = time.perf_counter()
    h0     = manhattan(start, goal)
    ctr    = 0   # tie-breaker pour le tas

    # (f, g, counter, position, chemin)
    open_heap = [(h0 if algo != "ucs" else 0, 0, ctr, start, [start])]
    g_score   = {start: 0}
    closed    = set()
    nodes_exp = 0

    while open_heap:
        f, g, _, pos, path_so_far = heapq.heappop(open_heap)

        if pos in closed:
            continue
        closed.add(pos)
        nodes_exp += 1

        if pos == goal:
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "path"    : path_so_far,
                "explored": list(closed),
                "cost"    : g,
                "nodes"   : nodes_exp,
                "time_ms" : round(elapsed, 3),
                "algo"    : algo,
            }

        for nb in get_neighbors(pos, grid_n, obstacles):
            ng = g + 1
            if ng < g_score.get(nb, float("inf")):
                g_score[nb] = ng
                h  = manhattan(nb, goal)
                if   algo == "ucs":    fval = ng
                elif algo == "greedy": fval = h
                else:                  fval = ng + h   # A*
                ctr += 1
                heapq.heappush(open_heap,
                               (fval, ng, ctr, nb, path_so_far + [nb]))

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "path"    : [],
        "explored": list(closed),
        "cost"    : 0,
        "nodes"   : nodes_exp,
        "time_ms" : round(elapsed, 3),
        "algo"    : algo,
    }


# ─────────────────────────────────────────────
#  FONCTIONS DE COMMODITÉ
# ─────────────────────────────────────────────

def run_astar(start, goal, grid_n, obstacles):
    """Lance A* et retourne le résultat."""
    return search("astar", start, goal, grid_n, obstacles)

def run_ucs(start, goal, grid_n, obstacles):
    """Lance UCS (Uniform Cost Search) et retourne le résultat."""
    return search("ucs", start, goal, grid_n, obstacles)

def run_greedy(start, goal, grid_n, obstacles):
    """Lance la recherche gloutonne et retourne le résultat."""
    return search("greedy", start, goal, grid_n, obstacles)

def run_all(start, goal, grid_n, obstacles):
    """
    Lance les 3 algorithmes et retourne un dict comparatif.
    Retour : {"astar": {...}, "ucs": {...}, "greedy": {...}}
    """
    return {
        "astar" : run_astar (start, goal, grid_n, obstacles),
        "ucs"   : run_ucs   (start, goal, grid_n, obstacles),
        "greedy": run_greedy(start, goal, grid_n, obstacles),
    }


# ─────────────────────────────────────────────
#  ITINÉRAIRE (tableau pas-à-pas)
# ─────────────────────────────────────────────

DIRECTION_NAMES = {
    (-1,  0): "↑ Nord",
    ( 1,  0): "↓ Sud",
    ( 0, -1): "← Ouest",
    ( 0,  1): "→ Est",
}

def build_itinerary(path: list, goal: tuple) -> list:
    """
    Construit un itinéraire détaillé à partir d'un chemin.

    Retour : liste de dicts, un par étape :
        step  : indice de l'étape
        cell  : (row, col)
        g     : coût depuis le départ
        h     : heuristique Manhattan vers le but
        f     : g + h
        dir   : direction du prochain mouvement
        type  : 'START' | 'PATH' | 'GOAL'
    """
    itinerary = []
    for i, cell in enumerate(path):
        direction = "—"
        if i < len(path) - 1:
            dr = path[i+1][0] - cell[0]
            dc = path[i+1][1] - cell[1]
            direction = DIRECTION_NAMES.get((dr, dc), "?")

        g_val = i
        h_val = manhattan(cell, goal)
        f_val = g_val + h_val
        cell_type = "START" if i == 0 else "GOAL" if i == len(path)-1 else "PATH"

        itinerary.append({
            "step": i,
            "cell": cell,
            "g"   : g_val,
            "h"   : h_val,
            "f"   : f_val,
            "dir" : direction,
            "type": cell_type,
        })
    return itinerary


# ─────────────────────────────────────────────
#  TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    N    = 10
    S    = (0, 0)
    GOAL = (N-1, N-1)
    OBS  = {(2,3),(3,3),(4,3),(4,4),(4,5),(6,2),(7,2)}

    print("=" * 50)
    print("  Test — astar.py")
    print("=" * 50)

    results = run_all(S, GOAL, N, OBS)
    for algo, r in results.items():
        status = f"coût={r['cost']}  nœuds={r['nodes']}  {r['time_ms']}ms"
        print(f"  {algo.upper():8s} : {status if r['path'] else 'Aucun chemin'}")

    print()
    itin = build_itinerary(results["astar"]["path"], GOAL)
    print(f"  Itinéraire A* ({len(itin)} étapes) :")
    print(f"  {'#':>3}  {'Cellule':>10}  {'g':>3}  {'h':>3}  {'f':>3}  {'Dir':>8}  Type")
    print("  " + "-"*50)
    for row in itin:
        print(f"  {row['step']:>3}  {str(row['cell']):>10}  "
              f"{row['g']:>3}  {row['h']:>3}  {row['f']:>3}  "
              f"{row['dir']:>8}  {row['type']}")