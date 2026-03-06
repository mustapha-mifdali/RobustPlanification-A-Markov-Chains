"""
markov.py
────────────────────────────────────────────────────────────
Module de modélisation stochastique par Chaînes de Markov.
Fonctionnalités :
  - Construction de la matrice de transition P
  - Évolution de la distribution  π⁽ⁿ⁾ = π⁽⁰⁾ · Pⁿ
  - Calcul de P(atteindre GOAL) par puissance de matrice
  - Analyse des états absorbants (GOAL / FAIL)
  - Simulation Monte-Carlo de trajectoires
────────────────────────────────────────────────────────────
"""

import random
import time


# ─────────────────────────────────────────────
#  CONSTRUCTION DE LA MATRICE P
# ─────────────────────────────────────────────

def build_transition_matrix(path: list,
                             grid_n: int,
                             obstacles: set,
                             epsilon: float) -> dict:
    """
    Construit la matrice de transition P induite par la politique
    définie par le chemin A*/UCS/Greedy.

    Modèle d'incertitude (paramètre ε) :
      - avec probabilité (1 - ε)  : l'agent suit la direction prévue
      - avec probabilité ε/2      : déviation vers un voisin latéral
      - si collision / obstacle   : l'agent reste sur place

    États spéciaux :
      "GOAL"  : état absorbant de succès  (p_GOAL,GOAL = 1)
      "FAIL"  : état absorbant d'échec    (p_FAIL,FAIL = 1)

    Retour
    ------
    dict  {état_i : {état_j : probabilité}}   (représentation creuse)
    """
    if not path or len(path) < 2:
        return {}

    # politique : pour chaque cellule du chemin, action recommandée
    policy = {}
    for i in range(len(path) - 1):
        dr = path[i+1][0] - path[i][0]
        dc = path[i+1][1] - path[i][1]
        policy[tuple(path[i])] = (dr, dc)

    P = {}

    for i, cell in enumerate(path[:-1]):
        state   = tuple(cell)
        dr, dc  = policy[state]

        # voisins latéraux (perpendiculaires à la direction prévue)
        if dr == 0:   laterals = [(-1, dc), ( 1, dc)]
        else:         laterals = [(dr, -1), (dr,  1)]

        row = {}

        # --- action voulue ---
        nr, nc = cell[0]+dr, cell[1]+dc
        intended = tuple(path[i+1])
        if 0 <= nr < grid_n and 0 <= nc < grid_n and (nr,nc) not in obstacles:
            row[intended] = row.get(intended, 0) + (1 - epsilon)
        else:
            row[state] = row.get(state, 0) + (1 - epsilon)   # reste sur place

        # --- déviations latérales ---
        for ldr, ldc in laterals:
            ln, lc = cell[0]+ldr, cell[1]+ldc
            if 0 <= ln < grid_n and 0 <= lc < grid_n and (ln,lc) not in obstacles:
                lateral_state = (ln, lc)
            else:
                lateral_state = state   # rebond
            row[lateral_state] = row.get(lateral_state, 0) + epsilon / 2

        P[state] = row

    # --- état final → GOAL absorbant ---
    goal_state = tuple(path[-1])
    P[goal_state] = {"GOAL": 1.0}

    # --- état absorbants ---
    P["GOAL"] = {"GOAL": 1.0}
    P["FAIL"] = {"FAIL": 1.0}

    return P


def verify_stochastic(P: dict) -> bool:
    """
    Vérifie que P est bien stochastique (somme des lignes = 1).
    Retourne True si valide, False sinon.
    """
    for state, row in P.items():
        total = sum(row.values())
        if abs(total - 1.0) > 1e-9:
            return False
    return True


# ─────────────────────────────────────────────
#  ÉVOLUTION DE LA DISTRIBUTION π⁽ⁿ⁾
# ─────────────────────────────────────────────

def evolve_distribution(path: list,
                         epsilon: float,
                         n_steps: int = 40) -> list:
    """
    Calcule π⁽ⁿ⁾ = π⁽⁰⁾ · Pⁿ sur le chemin simplifié.

    Modèle linéaire sur le chemin (états 0..k → GOAL) :
      - dist[i]   = probabilité d'être à l'étape i du chemin
      - dist[k+1] = probabilité d'avoir atteint GOAL

    Retour : liste de P(GOAL) à chaque pas de temps 0..n_steps
    """
    k = len(path)
    if k < 2:
        return [0.0] * (n_steps + 1)

    dist = [0.0] * (k + 1)
    dist[0] = 1.0
    history = [dist[k]]   # P(GOAL) à t=0

    for _ in range(n_steps):
        nd = [0.0] * (k + 1)
        nd[k] = dist[k]   # GOAL absorbant

        for i in range(k - 1):
            # avance vers i+1
            nd[i+1] += dist[i] * (1 - epsilon)
            # reste sur place (déviation latérale rebondie)
            nd[i]   += dist[i] * (epsilon * 0.5)
            # recule vers i-1
            if i > 0: nd[i-1] += dist[i] * (epsilon * 0.5)
            else:     nd[i]   += dist[i] * (epsilon * 0.5)

        # dernier état avant GOAL
        nd[k]   += dist[k-1] * (1 - epsilon * 0.5)
        nd[k-1] += dist[k-1] * (epsilon * 0.5)

        dist = nd
        history.append(dist[k])

    return history


def goal_probability(path: list, epsilon: float, steps: int = 40) -> float:
    """
    Retourne P(atteindre GOAL après `steps` pas) pour un chemin donné
    et un niveau d'incertitude ε.
    """
    hist = evolve_distribution(path, epsilon, steps)
    return hist[-1] if hist else 0.0


def goal_probability_vs_epsilon(path: list,
                                 epsilon_values: list = None,
                                 steps: int = 40) -> dict:
    """
    Calcule P(GOAL) pour chaque valeur de ε dans epsilon_values.

    Retour : {epsilon: probability}
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    return {e: goal_probability(path, e, steps) for e in epsilon_values}


# ─────────────────────────────────────────────
#  ANALYSE DE L'ABSORPTION
# ─────────────────────────────────────────────

def absorption_analysis(path: list, epsilon: float) -> dict:
    """
    Analyse les probabilités d'absorption vers GOAL et FAIL.

    Décomposition canonique P = [I 0 / R Q] :
      - Q : sous-matrice des états transitoires
      - N = (I - Q)^{-1} : matrice fondamentale
      - b = N · 1 : temps moyen d'absorption

    Retour : dict avec prob_goal, prob_fail, expected_steps
    """
    n = len(path)
    if n < 2:
        return {"prob_goal": 0.0, "prob_fail": 1.0, "expected_steps": float("inf")}

    # approximation via simulation de la distribution limite
    prob_goal = goal_probability(path, epsilon, steps=100)
    prob_fail = 1.0 - prob_goal

    # temps moyen estimé : longueur du chemin / (1 - ε)
    if epsilon < 1.0:
        expected_steps = (n - 1) / (1 - epsilon * 0.5)
    else:
        expected_steps = float("inf")

    return {
        "prob_goal"     : round(prob_goal, 4),
        "prob_fail"     : round(prob_fail, 4),
        "expected_steps": round(expected_steps, 2),
        "path_length"   : n - 1,
        "epsilon"       : epsilon,
    }


# ─────────────────────────────────────────────
#  SIMULATION MONTE-CARLO
# ─────────────────────────────────────────────

def monte_carlo(path: list,
                grid_n: int,
                obstacles: set,
                epsilon: float,
                n_simulations: int = 200,
                max_steps_factor: int = 6,
                record_trajs: int = 30) -> dict:
    """
    Simule N trajectoires de l'agent suivant la politique du chemin,
    avec déviation stochastique de paramètre ε.

    Paramètres
    ----------
    path             : chemin planifié (liste de tuples)
    grid_n           : taille de la grille
    obstacles        : ensemble des obstacles
    epsilon          : taux de déviation
    n_simulations    : nombre de trajectoires simulées
    max_steps_factor : max_steps = len(path) * max_steps_factor
    record_trajs     : nombre de trajectoires à enregistrer

    Retour
    ------
    dict :
        prob_goal     : P̂(atteindre GOAL)
        prob_fail     : P̂(échec)
        avg_steps     : nombre moyen de pas (succès uniquement)
        std_steps     : écart-type du nombre de pas
        trajectories  : liste des premières trajectoires enregistrées
        n_success     : nombre de succès
        n_fail        : nombre d'échecs
        n_simulations : N total
    """
    if not path or len(path) < 2:
        return {"prob_goal": 0.0, "prob_fail": 1.0,
                "avg_steps": 0, "std_steps": 0,
                "trajectories": [], "n_success": 0,
                "n_fail": n_simulations, "n_simulations": n_simulations}

    goal        = tuple(path[-1])
    max_steps   = len(path) * max_steps_factor
    path_index  = {tuple(p): i for i, p in enumerate(path)}

    recorded    = []
    n_success   = 0
    steps_list  = []

    # directions latérales perpendiculaires à (dr, dc)
    def lateral_dirs(dr, dc):
        return [(-1, dc), (1, dc)] if dr == 0 else [(dr, -1), (dr, 1)]

    for sim in range(n_simulations):
        pos  = list(path[0])
        traj = [tuple(pos)]

        for step in range(max_steps):
            key = tuple(pos)
            pi  = path_index.get(key, -1)

            if pi < 0 or pi >= len(path) - 1:
                break   # hors chemin → FAIL

            intended = path[pi + 1]
            dr = intended[0] - pos[0]
            dc = intended[1] - pos[1]

            if random.random() < epsilon:
                # déviation latérale
                lr, lc = random.choice(lateral_dirs(dr, dc))
                nr, nc = pos[0] + lr, pos[1] + lc
                if (0 <= nr < grid_n and 0 <= nc < grid_n
                        and (nr, nc) not in obstacles):
                    pos = [nr, nc]
                # sinon reste sur place
            else:
                pos = list(intended)

            traj.append(tuple(pos))

            if tuple(pos) == goal:
                n_success  += 1
                steps_list.append(step + 1)
                break

        if sim < record_trajs:
            recorded.append(traj)

    prob_goal = n_success / n_simulations
    avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0
    # écart-type
    if len(steps_list) > 1:
        mean = avg_steps
        std  = (sum((x - mean)**2 for x in steps_list) / len(steps_list)) ** 0.5
    else:
        std = 0.0

    return {
        "prob_goal"    : round(prob_goal, 4),
        "prob_fail"    : round(1 - prob_goal, 4),
        "avg_steps"    : round(avg_steps, 2),
        "std_steps"    : round(std, 2),
        "trajectories" : recorded,
        "n_success"    : n_success,
        "n_fail"       : n_simulations - n_success,
        "n_simulations": n_simulations,
    }


# ─────────────────────────────────────────────
#  TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # chemin de test simple
    path = [(0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(3,4),(3,5),(4,5),(5,5)]
    N    = 6
    OBS  = set()
    EPS  = 0.10

    print("=" * 55)
    print("  Test — markov.py")
    print("=" * 55)

    # matrice P
    P = build_transition_matrix(path, N, OBS, EPS)
    ok = verify_stochastic(P)
    print(f"\n  Matrice P  :  {len(P)} états  |  stochastique = {ok}")

    # évolution π^n
    hist = evolve_distribution(path, EPS, 30)
    print(f"\n  π⁽ⁿ⁾ = π⁽⁰⁾·Pⁿ  →  P(GOAL) à n=30 : {hist[-1]*100:.1f}%")

    # P(GOAL) vs ε
    probs = goal_probability_vs_epsilon(path)
    print("\n  P(GOAL) vs ε :")
    for e, p in probs.items():
        bar = "█" * int(p * 30)
        print(f"    ε={e:.2f}  {bar:<30}  {p*100:.1f}%")

    # absorption
    ab = absorption_analysis(path, EPS)
    print(f"\n  Absorption  :  P(GOAL)={ab['prob_goal']}  "
          f"P(FAIL)={ab['prob_fail']}  E[T]={ab['expected_steps']} pas")

    # Monte-Carlo
    mc = monte_carlo(path, N, OBS, EPS, n_simulations=500)
    print(f"\n  Monte-Carlo (N={mc['n_simulations']}) :")
    print(f"    P̂(GOAL)  = {mc['prob_goal']*100:.1f}%")
    print(f"    P̂(FAIL)  = {mc['prob_fail']*100:.1f}%")
    print(f"    Moy. pas = {mc['avg_steps']}  ±  {mc['std_steps']}")