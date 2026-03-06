"""
markov_analysis.py  (Version Française)
────────────────────────────────────────────────────────────
Analyse avancée des Chaînes de Markov avec visualisations :
  - Matrice de transition (heatmap)
  - Comparaison Markov vs Monte-Carlo
  - Analyse des probabilités d'absorption
  - Classement de robustesse
Corrections : pas de chevauchement texte/graphique, police adaptée.
────────────────────────────────────────────────────────────
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astar import run_all
from markov import (build_transition_matrix, verify_stochastic,
                    goal_probability_vs_epsilon, absorption_analysis,
                    monte_carlo, evolve_distribution)
import os

# ─── Constantes partagées ────────────────────────────────────
ALGO_LIST   = ["astar", "ucs", "greedy"]
ALGO_LABELS = {"astar": "A*", "ucs": "UCS (CUC)", "greedy": "Glouton"}
COLORS      = {"astar": "#6B63FF", "ucs": "#3B82F6", "greedy": "#F59E0B"}


# ════════════════════════════════════════════════════════════
#  1. MATRICE DE TRANSITION — heatmap + connectivité
# ════════════════════════════════════════════════════════════

def visualize_transition_matrix(path, grid_n, obstacles, epsilon,
                                 filename="matrice_transition.png"):
    """
    Heatmap de P et graphe de connectivité des états.
    """
    P = build_transition_matrix(path, grid_n, obstacles, epsilon)
    if not P:
        print("  ⚠ Impossible de construire P (chemin vide)")
        return

    states    = list(P.keys())
    idx_map   = {s: i for i, s in enumerate(states)}
    n         = len(states)
    P_mat     = np.zeros((n, n))
    for i, si in enumerate(states):
        for sj, p in P[si].items():
            if sj in idx_map:
                P_mat[i, idx_map[sj]] = p

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"Matrice de Transition Markov  (ε={epsilon:.2f}, longueur chemin={len(path)-1})",
        fontsize=14, fontweight="bold", y=1.02
    )

    # ── heatmap ──────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(P_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Vers état j", fontsize=11, fontweight="bold")
    ax.set_ylabel("Depuis état i", fontsize=11, fontweight="bold")
    ax.set_title("Probabilités de transition P[i,j]",
                 fontsize=12, fontweight="bold")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Probabilité", fontsize=10, fontweight="bold")

    if n > 15:
        skip = max(1, n // 10)
        ax.set_xticks(range(0, n, skip))
        ax.set_yticks(range(0, n, skip))

    # ── connectivité ─────────────────────────────────────────
    ax = axes[1]
    strong = np.sum(P_mat > 0.1, axis=1)
    sc = ax.scatter(range(n), strong, s=80, c=strong,
                    cmap="viridis", alpha=0.75, edgecolors="black", linewidth=1)
    plt.colorbar(sc, ax=ax, label="Connexions sortantes fortes")
    ax.set_xlabel("Indice d'état", fontsize=11, fontweight="bold")
    ax.set_ylabel("Connexions sortantes (p > 0,1)",
                  fontsize=11, fontweight="bold")
    ax.set_title("Connectivité des états",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, np.max(strong) + 1.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════
#  2. MARKOV VS MONTE-CARLO
# ════════════════════════════════════════════════════════════

def compare_markov_vs_montecarlo(results, grid_n, obstacles,
                                  epsilon_values=None,
                                  filename="markov_vs_monte_carlo.png"):
    """
    4 sous-graphiques : analytique, MC, erreur de convergence,
    et classement de robustesse (texte dans un cadre dédié, sans overlap).
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Chaînes de Markov : Analytique vs Monte-Carlo",
        fontsize=16, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.93, bottom=0.06)

    # ── pré-calcul ────────────────────────────────────────────
    markov_data = {}
    mc_data     = {}
    for algo in ALGO_LIST:
        path = results[algo]["path"]
        if path:
            markov_data[algo] = list(
                goal_probability_vs_epsilon(path, epsilon_values).values()
            )
            mc_data[algo] = []
            for eps in epsilon_values:
                mc = monte_carlo(path, grid_n, obstacles, eps,
                                 n_simulations=500, max_steps_factor=6)
                mc_data[algo].append(mc["prob_goal"])
        else:
            markov_data[algo] = [0.0] * len(epsilon_values)
            mc_data[algo]     = [0.0] * len(epsilon_values)

    # ── analytique Markov ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for algo in ALGO_LIST:
        ax.plot(epsilon_values, markov_data[algo],
                marker="o", linewidth=2.5, markersize=8,
                label=ALGO_LABELS[algo], color=COLORS[algo], alpha=0.85)
    ax.set_xlabel("ε (probabilité de déviation)", fontsize=11, fontweight="bold")
    ax.set_ylabel("P(But)", fontsize=11, fontweight="bold")
    ax.set_title("Analytique : π⁽ⁿ⁾ = π⁽⁰⁾·Pⁿ",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=10)

    # ── Monte-Carlo ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for algo in ALGO_LIST:
        ax.plot(epsilon_values, mc_data[algo],
                marker="s", linewidth=2.5, markersize=8,
                label=ALGO_LABELS[algo], color=COLORS[algo], alpha=0.85)
    ax.set_xlabel("ε (probabilité de déviation)", fontsize=11, fontweight="bold")
    ax.set_ylabel("P̂(But)", fontsize=11, fontweight="bold")
    ax.set_title("Empirique : Simulation Monte-Carlo (N=500)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=10)

    # ── erreur de convergence ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for algo in ALGO_LIST:
        errors = [abs(m - mc) * 100
                  for m, mc in zip(markov_data[algo], mc_data[algo])]
        ax.plot(epsilon_values, errors,
                marker="^", linewidth=2.5, markersize=8,
                label=ALGO_LABELS[algo], color=COLORS[algo], alpha=0.85)
    ax.axhline(5, color="red", linestyle="--", linewidth=2,
               label="Seuil 5 %", alpha=0.6)
    ax.set_xlabel("ε (probabilité de déviation)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Erreur absolue (%)", fontsize=11, fontweight="bold")
    ax.set_title("Convergence : |Markov − MC|",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # ── classement de robustesse ──────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    ranking = sorted(
        [(ALGO_LABELS[a], np.mean(markov_data[a]), COLORS[a])
         for a in ALGO_LIST],
        key=lambda x: -x[1]
    )

    lines = ["CLASSEMENT DE ROBUSTESSE\n" + "─"*38 + "\n",
             f"P(But) moyen sur ε ∈ [0 ; 0,30] :\n"]
    for rank, (lbl, prob, _) in enumerate(ranking, 1):
        bar = "█" * int(prob * 25)
        lines.append(f"  {rank}. {lbl:12s}  {prob*100:5.1f}%  {bar}")

    lines += [
        "\n─"*38,
        "\nInterprétation :",
        "• Prob. moyenne plus haute = plus robuste",
        "• Chemins plus courts → meilleure robustesse",
        "• La dégradation croît avec longueur × ε"
    ]

    ax.text(0.04, 0.96, "\n".join(lines),
            transform=ax.transAxes,
            fontfamily="monospace", fontsize=10.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#F0F4FF", alpha=0.92))
    ax.set_title("Synthèse de robustesse",
                 fontsize=12, fontweight="bold", pad=10)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════
#  3. ANALYSE D'ABSORPTION
# ════════════════════════════════════════════════════════════

def absorption_comparison(results, grid_n, obstacles, epsilon,
                           filename="absorption.png"):
    """
    Barres : P(But) vs P(Échec) + longueur chemin vs temps espéré.
    Corrections : axes doubles lisibles, étiquettes de valeurs claires.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(f"Analyse d'Absorption  (ε = {epsilon:.2f})",
                 fontsize=14, fontweight="bold", y=1.02)

    labels  = [ALGO_LABELS[a] for a in ALGO_LIST]
    colors  = [COLORS[a]      for a in ALGO_LIST]

    absorbs = {}
    for algo in ALGO_LIST:
        path = results[algo]["path"]
        absorbs[algo] = absorption_analysis(path, epsilon) if path else {
            "prob_goal": 0.0, "prob_fail": 1.0,
            "expected_steps": float("inf"), "path_length": 0
        }

    # ── probabilités d'absorption ─────────────────────────────
    ax  = axes[0]
    x   = np.arange(len(ALGO_LIST))
    w   = 0.35
    g   = [absorbs[a]["prob_goal"] for a in ALGO_LIST]
    f   = [absorbs[a]["prob_fail"] for a in ALGO_LIST]

    b1 = ax.bar(x - w/2, g, w, label="P(But)",
                color="#10B981", alpha=0.82, edgecolor="black", linewidth=1.5)
    b2 = ax.bar(x + w/2, f, w, label="P(Échec)",
                color="#EF4444", alpha=0.82, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Probabilité", fontsize=11, fontweight="bold")
    ax.set_title("Probabilités d'absorption", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=10)

    # ── longueur chemin + temps espéré ────────────────────────
    ax  = axes[1]
    lengths = [absorbs[a]["path_length"] for a in ALGO_LIST]
    etimes  = [absorbs[a]["expected_steps"] for a in ALGO_LIST]

    b1 = ax.bar(x - w/2, lengths, w, label="Longueur du chemin",
                color="#3B82F6", alpha=0.82, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Longueur du chemin (pas)",
                  fontsize=11, fontweight="bold", color="#3B82F6")
    ax.tick_params(axis="y", labelcolor="#3B82F6")

    ax2 = ax.twinx()
    b2  = ax2.bar(x + w/2, etimes, w, label="E[Temps absorption]",
                  color="#F59E0B", alpha=0.82, edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("Temps espéré (pas)",
                   fontsize=11, fontweight="bold", color="#F59E0B")
    ax2.tick_params(axis="y", labelcolor="#F59E0B")

    ax.set_title("Longueur chemin vs Temps d'absorption espéré",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=10, loc="upper left")

    # étiquettes valeurs
    for bar, v in zip(b1, lengths):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(lengths) * 0.02,
                f"{v}", ha="center", va="bottom",
                fontweight="bold", fontsize=10, color="#1E40AF")
    max_e = max(e for e in etimes if e < float("inf")) if any(e < float("inf") for e in etimes) else 1
    for bar, v in zip(b2, etimes):
        label_v = f"{v:.1f}" if v < float("inf") else "∞"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max_e * 0.02,
                 label_v, ha="center", va="bottom",
                 fontweight="bold", fontsize=10, color="#B45309")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def generate_markov_analysis(results, grid_n, obstacles,
                              save_dir="resultats_markov"):
    """Génère toutes les visualisations Markov."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("ANALYSE DES CHAÎNES DE MARKOV")
    print(f"{'='*60}\n")

    if results["astar"]["path"]:
        print("  Génération de la heatmap de la matrice P…")
        visualize_transition_matrix(
            results["astar"]["path"], grid_n, obstacles, 0.10,
            os.path.join(save_dir, "01_matrice_transition.png")
        )

    print("  Comparaison analytique vs Monte-Carlo…")
    compare_markov_vs_montecarlo(
        results, grid_n, obstacles,
        filename=os.path.join(save_dir, "02_markov_vs_monte_carlo.png")
    )

    print("  Calcul des probabilités d'absorption…")
    absorption_comparison(
        results, grid_n, obstacles, 0.10,
        filename=os.path.join(save_dir, "03_absorption.png")
    )

    print(f"\n✓ Toutes les analyses Markov sauvegardées dans : {save_dir}/\n")


if __name__ == "__main__":
    import random
    grid_n    = 15
    goal      = (grid_n - 1, grid_n - 1)
    obstacles = {
        (r, c)
        for r in range(grid_n) for c in range(grid_n)
        if (r, c) != (0, 0) and (r, c) != goal and random.random() < 0.20
    }
    results = run_all((0, 0), goal, grid_n, obstacles)
    generate_markov_analysis(results, grid_n, obstacles)