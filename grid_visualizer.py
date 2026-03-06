"""
grid_visualizer.py  (Version Française)
────────────────────────────────────────────────────────────
Génération de visualisations haute-qualité pour la grille 2D.
Tous les textes, titres et légendes sont en français.
Corrections : débordements de texte, chevauchements, tableaux.
────────────────────────────────────────────────────────────
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from astar import run_all, build_itinerary, manhattan
from markov import (goal_probability_vs_epsilon, monte_carlo,
                    build_transition_matrix, absorption_analysis)
import os

# ─── Palette globale ────────────────────────────────────────
COLORS = {
    "astar":    "#6B63FF",
    "ucs":      "#3B82F6",
    "greedy":   "#F59E0B",
    "start":    "#10B981",
    "goal":     "#EF4444",
    "obstacle": "#1F2937",
    "explored": "#DDD6FE",
}
ALGO_LABELS = {"astar": "A*", "ucs": "UCS (CUC)", "greedy": "Glouton"}
ALGO_LIST   = ["astar", "ucs", "greedy"]


# ════════════════════════════════════════════════════════════
#  1. COMPARAISON DES GRILLES
# ════════════════════════════════════════════════════════════

def create_grid_visualization(grid_n=15, obstacles=None, filename="grilles.png"):
    """
    Grille de référence + 3 grilles résultats (A*, UCS, Glouton).
    """
    if obstacles is None:
        obstacles = set()

    start   = (0, 0)
    goal    = (grid_n - 1, grid_n - 1)
    results = run_all(start, goal, grid_n, obstacles)

    # Layout : 2 lignes × 3 colonnes + espace pour les titres
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#F8F9FA")

    # Titre principal
    fig.suptitle(
        f"Comparaison A* vs UCS vs Glouton — Grille {grid_n}×{grid_n}  "
        f"({len(obstacles)} obstacles)",
        fontsize=17, fontweight="bold", y=0.97
    )

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.05, right=0.97, top=0.90, bottom=0.04)

    # ── grille de référence ──────────────────────────────────
    ax_ref = fig.add_subplot(gs[0, 0])
    _draw_grid(ax_ref, grid_n, obstacles, start, goal)
    ax_ref.set_title("Grille de référence", fontsize=12, fontweight="bold", pad=8)

    # ── statistiques textuelles ──────────────────────────────
    ax_stats = fig.add_subplot(gs[0, 1:])
    ax_stats.axis("off")

    lines = ["COMPARAISON DES ALGORITHMES\n" + "─"*42 + "\n"]
    for algo in ALGO_LIST:
        res = results[algo]
        lbl = ALGO_LABELS[algo]
        if res["path"]:
            lines.append(
                f"{lbl}\n"
                f"  Coût du chemin   : {res['cost']}\n"
                f"  Nœuds développés : {res['nodes']}\n"
                f"  Temps d'exéc.    : {res['time_ms']} ms\n"
                f"  Longueur chemin  : {len(res['path'])}\n"
            )
        else:
            lines.append(f"{lbl} : Aucun chemin trouvé\n")

    ax_stats.text(
        0.03, 0.97, "\n".join(lines),
        transform=ax_stats.transAxes,
        fontfamily="monospace", fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#EEF0FF", alpha=0.9)
    )

    # ── 3 grilles résultats ──────────────────────────────────
    for idx, algo in enumerate(ALGO_LIST):
        ax = fig.add_subplot(gs[1, idx])
        res = results[algo]
        color = COLORS[algo]
        label = ALGO_LABELS[algo]

        _draw_grid(ax, grid_n, obstacles, start, goal,
                   explored=res["explored"], path=res["path"],
                   path_color=color)

        if res["path"]:
            subtitle = (f"Coût: {res['cost']}  |  Nœuds: {res['nodes']}  "
                        f"|  Temps: {res['time_ms']} ms")
        else:
            subtitle = "Aucun chemin trouvé"

        ax.set_title(f"{label}\n{subtitle}",
                     fontsize=10, fontweight="bold", color=color, pad=6)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()
    return results


def _draw_grid(ax, grid_n, obstacles, start, goal,
               explored=None, path=None, path_color="#6B63FF"):
    ax.set_xlim(-0.5, grid_n - 0.5)
    ax.set_ylim(grid_n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Colonne", fontsize=8)
    ax.set_ylabel("Ligne",   fontsize=8)

    # obstacles
    for obs in obstacles:
        rect = patches.Rectangle(
            (obs[1] - 0.45, obs[0] - 0.45), 0.9, 0.9,
            linewidth=0, facecolor=COLORS["obstacle"], alpha=0.85
        )
        ax.add_patch(rect)

    # nœuds explorés
    if explored:
        for exp in explored:
            if exp != start and exp != goal and exp not in obstacles:
                rect = patches.Rectangle(
                    (exp[1] - 0.38, exp[0] - 0.38), 0.76, 0.76,
                    linewidth=0, facecolor=COLORS["explored"], alpha=0.55
                )
                ax.add_patch(rect)

    # chemin
    if path:
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, color=path_color, linewidth=2.5, alpha=0.85, zorder=2)
        for p in path[1:-1]:
            ax.plot(p[1], p[0], "o", color=path_color,
                    markersize=4, alpha=0.6, zorder=2)

    ax.plot(start[1], start[0], "o", color=COLORS["start"],
            markersize=11, zorder=4, label="Départ")
    ax.plot(goal[1],  goal[0],  "s", color=COLORS["goal"],
            markersize=11, zorder=4, label="But")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)


# ════════════════════════════════════════════════════════════
#  2. MÉTRIQUES DE PERFORMANCE
# ════════════════════════════════════════════════════════════

def create_metrics_comparison(results, filename="metriques.png"):
    """
    Graphique 2×2 : nœuds développés, coût, temps, efficacité.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Comparaison des Métriques de Performance",
                 fontsize=16, fontweight="bold", y=0.98)

    labels = [ALGO_LABELS[a] for a in ALGO_LIST]
    cols   = [COLORS[a]      for a in ALGO_LIST]

    nodes = [results[a]["nodes"]                                    for a in ALGO_LIST]
    costs = [results[a]["cost"] if results[a]["path"] else 0        for a in ALGO_LIST]
    times = [float(results[a]["time_ms"])                           for a in ALGO_LIST]

    def _bar(ax, vals, ylabel, title, fmt="{:.0f}"):
        bars = ax.bar(labels, vals, color=cols, alpha=0.82,
                      edgecolor="black", linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title,   fontsize=12, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    fmt.format(v),
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(vals) * 1.18)

    _bar(axes[0, 0], nodes, "Nœuds développés",
         "Expansion de l'espace de recherche")
    _bar(axes[0, 1], costs, "Coût du chemin",
         "Qualité de la solution", "{:.0f}")
    _bar(axes[1, 0], times, "Temps (ms)",
         "Vitesse de calcul", "{:.3f}")

    # efficacité vs UCS
    ax = axes[1, 1]
    ucs_n = nodes[ALGO_LIST.index("ucs")]
    eff   = [ucs_n / n if n > 0 else 0 for n in nodes]
    bars  = ax.bar(labels, eff, color=cols, alpha=0.82,
                   edgecolor="black", linewidth=1.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=2,
               label="Référence UCS")
    ax.set_ylabel("Ratio d'efficacité (UCS = 1,0)",
                  fontsize=11, fontweight="bold")
    ax.set_title("Efficacité de recherche vs UCS",
                 fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    for bar, v in zip(bars, eff):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.2f}×",
                ha="center", va="bottom",
                fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(eff) * 1.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════
#  3. ANALYSE D'INCERTITUDE
# ════════════════════════════════════════════════════════════

def create_uncertainty_analysis(results, grid_n, obstacles,
                                epsilon_values=None,
                                filename="incertitude.png"):
    """
    Courbes P(GOAL) vs ε + tableau résumé (sans chevauchement).
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Analyse de Robustesse : P(Atteindre le But) vs Incertitude d'Action (ε)",
        fontsize=15, fontweight="bold", y=1.02
    )

    algo_probs = {}
    for algo in ALGO_LIST:
        if results[algo]["path"]:
            algo_probs[algo] = list(
                goal_probability_vs_epsilon(results[algo]["path"], epsilon_values).values()
            )
        else:
            algo_probs[algo] = [0.0] * len(epsilon_values)

    # ── tableau (axes[0]) ────────────────────────────────────
    ax = axes[0]
    ax.axis("off")

    header = ["Algorithme"] + [f"ε={e:.2f}" for e in epsilon_values]
    rows   = []
    for algo in ALGO_LIST:
        rows.append([ALGO_LABELS[algo]] +
                    [f"{p*100:.0f}%" for p in algo_probs[algo]])

    col_w = [0.16] + [0.12] * len(epsilon_values)
    tbl = ax.table(cellText=rows, colLabels=header,
                   cellLoc="center", loc="center",
                   colWidths=col_w)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.6)

    # style en-tête
    for j in range(len(header)):
        c = tbl[(0, j)]
        c.set_facecolor("#3B3B8C")
        c.set_text_props(weight="bold", color="white")

    # style lignes
    for i, algo in enumerate(ALGO_LIST, 1):
        tbl[(i, 0)].set_facecolor(COLORS[algo])
        tbl[(i, 0)].set_text_props(weight="bold", color="white")
        for j, p in enumerate(algo_probs[algo], 1):
            tbl[(i, j)].set_facecolor(
                "#D1FAE5" if p >= 0.80 else
                "#FEF3C7" if p >= 0.50 else
                "#FEE2E2"
            )

    ax.set_title("Tableau : P(But) par algorithme et par ε",
                 fontsize=11, fontweight="bold", pad=14)

    # ── courbes (axes[1]) ────────────────────────────────────
    ax = axes[1]
    for algo in ALGO_LIST:
        ax.plot(epsilon_values, algo_probs[algo],
                marker="o", linewidth=2.5, markersize=8,
                label=ALGO_LABELS[algo],
                color=COLORS[algo], alpha=0.85)

    ax.set_xlabel("Probabilité de déviation (ε)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("P(Atteindre le but)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Probabilité de succès vs incertitude",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════
#  4. TABLEAU D'ITINÉRAIRE
# ════════════════════════════════════════════════════════════

def create_itinerary_table(results, goal, filename="itineraires.png"):
    """
    Tableaux itinéraires pas-à-pas pour les 3 algorithmes.
    Corrections : police réduite, hauteur suffisante, pas de débordement.
    """
    MAX_ROWS = 18   # lignes max affichées par algorithme

    fig, axes = plt.subplots(1, 3, figsize=(22, 12))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Itinéraires Détaillés — Progression Pas-à-Pas",
                 fontsize=16, fontweight="bold", y=0.99)

    headers = ["#", "Cellule", "g", "h", "f", "Dir.", "Type"]
    col_w   = [0.07, 0.17, 0.08, 0.08, 0.08, 0.14, 0.13]

    for ax, algo in zip(axes, ALGO_LIST):
        ax.axis("off")
        color = COLORS[algo]
        label = ALGO_LABELS[algo]
        path  = results[algo]["path"]

        if not path:
            ax.text(0.5, 0.5, f"{label}\nAucun chemin trouvé",
                    ha="center", va="center", fontsize=13,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round", facecolor="#FEE2E2"))
            continue

        itin = build_itinerary(path, goal)[:MAX_ROWS]

        rows = [
            [str(r["step"]),
             f"({r['cell'][0]},{r['cell'][1]})",
             str(r["g"]), str(r["h"]), str(r["f"]),
             r["dir"], r["type"]]
            for r in itin
        ]

        tbl = ax.table(cellText=rows, colLabels=headers,
                       cellLoc="center", loc="center",
                       colWidths=col_w)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.65)

        # en-tête
        for j in range(len(headers)):
            c = tbl[(0, j)]
            c.set_facecolor(color)
            c.set_text_props(weight="bold", color="white")

        # alternance lignes
        type_colors = {"START": "#D1FAE5", "GOAL": "#FEE2E2", "PATH": "white"}
        for i, row in enumerate(rows, 1):
            bg = type_colors.get(row[-1], "#F3F4F6" if i % 2 == 0 else "white")
            for j in range(len(headers)):
                tbl[(i, j)].set_facecolor(bg)

        shown = len(itin)
        total = len(path) - 1
        extra = f" (+{total - shown} masqués)" if total > shown else ""

        ax.set_title(
            f"{label}  —  Coût total : {results[algo]['cost']}  "
            f"|  {shown}{extra} étapes affichées",
            fontsize=10, fontweight="bold", color=color, pad=12
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════
#  5. TABLEAU D'INCERTITUDE ÉTENDU
# ════════════════════════════════════════════════════════════

def create_uncertainty_table(results, grid_n, obstacles,
                              epsilon_values=None,
                              filename="tableau_incertitude.png"):
    """
    Un tableau par algorithme : Markov vs Monte-Carlo, erreur, stats.
    Corrections : hauteur de figure adaptative, police cohérente.
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    n_eps   = len(epsilon_values)
    row_h   = 0.38          # hauteur par ligne de données (inches)
    header_h = 1.4          # espace titre + en-tête
    pad_h    = 0.5          # espace bas
    fig_h    = len(ALGO_LIST) * (n_eps * row_h + header_h + pad_h)

    fig, axes = plt.subplots(len(ALGO_LIST), 1, figsize=(20, max(fig_h, 14)))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Analyse d'Incertitude — P(But) Markov vs Monte-Carlo par Algorithme",
        fontsize=15, fontweight="bold", y=1.01
    )

    headers = ["ε", "P(But) Markov", "P(But) MC",
               "|Erreur|", "Moy. Pas", "Éc.-Type", "Succès", "Échecs"]
    col_w   = [0.08, 0.13, 0.12, 0.10, 0.11, 0.11, 0.09, 0.09]

    for ax, algo in zip(axes, ALGO_LIST):
        ax.axis("off")
        color = COLORS[algo]
        path  = results[algo]["path"]

        rows = []
        for eps in epsilon_values:
            if path:
                mp = goal_probability_vs_epsilon(path, [eps])[eps]
                mc = monte_carlo(path, grid_n, obstacles, eps,
                                 n_simulations=300, max_steps_factor=6)
                rows.append([
                    f"ε={eps:.2f}",
                    f"{mp*100:.1f}%",
                    f"{mc['prob_goal']*100:.1f}%",
                    f"{abs(mp - mc['prob_goal'])*100:.1f}%",
                    f"{mc['avg_steps']:.1f}",
                    f"{mc['std_steps']:.1f}",
                    str(mc["n_success"]),
                    str(mc["n_fail"]),
                ])
            else:
                rows.append([f"ε={eps:.2f}"] + ["N/D"] * 7)

        scale_y = 1.9 + max(0, (18 - n_eps) * 0.05)   # scaling dynamique
        tbl = ax.table(cellText=rows, colLabels=headers,
                       cellLoc="center", loc="center",
                       colWidths=col_w)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, scale_y)

        # en-tête
        for j in range(len(headers)):
            c = tbl[(0, j)]
            c.set_facecolor(color)
            c.set_text_props(weight="bold", color="white")

        # couleurs lignes selon P(But)
        for i, eps in enumerate(epsilon_values, 1):
            p = goal_probability_vs_epsilon(path, [eps])[eps] if path else 0.0
            bg = "#D1FAE5" if p >= 0.80 else "#FEF3C7" if p >= 0.50 else "#FEE2E2"
            for j in range(len(headers)):
                tbl[(i, j)].set_facecolor(bg if j > 0 else "#F3F4F6")

        cost  = results[algo]["cost"]  if path else "N/D"
        nodes = results[algo]["nodes"] if path else "N/D"
        ax.set_title(
            f"{ALGO_LABELS[algo]}  —  Coût chemin : {cost}  "
            f"|  Nœuds développés : {nodes}",
            fontsize=12, fontweight="bold", color=color, pad=14
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════
#  6. MATRICE STOCHASTIQUE
# ════════════════════════════════════════════════════════════

def create_stochastic_matrix_viz(results, grid_n, obstacles,
                                  epsilon=0.10,
                                  filename="matrice_stochastique.png"):
    """
    Pour chaque algorithme : heatmap P, vérification sommes-lignes,
    tableau des transitions non-nulles.
    Corrections : titres lisibles, annotations sans débordement.
    """
    fig = plt.figure(figsize=(22, 8 * len(ALGO_LIST)))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"Matrice de Transition Stochastique P  (ε = {epsilon:.2f})\n"
        "Chaque ligne somme à 1 — P[i,j] = probabilité de passer de l'état i à j",
        fontsize=14, fontweight="bold", y=1.005
    )

    for row_idx, algo in enumerate(ALGO_LIST):
        color = COLORS[algo]
        label = ALGO_LABELS[algo]
        path  = results[algo]["path"]

        ax_heat = fig.add_subplot(len(ALGO_LIST), 3, row_idx * 3 + 1)
        ax_row  = fig.add_subplot(len(ALGO_LIST), 3, row_idx * 3 + 2)
        ax_tbl  = fig.add_subplot(len(ALGO_LIST), 3, row_idx * 3 + 3)

        if not path or len(path) < 2:
            for ax in (ax_heat, ax_row, ax_tbl):
                ax.axis("off")
                ax.text(0.5, 0.5, f"{label} : Aucun chemin",
                        ha="center", va="center", transform=ax.transAxes)
            continue

        P = build_transition_matrix(path, grid_n, obstacles, epsilon)
        all_st  = list(P.keys())
        idx_map = {s: i for i, s in enumerate(all_st)}
        n       = len(all_st)
        P_mat   = np.zeros((n, n))
        for i, si in enumerate(all_st):
            for sj, p in P[si].items():
                if sj in idx_map:
                    P_mat[i, idx_map[sj]] = p
        row_sums = P_mat.sum(axis=1)

        # ── heatmap ──────────────────────────────────────────
        im = ax_heat.imshow(P_mat, cmap="YlOrRd", aspect="auto",
                            vmin=0, vmax=1, interpolation="nearest")
        ax_heat.set_title(f"{label} — Matrice P ({n}×{n} états)",
                          fontsize=11, fontweight="bold", color=color, pad=8)
        ax_heat.set_xlabel("Vers état j", fontsize=9)
        ax_heat.set_ylabel("Depuis état i", fontsize=9)
        cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cb.set_label("Probabilité", fontsize=9)

        if n <= 14:
            for i in range(n):
                for j in range(n):
                    if P_mat[i, j] > 0.01:
                        ax_heat.text(j, i, f"{P_mat[i,j]:.2f}",
                                     ha="center", va="center",
                                     fontsize=6.5, color="black")

        # ── vérif sommes lignes ──────────────────────────────
        ax_row.bar(range(n), row_sums, color=color, alpha=0.7,
                   edgecolor="black", linewidth=0.5)
        ax_row.axhline(1.0, color="red", linestyle="--", linewidth=2,
                       label="Somme attendue = 1")
        ax_row.set_ylim(0.80, 1.20)
        ax_row.set_title(
            "Vérification des sommes de lignes\n(chaque ligne doit valoir 1,0)",
            fontsize=10, fontweight="bold", color=color, pad=8
        )
        ax_row.set_xlabel("Indice d'état i", fontsize=9)
        ax_row.set_ylabel("Σⱼ P[i,j]", fontsize=9)
        ax_row.legend(fontsize=9)
        ax_row.grid(axis="y", alpha=0.3)
        max_dev = np.max(np.abs(row_sums - 1.0))
        ax_row.text(0.98, 0.95, f"Écart max : {max_dev:.2e}",
                    transform=ax_row.transAxes,
                    ha="right", va="top", fontsize=9, color="darkred",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # ── tableau transitions non-nulles ───────────────────
        ax_tbl.axis("off")
        tbl_rows = []
        for si in all_st:
            if si in ("GOAL", "FAIL"):
                continue
            outgoing = sorted(P[si].items(), key=lambda x: -x[1])
            for sj, p in outgoing[:3]:
                tbl_rows.append([
                    str(si), str(sj), f"{p:.4f}",
                    "✓" if abs(sum(P[si].values()) - 1) < 1e-9 else "✗"
                ])
            if len(tbl_rows) >= 16:
                break

        if tbl_rows:
            th = ["État i", "État j", "P(i→j)", "OK"]
            t  = ax_tbl.table(cellText=tbl_rows, colLabels=th,
                               cellLoc="center", loc="center",
                               colWidths=[0.28, 0.28, 0.24, 0.16])
            t.auto_set_font_size(False)
            t.set_fontsize(8.5)
            t.scale(1, 1.75)
            for j in range(len(th)):
                c = t[(0, j)]
                c.set_facecolor(color)
                c.set_text_props(weight="bold", color="white")
            for r in range(1, len(tbl_rows) + 1):
                bg = "#EEF4FF" if r % 2 == 0 else "white"
                for j in range(len(th)):
                    t[(r, j)].set_facecolor(bg)
                ok = t[(r, 3)]
                ok.set_text_props(
                    color="green" if tbl_rows[r-1][3] == "✓" else "red"
                )

        ax_tbl.set_title(
            f"Transitions non-nulles (top 3 / état, 16 max)\n"
            f"Total états : {n}  |  ε = {epsilon:.2f}",
            fontsize=10, fontweight="bold", color=color, pad=8
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Sauvegardé : {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════

def generate_all_visualizations(grid_n=15, obstacle_percentage=20,
                                 save_dir="resultats_grille"):
    """Génère toutes les visualisations et retourne les résultats."""
    import random
    os.makedirs(save_dir, exist_ok=True)

    obstacles = set()
    goal = (grid_n - 1, grid_n - 1)
    for r in range(grid_n):
        for c in range(grid_n):
            if (r, c) != (0, 0) and (r, c) != goal:
                if random.random() < obstacle_percentage / 100:
                    obstacles.add((r, c))

    print(f"\n{'='*60}")
    print(f"Génération : grille {grid_n}×{grid_n} ({len(obstacles)} obstacles)")
    print(f"{'='*60}\n")

    results = create_grid_visualization(
        grid_n, obstacles, os.path.join(save_dir, "01_comparaison_grilles.png"))
    create_metrics_comparison(
        results,            os.path.join(save_dir, "02_metriques.png"))
    create_uncertainty_analysis(
        results, grid_n, obstacles,
        filename=          os.path.join(save_dir, "03_incertitude.png"))
    create_itinerary_table(
        results, goal,      os.path.join(save_dir, "04_itineraires.png"))
    create_uncertainty_table(
        results, grid_n, obstacles,
        filename=          os.path.join(save_dir, "05_tableau_incertitude.png"))
    create_stochastic_matrix_viz(
        results, grid_n, obstacles,
        filename=          os.path.join(save_dir, "06_matrice_stochastique.png"))

    print(f"\n✓ Toutes les visualisations sauvegardées dans : {save_dir}/")
    return results


if __name__ == "__main__":
    results = generate_all_visualizations(grid_n=15, obstacle_percentage=20)

    print("\nRÉSUMÉ")
    print("=" * 60)
    for algo in ALGO_LIST:
        lbl = ALGO_LABELS[algo]
        res = results[algo]
        if res["path"]:
            print(f"{lbl:12s} | Coût: {res['cost']:3d} | "
                  f"Nœuds: {res['nodes']:4d} | Temps: {res['time_ms']:>8.3f} ms")
        else:
            print(f"{lbl:12s} | Aucun chemin trouvé")
    print("=" * 60)