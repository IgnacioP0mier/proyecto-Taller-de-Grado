"""Espacio de trabajo alcanzable: nube de puntos (Monte Carlo), cobertura de las zonas de interés,
índice de singularidad y figuras. Genera archivos en resultados/."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from cinematica import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")


def cobertura(centros, standoff, beta, extra_tol=0.0):
    """Fracción de objetivos alcanzables (IK con límites y sin colisión) y menor índice de singularidad."""
    ok, smin, qs = 0, np.inf, []
    for m in centros:
        q = ik(pose_entrega(m, standoff) if standoff is not None else m, beta)
        if q is not None:
            ok += 1
            smin = min(smin, indice_singularidad(q))
            qs.append(q)
    qs = np.array(qs) if qs else np.zeros((0, 5))
    margen = float("nan")
    if len(qs):   # margen mínimo a los límites articulares [rad]
        margen = float(min((qs - Q_MIN).min(), (Q_MAX - qs).min()))
    return dict(total=len(centros), alcanzables=ok, porcentaje=100.0 * ok / len(centros),
                sing_min=float(smin) if ok else float("nan"), margen_limite_deg=float(np.degrees(margen)))


def cubo(centro, semi, n, plano_z=False):
    g = np.linspace(-semi, semi, n)
    if plano_z:
        return np.array([centro + [dx, dy, 0] for dx in g for dy in g])
    return np.array([centro + [dx, dy, dz] for dx in g for dy in g for dz in g])


def generar(n_mc=300000, seed=0, guardar=True):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(Q_MIN, Q_MAX, (n_mc, 5))
    P, b = fk_lote(Q)
    sobre_mesa = P[:, 2] >= Z_MESA                    # la punta no puede quedar bajo la mesa
    n_total = n_mc
    Q, P, b = Q[sobre_mesa], P[sobre_mesa], b[sobre_mesa]
    n_mc = len(P)
    res = dict(n_muestras=n_total, n_sobre_mesa=n_mc,
               alcance_radial_max=float(np.hypot(P[:, 0], P[:, 1]).max()),
               z_min=float(P[:, 2].min()), z_max=float(P[:, 2].max()))
    # --- cobertura de las zonas de interés ---
    res["boca_entrega"] = cobertura(cubo(BOCA_NOMINAL, VOLUMEN_BOCA, 9), STANDOFF_ENTREGA, BETA_TRANSPORTE)
    res["boca_traslado"] = cobertura(cubo(BOCA_NOMINAL, VOLUMEN_BOCA, 9), STANDOFF_TRASLADO, BETA_TRANSPORTE)
    res["plato"] = cobertura(cubo(PLATO, SEMI_PLATO, 9, plano_z=True), None, BETA_TOMA)
    # Cubo de ±VOLUMEN_BOCA a 3 inclinaciones de cuchara (tolerancia de nivel ±10°)
    res["boca_entrega_beta_pm10"] = {s: cobertura(cubo(BOCA_NOMINAL, VOLUMEN_BOCA, 7), STANDOFF_ENTREGA, s * 10 * D2R)["porcentaje"] for s in (-1, 0, 1)}
    res["boca_entrega_beta_pm10"] = {str(k): v for k, v in res["boca_entrega_beta_pm10"].items()}
    # singularidad sobre una muestra de la nube
    idx = rng.choice(n_mc, 3000, replace=False)
    sing = np.array([indice_singularidad(Q[i]) for i in idx])
    res["sing_nube_p5"] = float(np.percentile(sing, 5))
    if guardar:
        os.makedirs(OUT, exist_ok=True)
        sub = rng.choice(n_mc, 20000, replace=False)
        np.savetxt(os.path.join(OUT, "nube_espacio_trabajo.csv"), np.column_stack([P[sub], np.degrees(b[sub])]),
                   delimiter=",", header="x_mm,y_mm,z_mm,beta_deg", comments="", fmt="%.2f")
        _figuras(P, res)
        with open(os.path.join(OUT, "espacio_trabajo.json"), "w", encoding="utf8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def _figuras(P, res):
    sub = P[::15]
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], s=1, cmap="viridis", alpha=0.35)
    for c, semi, col, nom in [(BOCA_NOMINAL, VOLUMEN_BOCA, "r", "volumen de la boca"), (PLATO, SEMI_PLATO, "orange", "recipiente")]:
        ax.scatter(*c, color=col, s=40, label=nom)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Nube de puntos alcanzables (punta)")
    ax2 = fig.add_subplot(122)
    band = np.abs(P[:, 1]) < 15
    ax2.scatter(np.hypot(P[band, 0], P[band, 1]) * np.sign(P[band, 0]), P[band, 2], s=1, alpha=0.2, color="steelblue")
    cub = cubo(BOCA_NOMINAL, VOLUMEN_BOCA, 2)
    ax2.add_patch(plt.Rectangle((BOCA_NOMINAL[0] - VOLUMEN_BOCA, BOCA_NOMINAL[2] - VOLUMEN_BOCA), 2 * VOLUMEN_BOCA, 2 * VOLUMEN_BOCA, fill=False, ec="r", lw=2, label="volumen de la boca (±%d mm)" % VOLUMEN_BOCA))
    ax2.scatter([PLATO[0]], [PLATO[2]], color="orange", s=40, label="recipiente")
    ax2.axhline(Z_MESA, color="k", lw=0.8, ls="--")
    ax2.add_patch(plt.Rectangle((TORSO["x"][0], TORSO["z"][0]), 200, TORSO["z"][1], fc="gray", alpha=0.35, label="torso (caja prohibida)"))
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("z [mm]"); ax2.set_aspect("equal"); ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("Corte lateral (|y| < 15 mm)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_espacio_trabajo.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    r = generar()
    print(json.dumps(r, indent=2, ensure_ascii=False))
