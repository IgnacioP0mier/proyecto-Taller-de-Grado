"""Simulación del ciclo completo: home -> toma -> traslado -> aproximación guiada (boca móvil) -> entrega -> retiro.
La boca se mueve (deriva de postura) y se mide con ruido; el controlador corrige la entrega y el monitor de
seguridad vigila cada paso. Los parámetros del ruido y de la deriva son SUPUESTOS de simulación, no mediciones."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from cinematica import *
from trayectoria import *
from seguridad import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
SIGMA_LOC = 5.0        # ruido de localización simulado [mm] (SUPUESTO)
ALPHA = 0.4            # suavizado de la medición (igual que vision_boca.py)


T_CAMBIO = 6.0   # instante del cambio brusco de postura, medido desde el inicio de la aproximación [s]
t_ap0 = None     # instante en que empieza la aproximación guiada (se fija al llegar a ella)


def boca_real(t):
    """Deriva de postura dentro del volumen acotado (±50 mm) y un cambio brusco de +25 mm en y."""
    tau = 0.0 if t_ap0 is None else max(0.0, t - t_ap0)
    d = np.array([20 * np.sin(2 * np.pi * 0.10 * tau), 15 * np.sin(2 * np.pi * 0.07 * tau + 1.0), 10 * np.sin(2 * np.pi * 0.05 * tau)])
    if tau > T_CAMBIO:
        d += np.array([0.0, 25.0, 0.0])
    return BOCA_NOMINAL + d


def ejecutar(seed=0, guardar=True):
    global t_ap0
    t_ap0 = None
    rng = np.random.default_rng(seed)
    mon = MonitorSeguridad()
    log = dict(t=[], q=[], tip=[], v=[], d_real=[], err=[], fase=[], estado=[], boca=[])
    t, q = 0.0, Q_HOME.copy()
    tip_prev = punta(q)
    t_rostro = t_latido = 0.0
    fase_fin = {}

    def registrar(q_, tip_, v_, d_real, err, fase):
        log["t"].append(t); log["q"].append(q_.copy()); log["tip"].append(tip_.copy()); log["v"].append(v_)
        log["d_real"].append(d_real); log["err"].append(err); log["fase"].append(fase); log["estado"].append(mon.estado.value); log["boca"].append(boca_real(t))

    def vigilar(q_, tip_, v_, aprox, boca_est=None):
        e = Entradas(t=t, tip=tip_, q=q_, v_tip=v_, t_ultimo_rostro=t_rostro, t_ultimo_latido=t_latido,
                     fase_aprox=aprox, boca_est=boca_est)
        return mon.evaluar(e)

    # ---------- 1) Transporte: home -> pre-toma -> toma -> levantar -> traslado (cuchara a nivel) ----------
    for nombre, p0, p1, beta, vmax in plan_transporte(q, BOCA_NOMINAL):
        P, Q = segmento_cartesiano(p0, p1, beta, vmax, q)
        assert P is not None, "segmento inalcanzable: " + nombre
        for p, qq in zip(P, Q):
            t += T_CTRL; t_rostro = t_latido = t
            v = np.linalg.norm(p - tip_prev) / T_CTRL
            vigilar(qq, p, v, False)
            registrar(mon.comando(qq), p, v, np.linalg.norm(p - boca_real(t)), np.nan, nombre)
            tip_prev, q = p, qq
    # ---------- 2) Aproximación final guiada por visión (boca móvil, ruido y suavizado) ----------
    ctrl = ControladorAproximacion(tip_prev)
    t_ap0 = t
    est = boca_real(t) + rng.normal(0, SIGMA_LOC, 3)
    llegada, t_llegada, ok_aprox = 0, None, True
    while t - t_ap0 < 30.0:
        t += T_CTRL; t_rostro = t_latido = t
        meas = boca_real(t) + rng.normal(0, SIGMA_LOC, 3)
        est = ALPHA * meas + (1 - ALPHA) * est
        objetivo = pose_entrega(est, STANDOFF_ENTREGA)
        tip, v = ctrl.paso(objetivo)
        qq = ik(tip, BETA_TRANSPORTE, q_prev=q)
        if qq is None:
            ok_aprox = False
            break
        estado, motivo = vigilar(qq, tip, v, True, boca_est=est)
        err = np.linalg.norm(tip - pose_entrega(boca_real(t), STANDOFF_ENTREGA))
        registrar(mon.comando(qq), tip, v, np.linalg.norm(tip - boca_real(t)), err, "aproximación guiada")
        q = qq
        if estado is Estado.PARADA:
            break
        llegada = llegada + 1 if err < 8.0 else 0
        if llegada * T_CTRL >= 1.0 and (t - t_ap0) > T_CAMBIO + 3.0:   # 1 s dentro de 8 mm, tras el cambio de postura
            t_llegada = t
            break
    # ---------- 3) Retiro a la pose de traslado y regreso a home ----------
    if mon.estado is Estado.NORMAL and t_llegada is not None:
        for nombre, p0, p1, beta in [("retiro", tip, pose_entrega(boca_real(t), STANDOFF_TRASLADO), BETA_TRANSPORTE)]:
            P, Q = segmento_cartesiano(p0, p1, beta, V_TRASLADO, q)
            if P is not None:
                for p, qq in zip(P, Q):
                    t += T_CTRL; t_rostro = t_latido = t
                    v = np.linalg.norm(p - log["tip"][-1]) / T_CTRL
                    vigilar(qq, p, v, False)
                    registrar(mon.comando(qq), p, v, np.linalg.norm(p - boca_real(t)), np.nan, nombre)
                    q = qq
    L = {k: np.array(v) for k, v in log.items()}
    ap = L["fase"] == "aproximación guiada"
    qdot = np.abs(np.diff(L["q"], axis=0)) / T_CTRL
    res = dict(
        semilla=seed, sigma_loc_mm=SIGMA_LOC, estado_final=mon.estado.value, motivo_parada=mon.motivo,
        llego_a_entrega=bool(t_llegada is not None), t_total_s=float(L["t"][-1]),
        duracion_aprox_s=float(ap.sum() * T_CTRL),
        v_max_aprox_mm_s=float(L["v"][ap].max()) if ap.any() else None,
        **_metricas_seguimiento(L, ap),
        dist_min_boca_mm=float(L["d_real"][ap].min()) if ap.any() else None,
        v_art_max_deg_s=float(np.degrees(qdot.max())),
        dentro_limites_art=bool(np.all(L["q"] >= Q_MIN - 1e-9) and np.all(L["q"] <= Q_MAX + 1e-9)),
        sing_min=float(min(indice_singularidad(q_) for q_ in L["q"][::5])),
        x_min_articulaciones_mm=float(min(puntos_brazo(q_)[2:, 0].min() for q_ in L["q"][::5])),
        **_metricas_derrame(L),
        pasos=int(len(L["t"])))
    if guardar:
        os.makedirs(OUT, exist_ok=True)
        num = np.column_stack([L["t"], np.degrees(L["q"]), L["tip"], L["v"], L["d_real"], L["boca"]])
        with open(os.path.join(OUT, "ciclo_log.csv"), "w", encoding="utf8") as f:
            f.write("t_s,q1_deg,q2_deg,q3_deg,q4_deg,q5_deg,x_mm,y_mm,z_mm,v_mm_s,dist_boca_mm,boca_x_mm,boca_y_mm,boca_z_mm,fase\n")
            for fila, fase in zip(num, L["fase"]):
                f.write(",".join(f"{x:.3f}" for x in fila) + "," + str(fase) + "\n")
        with open(os.path.join(OUT, "ciclo_resumen.json"), "w", encoding="utf8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        _figuras(L, ap)
    return res


def _metricas_derrame(L):
    """Acarreo = desde que termina el gesto de recoger hasta el final de la aproximación guiada (la comida va en la cuchara).
    Criterio de no derrame (SUPUESTO): inclinación respecto de la horizontal <= TILT_DERRAME y aceleración de la punta <= A_DERRAME."""
    f = L["fase"]
    ini = np.where(f == "toma -> levantar")[0]
    fin = np.where(f == "aproximación guiada")[0]
    if not len(ini) or not len(fin):
        return {}
    sl = slice(ini[0] + int(0.4 * (np.sum(f == "toma -> levantar"))), fin[-1] + 1)   # la cuchara ya casi a nivel a mitad de la subida
    beta = np.array([q[1] + q[2] + q[3] for q in L["q"][sl]])
    tip = L["tip"][sl]
    acc = np.linalg.norm(np.diff(tip, 2, axis=0), axis=1) / T_CTRL ** 2
    acc_f = np.convolve(acc, np.ones(5) / 5, mode="same")      # media móvil 0.1 s: la derivada segunda amplifica el ruido
    roll = np.abs(L["q"][sl][:, 4])
    inc = np.degrees(np.maximum(np.abs(beta), roll))
    return dict(acarreo_incl_max_deg=float(inc.max()), acarreo_acel_max_mm_s2=float(acc_f.max()),
                acarreo_sin_derrame=bool(inc.max() <= np.degrees(TILT_DERRAME) and acc_f.max() <= A_DERRAME),
                t_acarreo_s=float(len(beta) * T_CTRL))


def _metricas_seguimiento(L, ap):
    """t_conv: primer instante con error < 8 mm; régimen: RMS desde t_conv hasta el cambio de postura;
    recuperación: tiempo desde el cambio brusco hasta volver a < 8 mm sostenido 0.5 s."""
    if not ap.any():
        return {}
    tau = L["t"][ap] - L["t"][ap][0]
    err = L["err"][ap]
    ya = np.where(err < 8.0)[0]
    t_conv = float(tau[ya[0]]) if len(ya) else None
    reg = (tau >= (t_conv or 0)) & (tau < T_CAMBIO)
    rec = None
    post = np.where(tau > T_CAMBIO)[0]
    for i in post:
        j = min(i + 25, len(err))
        if np.all(err[i:j] < 8.0) and j - i >= 25:
            rec = float(tau[i] - T_CAMBIO)
            break
    return dict(t_convergencia_s=t_conv,
                error_rms_regimen_mm=float(np.sqrt(np.mean(err[reg] ** 2))) if reg.any() else None,
                error_max_tras_cambio_mm=float(err[tau > T_CAMBIO].max()) if (tau > T_CAMBIO).any() else None,
                t_recuperacion_s=rec, error_final_mm=float(err[-1]))


def _dibujar_brazo(ax, q, color, lw=2, alpha=1.0, etiqueta=None):
    P = puntos_brazo(q)
    ax.plot(P[:, 0], P[:, 2], "-o", color=color, lw=lw, ms=3, alpha=alpha, label=etiqueta)


def _figuras(L, ap):
    # 1) Poses clave del ciclo (vista lateral)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    claves = [("home", 0, "gray"), ("toma", None, "orange"), ("traslado", None, "tab:blue"), ("entrega", None, "tab:red")]
    idx = {"home": 0}
    fases = L["fase"]
    idx["toma"] = int(np.where(fases == "pre-toma -> toma")[0][-1])
    idx["traslado"] = int(np.where(fases == "levantar -> traslado")[0][-1])
    idx["entrega"] = int(np.where(ap)[0][-1]) if ap.any() else len(fases) - 1
    for nom, _, col in claves:
        _dibujar_brazo(ax, L["q"][idx[nom]], col, etiqueta=nom)
    ax.plot(BOCA_NOMINAL[0], BOCA_NOMINAL[2], "r*", ms=12, label="boca (nominal)")
    ax.add_patch(plt.Rectangle((TORSO["x"][0], 0), 150, TORSO["z"][1], fc="gray", alpha=0.3))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]"); ax.legend(fontsize=8); ax.set_title("Poses clave del ciclo (vista lateral)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_poses_ciclo.png"), dpi=160); plt.close(fig)
    # 2) Series temporales
    t = L["t"]
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=True)
    axs[0].plot(t, np.degrees(L["q"])); axs[0].set_ylabel("q [°]"); axs[0].legend(["q1", "q2", "q3", "q4", "q5"], ncol=5, fontsize=7)
    axs[1].plot(t, L["v"], color="tab:green"); axs[1].axhline(V_APROX, color="r", ls="--", label="límite aproximación (%.0f mm/s)" % V_APROX)
    axs[1].set_ylabel("v punta [mm/s]"); axs[1].legend(fontsize=8)
    axs[2].plot(t, L["d_real"], color="tab:purple"); axs[2].axhline(DIST_MIN, color="r", ls="--", label="distancia mínima (%.0f mm)" % DIST_MIN)
    axs[2].set_ylabel("dist. cuchara-boca [mm]"); axs[2].set_xlabel("t [s]"); axs[2].legend(fontsize=8); axs[2].set_ylim(0, 400)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_series_ciclo.png"), dpi=160); plt.close(fig)
    # 3) Seguimiento en la aproximación
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(t[ap], L["err"][ap], color="tab:red"); ax.axvline(t[ap][0] + T_CAMBIO, color="gray", ls=":", label="cambio brusco de postura (+25 mm en y)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("error de seguimiento [mm]"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_seguimiento.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(ejecutar(), indent=2, ensure_ascii=False))
