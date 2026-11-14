"""Cinemática directa e inversa del brazo de 5 GDL (convención D-H estándar)."""
import numpy as np
from config import *


def dh(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_todas(q):
    """Lista de transformaciones T_0^i, i = 0..5 (T_0^0 = identidad)."""
    Ts = [np.eye(4)]
    for i in range(5):
        Ts.append(Ts[-1] @ dh(q[i] + DH[i, 0], DH[i, 1], DH[i, 2], DH[i, 3]))
    return Ts


def fk(q):
    return fk_todas(q)[-1]


def punta(q):
    return fk(q)[:3, 3].copy()


def beta_herramienta(q):
    """Inclinación de la cuchara respecto a la horizontal (positivo = punta hacia arriba)."""
    return q[1] + q[2] + q[3]


def fk_lote(Q):
    """FK en forma cerrada para muchas configuraciones (N x 5). Devuelve puntas (N x 3) y beta (N)."""
    q1, q2, q3, q4 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    b = q2 + q3 + q4
    r = L2 * np.cos(q2) + L3 * np.cos(q2 + q3) + L4 * np.cos(b)
    z = H + L2 * np.sin(q2) + L3 * np.sin(q2 + q3) + L4 * np.sin(b)
    return np.stack([r * np.cos(q1), r * np.sin(q1), z], axis=1), b


def puntos_brazo(q):
    """Orígenes para dibujar: base, hombro, codo, muñeca, punta."""
    Ts = fk_todas(q)
    return np.array([Ts[0][:3, 3], Ts[1][:3, 3], Ts[2][:3, 3], Ts[3][:3, 3], Ts[5][:3, 3]])


def en_limites(q, tol=1e-9):
    return bool(np.all(q >= Q_MIN - tol) and np.all(q <= Q_MAX + tol))


def colision(q, n=10):
    """True si el brazo toca la mesa (z < Z_MESA) o entra en la caja del torso."""
    P = puntos_brazo(q)
    if np.any(P[2:, 0] < X_MIN_BRAZO - 1e-9):      # codo, muñeca o punta detrás de la base
        return True
    for a, b in zip(P[1:-1], P[2:]):
        for s in np.linspace(0, 1, n):
            p = a + s * (b - a)
            if p[2] < Z_MESA:
                return True
            if (TORSO["x"][0] <= p[0] <= TORSO["x"][1] and TORSO["y"][0] <= p[1] <= TORSO["y"][1]
                    and TORSO["z"][0] <= p[2] <= TORSO["z"][1]):
                return True
    return False


def ik_analitica(p, beta=0.0, codo=-1, q5=0.0):
    """IK cerrada: posición de la punta p y cuchara con inclinación beta.
    El azimut de la cuchara lo fija la base (5 GDL: no hay guiñada independiente).
    codo = -1: codo arriba (q3 < 0); +1: codo abajo. Devuelve q o None si es inalcanzable (sin revisar límites)."""
    x, y, z = p
    r = np.hypot(x, y)
    th1 = np.arctan2(y, x) if r > 1e-9 else 0.0
    wr = r - L4 * np.cos(beta)
    wh = z - H - L4 * np.sin(beta)
    c3 = (wr ** 2 + wh ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    if abs(c3) > 1 + 1e-9:
        return None
    c3 = float(np.clip(c3, -1, 1))
    th3 = np.arctan2(codo * np.sqrt(1 - c3 ** 2), c3)
    th2 = np.arctan2(wh, wr) - np.arctan2(L3 * np.sin(th3), L2 + L3 * np.cos(th3))
    q4 = beta - th2 - th3
    return np.array([th1, th2, th3, q4, q5])


def ik(p, beta=0.0, q_prev=None, q5=0.0, revisar_colision=True):
    """IK con límites articulares y colisiones; elige la solución más cercana a q_prev
    (o codo arriba si no hay q_prev). Devuelve q o None."""
    candidatos = []
    for codo in (-1, 1):
        q = ik_analitica(p, beta, codo, q5)
        if q is not None and en_limites(q) and not (revisar_colision and colision(q)):
            candidatos.append(q)
    if not candidatos:
        return None
    if q_prev is None:
        return candidatos[0]
    return min(candidatos, key=lambda q: np.linalg.norm(q - q_prev))


def ik_numerica(p, beta=0.0, q0=None, iters=500, lam=1e-3):
    """IK por mínimos cuadrados amortiguados (solo para validar la analítica)."""
    q = (Q_HOME if q0 is None else q0).copy()
    for _ in range(iters):
        e = np.concatenate([p - punta(q), [beta - beta_herramienta(q)]])
        if np.linalg.norm(e) < 1e-7:
            break
        J = np.vstack([jacobiano_pos(q), [0, 1, 1, 1, 0]])
        q = q + J.T @ np.linalg.solve(J @ J.T + lam * np.eye(4), e)
    return q


def jacobiano_pos(q):
    """Jacobiano geométrico de posición (3 x 5): columna i = z_{i-1} x (p_e - p_{i-1})."""
    Ts = fk_todas(q)
    pe = Ts[5][:3, 3]
    J = np.zeros((3, 5))
    for i in range(5):
        J[:, i] = np.cross(Ts[i][:3, 2], pe - Ts[i][:3, 3])
    return J


def indice_singularidad(q):
    """Menor valor singular del Jacobiano de posición de J1-J3 [mm/rad]. Cercano a 0 = singularidad."""
    return float(np.linalg.svd(jacobiano_pos(q)[:, :3], compute_uv=False).min())


def pose_entrega(boca, standoff=STANDOFF_ENTREGA):
    """Punta objetivo: a 'standoff' mm de la boca, sobre la dirección radial desde la base."""
    phi = np.arctan2(boca[1], boca[0])
    return np.asarray(boca, float) - standoff * np.array([np.cos(phi), np.sin(phi), 0.0])
