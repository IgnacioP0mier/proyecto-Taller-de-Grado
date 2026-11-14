"""Generación de trayectorias y control de aproximación con límites de velocidad y aceleración."""
import numpy as np
from config import *
from cinematica import *


def quintico(s):
    """Perfil suave 0->1 con velocidad y aceleración nulas en los extremos."""
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


OMEGA_MAX = 45 * D2R   # velocidad máxima de cambio de inclinación de la cuchara [rad/s] (supuesto)


def segmento_cartesiano(p0, p1, beta, v_max, q_prev, dt=T_CTRL):
    """Recta p0->p1; la inclinación de la cuchara pasa de beta[0] a beta[1] (o es constante si beta es un número).
    Duración tal que la velocidad pico de la punta (1.875*L/T en el perfil quíntico), la del cambio de
    inclinación y la velocidad articular (VEL_ART_MAX) no se excedan. Devuelve (P, Q) cada dt, o (None, None)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    b0, b1 = (beta, beta) if np.isscalar(beta) else beta
    L = np.linalg.norm(p1 - p0)
    T = max(0.5, 1.875 * L / v_max, 1.875 * abs(b1 - b0) / OMEGA_MAX)
    for _ in range(8):
        n = int(np.ceil(T / dt))
        P, Q, q = [], [], q_prev
        for k in range(1, n + 1):
            s = quintico(k / n)
            q = ik(p0 + s * (p1 - p0), b0 + s * (b1 - b0), q_prev=q)
            if q is None:
                return None, None
            P.append(p0 + s * (p1 - p0))
            Q.append(q)
        Q = np.array(Q)
        v_art = np.abs(np.diff(np.vstack([q_prev, Q]), axis=0)).max() / dt
        if v_art <= 0.9 * VEL_ART_MAX:
            return np.array(P), Q
        T *= v_art / (0.9 * VEL_ART_MAX)        # alargar el tiempo y reintentar
    return None, None


class ControladorAproximacion:
    """Lleva la punta hacia un objetivo móvil con límite de velocidad (V_APROX) y de aceleración (A_MAX)."""
    def __init__(self, tip0, kp=3.0, v_max=V_APROX, a_max=A_MAX):
        self.tip = np.array(tip0, float)
        self.v = np.zeros(3)
        self.kp, self.v_max, self.a_max = kp, v_max, a_max

    def paso(self, objetivo, dt=T_CTRL):
        v_des = self.kp * (np.asarray(objetivo, float) - self.tip)
        n = np.linalg.norm(v_des)
        if n > self.v_max:
            v_des *= self.v_max / n
        dv = v_des - self.v
        nd = np.linalg.norm(dv)
        if nd > self.a_max * dt:
            dv *= self.a_max * dt / nd
        self.v = self.v + dv
        self.tip = self.tip + self.v * dt
        return self.tip.copy(), float(np.linalg.norm(self.v))


def plan_transporte(q_inicio, boca):
    """Segmentos hasta la pose de traslado: home -> pre-toma -> toma -> levantar -> traslado.
    Cada elemento: (nombre, p0, p1, (beta0, beta1), v_max)."""
    p_home = punta(q_inicio)
    p_pre = PLATO + np.array([0, 0, 70.0])
    p_recoger = PLATO + np.array([-ARRASTRE_RECOGER, 0, 12.0])   # fin del "cuchareo": se arrastra hacia la base y se levanta la punta
    p_lev = p_recoger + np.array([0, 0, 50.0])
    p_tras = pose_entrega(boca, STANDOFF_TRASLADO)
    return [("home -> pre-toma", p_home, p_pre, (BETA_TRANSPORTE, BETA_TRANSPORTE), V_TRASLADO),
            ("pre-toma -> toma", p_pre, PLATO.copy(), (BETA_TRANSPORTE, BETA_TOMA), 40.0),
            ("recoger", PLATO.copy(), p_recoger, (BETA_TOMA, BETA_RECOGER), 25.0),
            ("toma -> levantar", p_recoger, p_lev, (BETA_RECOGER, BETA_TRANSPORTE), 40.0),   # se nivela la cuchara al subir
            ("levantar -> traslado", p_lev, p_tras, (BETA_TRANSPORTE, BETA_TRANSPORTE), V_TRASLADO)]
