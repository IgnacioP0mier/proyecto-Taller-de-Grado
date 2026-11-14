"""Lógica del sistema de seguridad (OE6): monitor de condiciones de parada.
Al detectar una condición de riesgo pasa a PARADA (enclavada): el comando deja de avanzar y se
mantiene la última posición. Solo se rearma de forma explícita cuando ya no hay condición activa.
NOTA: en simulación solo se mide la latencia lógica del monitor; la latencia real incluye la
respuesta de los actuadores y la comunicación, que debe medirse en el hardware."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from config import *


class Estado(Enum):
    NORMAL = "NORMAL"
    PARADA = "PARADA"


@dataclass
class Entradas:
    t: float                          # tiempo actual [s]
    tip: np.ndarray                   # posición de la cuchara [mm]
    q: np.ndarray                     # ángulos actuales [rad]
    v_tip: float                      # velocidad de la punta [mm/s]
    t_ultimo_rostro: float            # instante de la última detección válida del rostro
    t_ultimo_latido: float            # instante del último mensaje de comunicación recibido
    fase_aprox: bool = False          # True durante la aproximación final
    boca_est: np.ndarray = None       # boca estimada en el marco del robot [mm] (si existe)
    corriente: np.ndarray = None      # corriente por articulación [A] (si se mide)
    desviacion: float = None          # máx |q_pedido - q_medido| [rad] (si se mide)
    paro_emergencia: bool = False


class MonitorSeguridad:
    def __init__(self):
        self.estado = Estado.NORMAL
        self.motivo = None
        self.t_parada = None
        self.q_hold = None

    def _parar(self, e, motivo):
        self.estado, self.motivo, self.t_parada = Estado.PARADA, motivo, e.t
        self.q_hold = np.array(e.q, float).copy()

    def evaluar(self, e: Entradas):
        """Revisa todas las condiciones; devuelve (estado, motivo)."""
        if self.estado is Estado.PARADA:
            return self.estado, self.motivo
        if e.paro_emergencia:
            self._parar(e, "paro de emergencia")
        elif e.t - e.t_ultimo_latido > TIMEOUT_COM:
            self._parar(e, "falla de comunicación (> %.0f ms)" % (TIMEOUT_COM * 1000))
        elif np.any(e.q < Q_MIN - 1e-9) or np.any(e.q > Q_MAX + 1e-9):
            self._parar(e, "límite articular excedido")
        elif e.corriente is not None and np.any(np.asarray(e.corriente) > I_MAX):
            self._parar(e, "corriente fuera de rango")
        elif e.desviacion is not None and e.desviacion > DESV_MAX:
            self._parar(e, "desviación de posición fuera de rango")
        elif e.fase_aprox:
            if e.t - e.t_ultimo_rostro > TIMEOUT_ROSTRO:
                self._parar(e, "pérdida de detección del rostro (> %.0f ms)" % (TIMEOUT_ROSTRO * 1000))
            elif e.v_tip > 1.05 * V_APROX:
                self._parar(e, "velocidad de aproximación excedida")
            elif e.boca_est is not None and np.linalg.norm(e.tip - e.boca_est) < DIST_MIN:
                self._parar(e, "distancia mínima a la boca (< %.0f mm)" % DIST_MIN)
        return self.estado, self.motivo

    def comando(self, q_cmd):
        """Filtra el comando: en PARADA mantiene la posición enclavada."""
        return self.q_hold if self.estado is Estado.PARADA else q_cmd

    def rearmar(self, e: Entradas):
        """Rearme manual: solo si la condición que causó la parada ya no existe."""
        previo = (self.estado, self.motivo, self.t_parada, self.q_hold)
        self.estado = Estado.NORMAL
        self.estado, self.motivo = self.evaluar(e)
        if self.estado is Estado.PARADA:        # sigue habiendo riesgo
            self.estado, self.motivo, self.t_parada, self.q_hold = previo
            return False
        self.motivo = self.t_parada = self.q_hold = None
        return True
