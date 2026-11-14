"""Parámetros del brazo, del escenario y de la seguridad.
>>> LOS VALORES GEOMÉTRICOS SON DE PARTIDA: reemplazarlos con las dimensiones reales del CAD. <<<
Todo el paquete (simulación, reporte y pruebas) se recalcula a partir de este archivo.
Unidades: mm, rad, s."""
import numpy as np

D2R = np.pi / 180.0

# ------------------------- Geometría (VALORES DE PARTIDA) -------------------------
H = 80.0     # altura del eje del hombro sobre la base [mm]
L2 = 190.0   # hombro -> codo [mm]
L3 = 190.0   # codo -> muñeca [mm]
L4 = 100.0   # muñeca -> punta de la cuchara [mm]

# Parámetros D-H estándar: fila i = [offset_theta_i, d_i, a_i, alpha_i];  theta_i = q_i + offset_i
DH = np.array([
    [0.0,        H,   0.0, 90 * D2R],   # J1: giro de la base (NEMA 17)
    [0.0,        0.0, L2,  0.0],        # J2: hombro (cabeceo)
    [0.0,        0.0, L3,  0.0],        # J3: codo (cabeceo)
    [90 * D2R,   0.0, 0.0, 90 * D2R],   # J4: muñeca (cabeceo de la cuchara)
    [0.0,        L4,  0.0, 0.0],        # J5: giro propio de la cuchara
])
NOMBRES_ART = ["J1 base", "J2 hombro", "J3 codo", "J4 muñeca", "J5 cuchara"]

# Límites articulares (VALORES DE PARTIDA: usar el rango real de cada actuador)
Q_MIN = np.array([-100, -30, -150, -90, -90]) * D2R
Q_MAX = np.array([100, 135, -10, 90, 90]) * D2R
Q_HOME = np.array([0, 75, -140, 65, 0]) * D2R
VEL_ART_MAX = 90 * D2R          # velocidad articular máxima [rad/s] (supuesto)

# ------------------------------ Escenario (marco de la base) ------------------------------
BOCA_NOMINAL = np.array([330.0, 0.0, 270.0])   # boca del usuario sentado
VOLUMEN_BOCA = 50.0                            # semilado del volumen acotado [mm] (perfil, OE4)
PLATO = np.array([240.0, -150.0, 20.0])        # centro del recipiente (altura de la cuchara)
SEMI_PLATO = 40.0                              # semilado del área del recipiente [mm]
BETA_TOMA = -15 * D2R                          # inclinación de la cuchara al tomar el alimento
BETA_RECOGER = 0.0                             # inclinación al final del gesto de recoger: cuchara a nivel (con +10° la muñeca pasaría de 90°) [rad]
ARRASTRE_RECOGER = 25.0                        # recorrido horizontal del gesto de recoger, hacia la base [mm]
TILT_DERRAME = 5 * D2R                         # inclinación máxima de la cuchara durante el acarreo sin derramar (SUPUESTO: depende del alimento)
A_DERRAME = 250.0                              # aceleración máxima de la punta durante el acarreo [mm/s^2] (SUPUESTO)
BETA_TRANSPORTE = 0.0                          # cuchara a nivel al trasladar y entregar
Z_MESA = 5.0                                   # altura mínima permitida de cualquier punto [mm]
X_MIN_BRAZO = -120.0                            # espacio libre detrás de la base: codo/muñeca no pueden pasar de aquí [mm]
                                               # (poner 0 si la base está contra una pared o el respaldo)
TORSO = dict(x=(BOCA_NOMINAL[0] + VOLUMEN_BOCA + 20.0, 900.0), y=(-300.0, 300.0), z=(0.0, 260.0))  # caja prohibida: torso, detrás del volumen de la boca

# ------------------------------------ Movimiento ------------------------------------
STANDOFF_ENTREGA = 60.0     # distancia nominal cuchara-boca en la entrega [mm]
STANDOFF_TRASLADO = 120.0   # distancia de la pose de traslado [mm] (a 200 mm exigía q2 > 135°)
V_TRASLADO = 100.0          # velocidad de punta en traslado [mm/s]
V_APROX = 50.0              # límite de velocidad en la aproximación final [mm/s] (perfil)
A_MAX = 200.0               # aceleración de punta máxima [mm/s^2] (supuesto)
T_CTRL = 0.02               # periodo del lazo de control [s]

# ------------------------------------ Seguridad ------------------------------------
DIST_MIN = 30.0             # distancia mínima cuchara-boca [mm] (perfil, OE6)
TIMEOUT_ROSTRO = 0.5        # pérdida de detección máxima [s] (perfil)
TIMEOUT_COM = 0.2           # pérdida de comunicación máxima [s] (perfil)
T_PARADA_MAX = 0.3          # tiempo de parada exigido [s] (perfil)
I_MAX = 2.0                 # corriente máxima por articulación [A] (supuesto: ajustar al actuador)
DESV_MAX = 5 * D2R          # desviación posición pedida-medida [rad] (supuesto)

# ---------------------- Cámara eye-in-hand (supuesto de montaje) ----------------------
# Cámara 30 mm sobre el eje de la cuchara y 50 mm detrás de la punta; mira hacia adelante.
# Ejes de la cámara (OpenCV): x derecha, y abajo, z adelante. Reemplazar con la calibración mano-ojo.
T_EE_CAM = np.array([[0, -1, 0, 30.0],
                     [1,  0, 0, 0.0],
                     [0,  0, 1, -50.0],
                     [0,  0, 0, 1.0]])
