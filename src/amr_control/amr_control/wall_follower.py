import math
class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""
    
    def __init__(self, dt: float):
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].

        """
        self._dt: float = dt
        
    def compute_commands(self, z_us: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        """Wall following exploration algorithm.

        Args:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 1.14. Complete the function body with your code (i.e., compute v and w).
        
   

        # --- Parámetros ---
        V_NOM = 0.45
        V_MIN = 0.05
        W_MAX = 0.8
        W_FOLLOW_MAX = 0.30

        D_WALL = 0.35          # objetivo pared
        KP = 1               # giro por error lateral

        D_STOP = 0.35          # parar si obstáculo REAL delante
        D_CLEAR = 0.50         # histéresis para salir del giro
        D_SLOW = 0.5          # ralentiza si algo delante se acerca

        TURN_RATE = 0.5
        FOLLOW_SIDE = "right"  # pared a la derecha

        # --- Estado sin tocar __init__ ---
        if not hasattr(self, "_wf_state"):
            self._wf_state = "follow"
            self._wf_turn_dir = 1.0
            self._wf_turn_time = 0.0

        def cap(x, lo, hi):
            return max(lo, min(hi, x))

        # normaliza lecturas (inf -> 1.0)
        z = []
        for d in z_us:
            if d is None or (isinstance(d, float) and math.isnan(d)) or math.isinf(d):
                z.append(1.0)
            else:
                z.append(cap(float(d), 0.0, 1.0))

        # =========================================================
        # Sectores CORREGIDOS (evita que "frente" use sensores inclinados)
        # indices 0..15 = sensores 1..16
        #
        # Frente REAL: sensores 4 y 5 (idx 3,4) (opcional incluye 3 y 6 idx 2,5)
        # Derecha para seguir pared: sensores 7 y 8 (idx 6,7)
        # Izquierda: sensores 2 y 1 (idx 1,0) o 16 y 1 (idx 15,0)
        # =========================================================
        front = min(z[3], z[4], z[2], z[5])      # 4,5 (+3,6)
        right = min(z[6], z[7])                  # 7,8
        left  = min(z[0], z[15])                 # 1,16 (laterales)

        # --- Transiciones ---
        if self._wf_state == "follow":
            if front < D_STOP:
                self._wf_state = "turn"
                self._wf_turn_time = 0.0
                # elige giro hacia el lado más libre
                self._wf_turn_dir = 1.0 if left > right else -1.0

        elif self._wf_state == "turn":
            self._wf_turn_time += self._dt
            # salir del giro cuando el frente ya esté libre
            if self._wf_turn_time > 0.4 and front > D_CLEAR:
                self._wf_state = "follow"

        # --- Acciones ---
        if self._wf_state == "turn":
            v = 0.0
            w = cap(self._wf_turn_dir * TURN_RATE, -W_MAX, W_MAX)
            return v, w

        # FOLLOW: velocidad según frente
        if front < D_STOP:
            v = 0.0
        elif front < D_SLOW:
            a = (front - D_STOP) / (D_SLOW - D_STOP)
            a = cap(a, 0.0, 1.0)
            v = V_MIN + a * (V_NOM - V_MIN)
        else:
            v = V_NOM

        # FOLLOW: control lateral (pared derecha por defecto)
        if FOLLOW_SIDE == "right":
            err = (D_WALL - right)      # si right < D_WALL => err>0 => girar izq (w>0)
            w = KP * err
            # si no hay pared a la derecha, busca pared girando un poquito a la derecha
            if right > 0.95:
                w = -0.15
        else:
            err = (D_WALL - left)
            w = -KP * err
            if left > 0.95:
                w = +0.15

        w = cap(w, -W_FOLLOW_MAX, W_FOLLOW_MAX)
        return v, w