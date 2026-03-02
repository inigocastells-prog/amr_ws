import math


class PurePursuit:
    """Class to follow a path using a simple pure pursuit controller."""

    def __init__(self, dt: float, lookahead_distance: float = 0.5):
        """Pure pursuit class initializer.

        Args:
            dt: Sampling period [s].
            lookahead_distance: Distance to the next target point [m].

        """
        self._dt: float = dt
        self._lookahead_distance: float = lookahead_distance
        self._path: list[tuple[float, float]] = []

    def compute_commands(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Pure pursuit controller implementation.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].
            theta: Estimated robot heading [rad].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 4.4. Complete the function body with your code (i.e., compute v and w).
        v = 0.0
        w = 0.0

        # 0) Si no hay path todavía, no hacemos nada
        if not self._path:
            return 0.0, 0.0

        # 1) Punto del path más cercano al robot
        closest_xy, closest_idx = self._find_closest_point(x, y)

        # 2) Punto objetivo (carrot) a lookahead
        target_xy = self._find_target_point((x, y), closest_idx)
        tx, ty = target_xy

        # 3) Vector hacia el objetivo en frame global
        dx = tx - x
        dy = ty - y

        # 4) Transformación a frame del robot (rotación por -theta)
        # x_r = cos(theta)*dx + sin(theta)*dy
        # y_r = -sin(theta)*dx + cos(theta)*dy
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        x_r = cos_th * dx + sin_th * dy
        y_r = -sin_th * dx + cos_th * dy

        # Si el objetivo cae detrás, gira en sitio para recolocarse.
        if x_r <= 1e-6:
            v = 0.0
            w = 0.6 if y_r > 0.0 else -0.6
            return v, w

        # 5) Curvatura pure pursuit: kappa = 2*y_r / L^2
        L = self._lookahead_distance
        if L <= 1e-6:
            return 0.0, 0.0

        kappa = 2.0 * y_r / (L * L)

        # 6) Elegimos una v (constante) y calculamos w = v * kappa
        # (Luego se satura para evitar valores absurdos)
        v = 0.8  # m/s
        w = v * kappa

        # Saturaciones razonables
        W_MAX = 1.2
        if w > W_MAX:
            w = W_MAX
        elif w < -W_MAX:
            w = -W_MAX

        # Opcional: bajar v en curvas fuertes
        if abs(w) > 0.8:
            v = 0.4

        return v, w

    @property
    def path(self) -> list[tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: list[tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value

    def _find_closest_point(self, x: float, y: float) -> tuple[tuple[float, float], int]:
        """Find the closest path point to the current robot pose.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].

        Returns:
            tuple[float, float]: (x, y) coordinates of the closest path point [m].
            int: Index of the path point found.

        """
        # TODO: 4.2. Complete the function body (i.e., find closest_xy and closest_idx).
        closest_xy = (0.0, 0.0)
        closest_idx = 0

        # Si todavía no hay path, devolvemos algo seguro
        if not self._path:
            closest_xy = (x, y)
            closest_idx = 0
            return closest_xy, closest_idx

        # Inicializa con el primer punto del path
        closest_xy = self._path[0]
        closest_idx = 0
        min_dist2 = float("inf")

        # Busca el punto más cercano
        for i, (px, py) in enumerate(self._path):
            dx = px - x
            dy = py - y
            dist2 = dx * dx + dy * dy  # distancia al cuadrado

            if dist2 < min_dist2:
                min_dist2 = dist2
                closest_idx = i
                closest_xy = (px, py)

        return closest_xy, closest_idx

    def _find_target_point(
        self, origin_xy: tuple[float, float], origin_idx: int
    ) -> tuple[float, float]:
        """Find the destination path point based on the lookahead distance.

        Args:
            origin_xy: Current location of the robot (x, y) [m].
            origin_idx: Index of the current path point.

        Returns:
            tuple[float, float]: (x, y) coordinates of the target point [m].

        """
        # TODO: 4.3. Complete the function body with your code (i.e., determine target_xy).
        target_xy = (0.0, 0.0)

        # Si no hay path, devolvemos el origen
        if not self._path:
            target_xy = origin_xy
            return target_xy

        # Si origin_idx está fuera de rango, lo saturamos
        if origin_idx < 0:
            origin_idx = 0
        if origin_idx >= len(self._path):
            origin_idx = len(self._path) - 1

        L = self._lookahead_distance

        # Si el lookahead es 0 o negativo, target = punto actual
        if L <= 0.0:
            target_xy = origin_xy
            return target_xy

        # Recorremos el path desde origin_idx y acumulamos distancia a lo largo del camino
        acc = 0.0
        prev_x, prev_y = origin_xy

        for i in range(origin_idx, len(self._path)):
            px, py = self._path[i]

            dx = px - prev_x
            dy = py - prev_y
            seg_len = math.hypot(dx, dy)

            # Si este segmento ya alcanza el lookahead
            if acc + seg_len >= L:
                # Interpolación lineal en el segmento (prev -> p)
                remaining = L - acc
                if seg_len > 1e-9:
                    t = remaining / seg_len  # entre 0 y 1
                else:
                    t = 0.0
                tx = prev_x + t * dx
                ty = prev_y + t * dy
                target_xy = (tx, ty)
                return target_xy

            acc += seg_len
            prev_x, prev_y = px, py

        # Si no llegamos a la distancia lookahead, usamos el último punto del path
        target_xy = self._path[-1]

        return target_xy
