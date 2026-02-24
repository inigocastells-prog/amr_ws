import datetime
import math
import numpy as np
import os
import pytz

from amr_localization.map import Map
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        sensors: list[tuple[float, float, float]],
        sensor_range: float,
        particle_count: int,
        sigma_v: float = 0.15,
        sigma_w: float = 0.75,
        sigma_z: float = 0.25,
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            sensors: Robot sensors' pose in the robot coordinate frame (x, y, theta) [m, m, rad].
            sensor_range: Sensor measurement range [m].
            particle_count: Initial number of particles.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].

        """
        self._dt: float = dt
        self._initial_particle_count: int = particle_count
        self._particle_count: int = particle_count
        self._sensors: list[tuple[float, float, float]] = sensors
        self._sensor_range: float = sensor_range
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0

        self._map = Map(map_path, sensor_range, compiled_intersect=True, use_regions=True)
        self._particles = self._init_particles(particle_count)
        self._ds, self._phi = self._init_sensor_polar_coordinates(sensors)
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    def compute_pose(self) -> tuple[bool, tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Adapts the amount of particles depending on the number of clusters during localization.
        100 particles are kept for pose tracking.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        # TODO: 2.10. Complete the missing function body with your code.
        # Hyperparameters (ajústalos si hace falta)
        DBSCAN_EPS = 0.20          # [m] radio vecindad
        DBSCAN_MIN_SAMPLES = 15    # mínimo para formar cluster

        MIN_PARTICLES = 100
        MAX_PARTICLES = self._initial_particle_count

        TRACKING_PARTICLES = 100          # partículas cuando ya está localizado (modo “GPS”)
        PARTICLES_PER_CLUSTER = 100       # heurística cuando hay varios clusters

        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        # Seguridad básica
        if self._particles is None or len(self._particles) == 0:
            return False, pose

        # DBSCAN sobre (x,y)
        xy = np.array(self._particles[:, 0:2], dtype=float)
        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)

        # Clusters válidos (ignoramos ruido = -1)
        cluster_ids = [c for c in set(labels.tolist()) if c != -1]
        num_clusters = len(cluster_ids)

        # Decide si está localizado y cuántas partículas mantener
        if num_clusters == 1:
            localized = True
            target_count = TRACKING_PARTICLES
        else:
            localized = False
            if num_clusters == 0:
                target_count = MAX_PARTICLES
            else:
                target_count = PARTICLES_PER_CLUSTER * max(1, num_clusters)

            target_count = max(MIN_PARTICLES, min(MAX_PARTICLES, target_count))

        # Reducir/aumentar partículas por submuestreo uniforme (NO likelihood aquí)
        n = int(self._particles.shape[0])
        if target_count != n:
            idx = np.random.choice(
                np.arange(n),
                size=int(target_count),
                replace=(target_count > n),
            )
            self._particles = self._particles[idx].copy()

            # Recalcular labels si hemos cambiado el set (importante)
            xy = np.array(self._particles[:, 0:2], dtype=float)
            labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)
            cluster_ids = [c for c in set(labels.tolist()) if c != -1]
            num_clusters = len(cluster_ids)

            # Si tras reducir ya no queda un solo cluster, no damos pose
            if localized and num_clusters != 1:
                return False, pose

        # Si localizado, estimamos pose desde el único cluster
        if localized:
            # Asegura que hay exactamente un cluster no ruido
            cluster_ids = [c for c in set(labels.tolist()) if c != -1]
            if len(cluster_ids) != 1:
                return False, pose

            cid = cluster_ids[0]
            mask = labels == cid
            pts = self._particles[mask]

            if pts.shape[0] == 0:
                return False, pose

            xs = pts[:, 0].astype(float)
            ys = pts[:, 1].astype(float)
            th = pts[:, 2].astype(float)

            x_h = float(np.mean(xs))
            y_h = float(np.mean(ys))

            # Media circular para theta (maneja 0 ~ 2π)
            s = float(np.mean(np.sin(th)))
            c = float(np.mean(np.cos(th)))
            theta_h = float(math.atan2(s, c) % (2.0 * math.pi))

            pose = (x_h, y_h, theta_h)

        return localized, pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1

        # TODO: 2.5. Complete the function body with your code (i.e., replace the pass statement).
        dt = self._dt
        sigma_v = self._sigma_v
        sigma_w = self._sigma_w

        n = self._particles.shape[0]

        for i in range(n):
            x0 = float(self._particles[i, 0])
            y0 = float(self._particles[i, 1])
            th0 = float(self._particles[i, 2])

            # 1) Add zero-mean Gaussian noise to velocities
            v_n = float(np.random.normal(loc=v, scale=sigma_v))
            w_n = float(np.random.normal(loc=w, scale=sigma_w))

            # 2) Motion model (simple unicycle integration)
            x1 = x0 + v_n * dt * math.cos(th0)
            y1 = y0 + v_n * dt * math.sin(th0)
            th1 = th0 + w_n * dt

            # 3) Wrap angle to [0, 2*pi)
            th1 = th1 % (2.0 * math.pi)

            # 4) Collision handling: if segment crosses a wall, make it a "ghost"
            segment = [(x0, y0), (x1, y1)]
            intersection, _ = self._map.check_collision(segment, compute_distance=False)

            if intersection:  # collision found -> stay at intersection point
                x1 = float(intersection[0])
                y1 = float(intersection[1])

            # 5) Save updated particle
            self._particles[i, 0] = x1
            self._particles[i, 1] = y1
            self._particles[i, 2] = th1
        
    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 2.9. Complete the function body with your code (i.e., replace the pass statement).
        n = self._particles.shape[0]

        weights = np.zeros(n, dtype=float)
        for i in range(n):
            p = (
                float(self._particles[i, 0]),
                float(self._particles[i, 1]),
                float(self._particles[i, 2]),
            )
            weights[i] = self._measurement_probability(measurements, p)

        w_sum = float(np.sum(weights))
        if w_sum <= 0.0 or not np.isfinite(w_sum):
            weights[:] = 1.0 / n
        else:
            weights /= w_sum

        idx = np.random.choice(np.arange(n), size=n, replace=True, p=weights)

        self._particles = self._particles[idx].copy()
        
    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(
                self._particles[:, 0],
                self._particles[:, 1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], "bo", markersize=1)

        return axes

    def show(
        self,
        title: str = "",
        orientation: bool = True,
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, orientation)

        axes.set_title(title + " (Iteration #" + str(self._iteration) + ")")
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", save_dir, self._timestamp)
            )

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = str(self._iteration).zfill(4) + " " + title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_particles(self, particle_count: int) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=object)

        # TODO: 2.4. Complete the missing function body with your code.
        
        thetas = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0], dtype=float)

        # bounds() -> (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = self._map.bounds()

        i = 0
        attempts = 0
        max_attempts = 200000

        while i < particle_count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "No se pudieron generar partículas válidas. Revisa bounds/contains."
                )

            x = float(np.random.uniform(x_min, x_max))
            y = float(np.random.uniform(y_min, y_max))
            theta = float(np.random.choice(thetas))

            if self._map.contains((x, y)):
                particles[i, 0] = x
                particles[i, 1] = y
                particles[i, 2] = theta
                i += 1

        return particles

    @staticmethod
    def _init_sensor_polar_coordinates(
        sensors: list[tuple[float, float, float]],
    ) -> tuple[list[float], list[float]]:
        """Converts the sensors' poses to polar coordinates wrt to the robot's coordinate frame.

        Args:
            sensors: Robot sensors location and orientation (x, y, theta) [m, m, rad].

        Return:
            ds: List of magnitudes [m].
            phi: List of angles [rad].

        """
        ds = [math.sqrt(sensor[0] ** 2 + sensor[1] ** 2) for sensor in sensors]
        phi = [math.atan2(sensor[1], sensor[0]) for sensor in sensors]

        return ds, phi

    def _sense(self, particle: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every sensor given the robot's pose.

        Args:
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements; inf if a sensor is out of range.

        """
        rays: list[list[tuple[float, float]]] = self._sensor_rays(particle)
        z_hat: list[float] = []

        # TODO: 2.6. Complete the missing function body with your code.
        for ray in rays:
            intersection, dist = self._map.check_collision(ray, compute_distance=True)

            # Si no hay intersección, el sensor está fuera de rango => inf
            if not intersection:
                z_hat.append(float("inf"))
            else:
                # dist ya es desde el inicio del rayo (posición del sensor) al obstáculo
                z_hat.append(float(dist))

        return z_hat
        

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        # TODO: 2.7. Complete the function body (i.e., replace the code below).
        if sigma <= 0.0:
            return 1.0 if x == mu else 0.0

        coeff = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
        exp = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * math.exp(exp)
        
    def _measurement_probability(
        self, measurements: list[float], particle: tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with
        1.25 times the sensor range to perform the computation. This value has experimentally been
        proven valid to deal with missing measurements. Nevertheless, it might not be the optimal
        replacement value.

        Args:
            measurements: Sensor measurements [m].
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns:
            float: Probability.

        """
        probability = 1.0

        # TODO: 2.8. Complete the missing function body with your code.
        
        z_hat = self._sense(particle)

        # Factor para reemplazar lecturas inf (cámbialo si quieres)
        replacement_factor = 1.25
        z_missing = replacement_factor * self._sensor_range

        for z, z_pred in zip(measurements, z_hat):
            # Reemplaza medidas "no disponibles"
            z_use = z_missing if (z is None or math.isinf(z)) else float(z)
            z_pred_use = z_missing if (z_pred is None or math.isinf(z_pred)) else float(z_pred)

            # Modelo de sensor: gaussiana centrada en la predicción
            # (asumo que tu sigma del sensor está en self._sigma_z; usa el nombre que tengas)
            probability *= self._gaussian(mu=z_pred_use, sigma=self._sigma_z, x=z_use)

        return probability

    def _sensor_rays(self, particle: tuple[float, float, float]) -> list[list[tuple[float, float]]]:
        """Determines the simulated sensor ray segments for a given particle.

        Args:
            particle: Particle pose (x, y, theta) in [m] and [rad].

        Returns: Ray segments. Format:
                 [[(x0_begin, y0_begin), (x0_end, y0_end)],
                  [(x1_begin, y1_begin), (x1_end, y1_end)],
                  ...]

        """
        x = particle[0]
        y = particle[1]
        theta = particle[2]

        # Convert sensors to world coordinates
        xw = [x + ds * math.cos(theta + phi) for ds, phi in zip(self._ds, self._phi)]
        yw = [y + ds * math.sin(theta + phi) for ds, phi in zip(self._ds, self._phi)]
        tw = [sensor[2] for sensor in self._sensors]

        rays = []

        for xs, ys, ts in zip(xw, yw, tw):
            x_end = xs + self._sensor_range * math.cos(theta + ts)
            y_end = ys + self._sensor_range * math.sin(theta + ts)
            rays.append([(xs, ys), (x_end, y_end)])
        
        return rays
