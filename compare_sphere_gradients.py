#!/usr/bin/env python3
"""Compare standard-gradient and natural-gradient ascent on the unit sphere.

Coordinates follow Earth latitude/longitude convention:
  lat in [-pi/2, pi/2], lon in (-pi, pi]

The objective is the spherical distance from a fixed target point A.  The maximum
is attained at A's antipode.  We compare efficiency by iteration count, not wall
clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class Point:
    lat: float
    lon: float


@dataclass
class Result:
    method: str
    iterations: int
    converged: bool
    final: Point
    final_distance: float
    error_to_antipode: float


def wrap_lon(lon: float) -> float:
    """Wrap longitude to (-pi, pi]."""
    return (lon + math.pi) % (2.0 * math.pi) - math.pi


def normalize_lat_lon(lat: float, lon: float) -> Point:
    """Keep coordinates valid if an update crosses a pole."""
    lon = wrap_lon(lon)

    while lat > math.pi / 2.0 or lat < -math.pi / 2.0:
        if lat > math.pi / 2.0:
            lat = math.pi - lat
            lon = wrap_lon(lon + math.pi)
        elif lat < -math.pi / 2.0:
            lat = -math.pi - lat
            lon = wrap_lon(lon + math.pi)

    return Point(lat=lat, lon=lon)


def antipode(point: Point) -> Point:
    return Point(lat=-point.lat, lon=wrap_lon(point.lon + math.pi))


def dot_on_sphere(point: Point, target: Point) -> float:
    """Dot product between two unit vectors represented by lat/lon."""
    return (
        math.sin(point.lat) * math.sin(target.lat)
        + math.cos(point.lat) * math.cos(target.lat) * math.cos(point.lon - target.lon)
    )


def spherical_distance(point: Point, target: Point) -> float:
    dot = float(np.clip(dot_on_sphere(point, target), -1.0, 1.0))
    return math.acos(dot)


def distance_gradient(point: Point, target: Point) -> np.ndarray:
    """Standard coordinate gradient [d distance / d lat, d distance / d lon]."""
    lat, lon = point.lat, point.lon
    target_lat, target_lon = target.lat, target.lon
    delta_lon = lon - target_lon

    dot = float(np.clip(dot_on_sphere(point, target), -1.0 + EPS, 1.0 - EPS))
    scale = -1.0 / math.sqrt(max(EPS, 1.0 - dot * dot))

    d_dot_d_lat = (
        math.cos(lat) * math.sin(target_lat)
        - math.sin(lat) * math.cos(target_lat) * math.cos(delta_lon)
    )
    d_dot_d_lon = -math.cos(lat) * math.cos(target_lat) * math.sin(delta_lon)

    return scale * np.array([d_dot_d_lat, d_dot_d_lon], dtype=float)


def standard_gradient(point: Point, target: Point) -> np.ndarray:
    return distance_gradient(point, target)


def natural_gradient(point: Point, target: Point) -> np.ndarray:
    """Natural gradient under the sphere metric ds^2 = dlat^2 + cos(lat)^2 dlon^2."""
    grad = distance_gradient(point, target)
    cos_lat = max(abs(math.cos(point.lat)), 1e-8)
    return np.array([grad[0], grad[1] / (cos_lat * cos_lat)], dtype=float)


def ascend(
    target: Point,
    start: Point,
    gradient_fn: Callable[[Point, Point], np.ndarray],
    method: str,
    learning_rate: float,
    decay_rate: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100_000,
) -> Result:
    point = start
    target_antipode = antipode(target)

    for iteration in range(max_iterations + 1):
        current_distance = spherical_distance(point, target)
        error = spherical_distance(point, target_antipode)
        if error < tolerance:
            return Result(
                method=method,
                iterations=iteration,
                converged=True,
                final=point,
                final_distance=current_distance,
                error_to_antipode=error,
            )

        gradient = gradient_fn(point, target)
        if np.linalg.norm(gradient) < 1e-10:
            gradient = np.array([1.0, 1.0], dtype=float)

        step_size = learning_rate * (decay_rate**iteration)
        point = normalize_lat_lon(
            point.lat + step_size * gradient[0],
            point.lon + step_size * gradient[1],
        )

    return Result(
        method=method,
        iterations=max_iterations,
        converged=False,
        final=point,
        final_distance=spherical_distance(point, target),
        error_to_antipode=spherical_distance(point, target_antipode),
    )


def random_point(rng: np.random.Generator) -> Point:
    """Sample uniformly on the unit sphere, returned as lat/lon."""
    sin_lat = rng.uniform(-1.0, 1.0)
    lon = rng.uniform(-math.pi, math.pi)
    return Point(lat=math.asin(sin_lat), lon=lon)


def deg(point: Point) -> tuple[float, float]:
    return math.degrees(point.lat), math.degrees(point.lon)


def main() -> None:
    rng = np.random.default_rng(20260623)
    targets = [random_point(rng) for _ in range(10)]

    # Fixed schedule: step_size_k = learning_rate * decay_rate**k.
    learning_rates = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.5]
    decay_rate = 0.999
    tolerance = 1e-6

    print(f"decay_rate={decay_rate}, tolerance={tolerance}")
    print("Each start point is exactly its target A.")
    print("The same 10 random target points are reused for every learning rate.\n")

    summary = []

    for learning_rate in learning_rates:
        print(f"learning_rate={learning_rate}")
        print(
            "idx | target_lat | target_lon | standard_iter | natural_iter | "
            "standard_error | natural_error"
        )
        print("-" * 92)

        standard_iterations = []
        natural_iterations = []

        for index, target in enumerate(targets, start=1):
            start = target
            standard_result = ascend(
                target,
                start,
                standard_gradient,
                "standard",
                learning_rate,
                decay_rate,
                tolerance,
            )
            natural_result = ascend(
                target,
                start,
                natural_gradient,
                "natural",
                learning_rate,
                decay_rate,
                tolerance,
            )

            standard_iterations.append(standard_result.iterations)
            natural_iterations.append(natural_result.iterations)

            target_lat, target_lon = deg(target)
            print(
                f"{index:3d} | "
                f"{target_lat:10.4f} | "
                f"{target_lon:10.4f} | "
                f"{standard_result.iterations:13d} | "
                f"{natural_result.iterations:12d} | "
                f"{standard_result.error_to_antipode:14.2e} | "
                f"{natural_result.error_to_antipode:13.2e}"
            )

        standard_mean = float(np.mean(standard_iterations))
        natural_mean = float(np.mean(natural_iterations))
        summary.append((learning_rate, standard_mean, natural_mean))

        print("-" * 92)
        print(
            f"avg | {'':10s} | {'':10s} | "
            f"{standard_mean:13.1f} | "
            f"{natural_mean:12.1f} |\n"
        )

    print("summary")
    print("learning_rate | standard_avg_iter | natural_avg_iter")
    print("-" * 51)
    for learning_rate, standard_mean, natural_mean in summary:
        print(f"{learning_rate:13.3f} | {standard_mean:17.1f} | {natural_mean:16.1f}")


if __name__ == "__main__":
    main()
