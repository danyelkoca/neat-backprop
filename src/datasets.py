"""
Dataset generators for 2D classification tasks.
All generators follow the same API:
- Input: n_samples (int)
- Output: Tuple[np.ndarray, np.ndarray] for (X, y)
- X shape: (n_samples, 2)
- y shape: (n_samples, 1)
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple


def generate_xor(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate XOR dataset with n_samples points.

    Args:
        n_samples: Number of samples to generate

    Returns:
        X: Input points with shape (n_samples, 2)
        y: Binary labels with shape (n_samples, 1)
    """
    rng = np.random.RandomState(42)

    # Define the four XOR corners
    corners = [
        ([-0.5, -0.5], 0),  # (-,-) -> 0
        ([-0.5, 0.5], 1),  # (-,+) -> 1
        ([0.5, -0.5], 1),  # (+,-) -> 1
        ([0.5, 0.5], 0),  # (+,+) -> 0
    ]

    # Ensure at least 5% of samples are corners
    corner_samples = max(5, n_samples // 20)
    remaining_samples = n_samples - 4 * corner_samples
    points_per_quadrant = remaining_samples // 4

    X = []
    y = []

    # Add corners with small noise
    for point, label in corners:
        for _ in range(corner_samples):
            noise = rng.normal(0, 0.05, 2)
            X.append([point[0] + noise[0], point[1] + noise[1]])
            y.append(label)

    # Add remaining points in each quadrant
    quadrants = [
        (lambda: (rng.uniform(-0.9, -0.1), rng.uniform(-0.9, -0.1)), 0),  # (-,-) -> 0
        (lambda: (rng.uniform(-0.9, -0.1), rng.uniform(0.1, 0.9)), 1),  # (-,+) -> 1
        (lambda: (rng.uniform(0.1, 0.9), rng.uniform(-0.9, -0.1)), 1),  # (+,-) -> 1
        (lambda: (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)), 0),  # (+,+) -> 0
    ]

    for gen_point, label in quadrants:
        for _ in range(points_per_quadrant):
            x1, x2 = gen_point()
            X.append([x1, x2])
            y.append(label)

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    # Shuffle the dataset
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def generate_two_circles(
    n_samples: int = 1000, noise: float = 0.05, factor: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two concentric circles dataset with clear separation.

    Args:
        n_samples: Number of points to generate
        noise: Standard deviation of Gaussian noise (default: 0.05 for clearer separation)
        factor: Scale factor between inner and outer circle (default: 0.4 for bigger gap)

    Returns:
        X: Input points with shape (n_samples, 2)
        y: Binary labels with shape (n_samples, 1)
    """
    rng = np.random.RandomState(42)

    # Split samples between outer and inner circle
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Generate larger circle (scaled up by 1.5)
    linspace = np.linspace(0, 2 * np.pi, n_samples_out)
    outer_circ_x = 1.5 * np.cos(linspace)
    outer_circ_y = 1.5 * np.sin(linspace)
    outer_circle = np.vstack([outer_circ_x, outer_circ_y]).T

    # Generate smaller circle
    linspace = np.linspace(0, 2 * np.pi, n_samples_in)
    inner_circ_x = factor * np.cos(linspace)
    inner_circ_y = factor * np.sin(linspace)
    inner_circle = np.vstack([inner_circ_x, inner_circ_y]).T

    # Add noise
    X = np.vstack([outer_circle, inner_circle])
    X += rng.normal(scale=noise, size=X.shape)
    X = X.astype(np.float32)

    # Generate labels
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])
    y = y.reshape(-1, 1).astype(np.float32)

    # Shuffle
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def generate_spiral(
    n_samples: int = 1000, noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate spiral dataset."""
    n = n_samples // 2

    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi

    r_a = 2 * theta + np.pi
    data_a = (
        np.column_stack([r_a * np.cos(theta), r_a * np.sin(theta)])
        + np.random.randn(n, 2) * noise
    )

    r_b = -2 * theta - np.pi
    data_b = (
        np.column_stack([r_b * np.cos(theta), r_b * np.sin(theta)])
        + np.random.randn(n, 2) * noise
    )

    X = np.vstack([data_a, data_b])
    y = np.hstack([np.ones(n), np.zeros(n)])

    return X, y.reshape(-1, 1)


def generate_two_gaussians(
    n_samples: int = 1000, noise: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two Gaussian clusters dataset."""
    n = n_samples // 2

    X1 = np.random.normal(-2, noise, (n, 2))
    X2 = np.random.normal(2, noise, (n, 2))

    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n), np.zeros(n)])

    return X, y.reshape(-1, 1)


def prepare_dataset(
    X: np.ndarray, y: np.ndarray, batch_size: int = 32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert numpy arrays to JAX arrays and create batches."""
    return jnp.array(X), jnp.array(y)
