"""
NEAT with backpropagation for general classification tasks.
Default task: XOR problem
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Optional, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from neat_backprop.neat import Population, Genome
from neat_backprop.network import Network, WeightsDict
from neat_backprop.visualization import plot_network, plot_decision_boundary
from neat_backprop.datasets import (
    generate_xor,
    generate_two_circles,
    generate_two_gaussians,
    generate_spiral,
)


def get_network_complexity(genome: Genome) -> Tuple[int, int]:
    """Calculate network complexity metrics."""
    n_enabled_connections = sum(
        1 for conn in genome.connections.values() if conn.enabled
    )
    n_hidden_nodes = sum(1 for node in genome.nodes.values() if node.type == "hidden")
    return n_enabled_connections, n_hidden_nodes


def train_single_genome(
    genome: Genome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    is_best: bool = False,
) -> Tuple[float, Optional[WeightsDict], Dict[str, float]]:
    """Train a single genome using backpropagation."""
    # Initialize network (only show debug info if this is the current best)
    network = Network(genome, debug=is_best)
    best_weights = None
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    # Train the network
    metrics = {"mse_loss": float("inf"), "accuracy": 0.0, "best_epoch": 0}

    # Train with early stopping
    max_epochs = 200
    patience = 20

    for epoch in range(max_epochs):
        # Perform training step
        mse_loss = network.train_step(X_train, y_train)
        accuracy = network.compute_binary_accuracy(X_val, y_val)
        # print(
        #     f"Epoch {epoch+1}/{max_epochs}, MSE Loss: {mse_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
        # )
        # Update best metrics if improved
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            metrics["mse_loss"] = mse_loss
            metrics["accuracy"] = accuracy
            metrics["best_epoch"] = epoch
            best_weights = network.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping for individual training
        if epochs_without_improvement >= patience:
            break

    # Calculate complexity penalties
    n_connections, n_hidden = get_network_complexity(genome)
    connection_penalty = 0.0001 * n_connections  # Reduced penalties
    node_penalty = 0.0003 * n_hidden
    total_penalty = connection_penalty + node_penalty

    # Final metrics
    metrics.update(
        {
            "connection_penalty": connection_penalty,
            "node_penalty": node_penalty,
            "total_penalty": total_penalty,
            "n_connections": n_connections,
            "n_hidden": n_hidden,
        }
    )

    # Calculate fitness (higher is better)
    # Use both accuracy and loss in fitness, with reduced penalty impact
    fitness = (best_val_accuracy * 10.0) - metrics["mse_loss"] - total_penalty

    # Save the best weights to the genome for inheritance
    if best_weights is not None:
        genome.saved_weights = {k: v.copy() for k, v in best_weights.items()}

    return fitness, best_weights, metrics


def train_genome_wrapper(args):
    """Helper function for parallel genome training."""
    genome, X_train, y_train, X_val, y_val, is_best = args
    return train_single_genome(genome, X_train, y_train, X_val, y_val, is_best)


def main(
    dataset_fn=generate_xor,
    n_samples: int = 1000,
    visualize: bool = True,
    parallel: bool = True,
):
    """Main training loop for NEAT with backpropagation."""
    # Create directories for visualization
    if visualize:
        os.makedirs("graphs/network", exist_ok=True)
        os.makedirs("graphs/decision_boundary", exist_ok=True)

    # Load dataset
    print(f"\nGenerating dataset using {dataset_fn.__name__}...")
    X, y = dataset_fn(n_samples=n_samples)

    # Split into train/val (80/20 split)
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    X_train = X[indices[:n_train]]
    X_val = X[indices[n_train:]]
    y_train = y[indices[:n_train]]
    y_val = y[indices[n_train:]]

    print(f"Dataset size: {len(X)} samples")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Initialize population
    pop_size = 20
    input_size = X.shape[1]
    output_size = 1
    population = Population(pop_size, input_size, output_size)

    # Training parameters
    n_generations = 50
    generations_without_improvement = 0
    best_fitness = float("-inf")
    best_ever_genome = None
    patience = 25  # Increased patience for evolution

    # Main evolution loop
    for gen in range(n_generations):
        print(f"\nGeneration {gen+1}/{n_generations}")

        # Prepare training arguments
        train_args = []
        for genome in population.genomes:
            is_best = (
                genome.fitness == best_fitness if hasattr(genome, "fitness") else False
            )
            train_args.append((genome, X_train, y_train, X_val, y_val, is_best))

        if parallel:
            # Get number of CPUs while leaving one free for system
            n_processes = max(1, multiprocessing.cpu_count() - 1)
            print(f"Training in parallel with {n_processes} processes...")

            # Train genomes in parallel
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                results = list(executor.map(train_genome_wrapper, train_args))
        else:
            # Train genomes sequentially
            results = [train_genome_wrapper(args) for args in train_args]

        # Track if any improvement this generation
        generation_improved = False

        # Update genomes with results
        for genome, (fitness, weights, metrics) in zip(population.genomes, results):
            genome.fitness = fitness
            if weights is not None:
                genome.saved_weights = weights

            if fitness > best_fitness:
                best_fitness = fitness
                best_ever_genome = genome.clone()
                generation_improved = True

                print("\nNew best genome metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

        # Update stagnation counter for the whole generation
        if not generation_improved:
            generations_without_improvement += 1
        else:
            generations_without_improvement = 0

            # Log improvement details
            print(f"Generation {gen+1}: Improvement found!")
            print(f"Best fitness: {best_fitness:.4f}")
            print(f"Generations without improvement: {generations_without_improvement}")

        # Early stopping
        if generations_without_improvement >= patience:
            print(f"\nStopping early - No improvement for {patience} generations")
            break

        # Visualize best network using validation data
        if visualize and best_ever_genome and (generation_improved or gen == 0):
            network = Network(
                best_ever_genome, debug=True
            )  # Show debug info for best network
            plot_network(best_ever_genome, title=f"graphs/network/gen_{gen+1}")
            plot_decision_boundary(
                network, X_val, y_val, title=f"graphs/decision_boundary/gen_{gen+1}"
            )

        # Evolve population
        population.evolve()


if __name__ == "__main__":
    # Run with parallel processing for faster execution
    main(dataset_fn=generate_spiral, n_samples=1000, visualize=True, parallel=True)
