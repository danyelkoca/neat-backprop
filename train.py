"""
NEAT with backpropagation for general classification tasks.
Default task: XOR problem
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import random
from src.neat import Population, Genome, NodeGene, ConnectionGene
from src.network import Network, WeightsDict
from src.visualization import (
    plot_network_and_decision_boundary,
)
from src.datasets import (
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


def has_input_output_path(genome: Genome) -> bool:
    """Check if there is at least one valid path from any input node to any output node."""
    input_ids = [n.id for n in genome.nodes.values() if n.type == "input"]
    output_ids = [n.id for n in genome.nodes.values() if n.type == "output"]
    if not input_ids or not output_ids:
        return False
    # Build adjacency list for enabled connections
    adj = {nid: [] for nid in genome.nodes}
    for conn in genome.connections.values():
        if conn.enabled and conn.in_node in adj and conn.out_node in adj:
            adj[conn.in_node].append(conn.out_node)
    # DFS from each input to see if any output is reachable
    for start in input_ids:
        stack = [start]
        visited = set()
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            if nid in output_ids:
                return True
            stack.extend(adj[nid])
    return False


def train_single_genome(
    genome: Genome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    is_best: bool = False,
) -> Tuple[float, Optional[WeightsDict], Dict[str, float]]:
    """Train a single genome using backpropagation."""
    # Validate genome before training
    if not genome.validate():
        return (
            -1.0,
            None,
            {
                "bce_loss": float("inf"),
                "accuracy": 0.0,
                "best_epoch": 0,
                "connection_penalty": 0.0,
                "node_penalty": 0.0,
                "total_penalty": 0.0,
                "n_connections": 0.0,
                "n_hidden": 0.0,
            },
        )

    # Prune and repair as needed
    from src.neat import prune_nonfunctional

    prune_nonfunctional(genome)
    repair_network_structure(genome)

    network = Network(genome)
    best_weights = None
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    # Initialize metrics
    metrics = {"bce_loss": float("inf"), "accuracy": 0.0, "best_epoch": 0}

    # General-purpose training parameters
    max_epochs = 100
    patience = 20

    for epoch in range(max_epochs):
        bce_loss = network.train_step(X_train, y_train)
        accuracy = network.compute_binary_accuracy(X_val, y_val)

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            metrics.update(
                {
                    "bce_loss": bce_loss,
                    "accuracy": accuracy,
                    "best_epoch": epoch,
                }
            )
            best_weights = network.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Prune before metrics and plotting
    from src.neat import prune_nonfunctional

    prune_nonfunctional(genome)

    # General-purpose penalties
    n_connections = sum(1 for conn in genome.connections.values() if conn.enabled)
    n_hidden = sum(1 for node in genome.nodes.values() if node.type == "hidden")

    connection_penalty = 0.002 * n_connections
    node_penalty = 0.004 * n_hidden

    metrics.update(
        {
            "connection_penalty": connection_penalty,
            "node_penalty": node_penalty,
            "total_penalty": connection_penalty + node_penalty,
            "n_connections": float(n_connections),
            "n_hidden": float(n_hidden),
        }
    )

    # Final fitness with reduced penalties
    fitness = best_val_accuracy - (connection_penalty + node_penalty)

    return fitness, best_weights, metrics


def train_genome_wrapper(args):
    """Helper function for parallel genome training."""
    genome, X_train, y_train, X_val, y_val, is_best = args
    return train_single_genome(genome, X_train, y_train, X_val, y_val, is_best)


def repair_network_structure(genome: Genome) -> None:
    """Repair network structure by ensuring proper connectivity."""
    # First ensure we have basic nodes (input and output)
    if len(genome.nodes) == 0:
        # Add input nodes
        for i in range(2):  # For 2D input
            genome.nodes[i] = NodeGene(i, "input", "linear")
        # Add output node
        genome.nodes[2] = NodeGene(2, "output", "sigmoid")

    # Add bias node if missing
    bias_nodes = [n for n in genome.nodes.values() if n.type == "bias"]
    if not bias_nodes:
        # Find next available ID
        bias_id = max(max(genome.nodes.keys()) if genome.nodes else -1, 2) + 1
        genome.nodes[bias_id] = NodeGene(bias_id, "bias", "linear")
        # Connect bias to all hidden and output nodes
        for node in genome.nodes.values():
            if node.type in ["hidden", "output"]:
                conn_key = (bias_id, node.id)
                if conn_key not in genome.connections:
                    genome.connections[conn_key] = ConnectionGene(
                        bias_id, node.id, random.uniform(-0.5, 0.5), True, 0
                    )

    # Ensure input-output connectivity
    if len(genome.connections) == 0 or not has_input_output_path(genome):
        input_ids = [n.id for n in genome.nodes.values() if n.type == "input"]
        output_ids = [n.id for n in genome.nodes.values() if n.type == "output"]
        hidden_ids = [n.id for n in genome.nodes.values() if n.type == "hidden"]

        if input_ids and output_ids:
            # First ensure each input is connected to something
            for in_id in input_ids:
                # 50% chance to connect through hidden node if available
                if hidden_ids and random.random() < 0.5:
                    mid_id = random.choice(hidden_ids)
                    out_id = random.choice(output_ids)
                    genome.connections[(in_id, mid_id)] = ConnectionGene(
                        in_id, mid_id, random.uniform(-0.5, 0.5), True, 0
                    )
                    genome.connections[(mid_id, out_id)] = ConnectionGene(
                        mid_id, out_id, random.uniform(-0.5, 0.5), True, 0
                    )
                else:
                    out_id = random.choice(output_ids)
                    genome.connections[(in_id, out_id)] = ConnectionGene(
                        in_id, out_id, random.uniform(-0.5, 0.5), True, 0
                    )

            # Ensure each hidden node has at least one input and one output connection
            for hid in hidden_ids:
                has_input = any(
                    c.out_node == hid and c.enabled for c in genome.connections.values()
                )
                has_output = any(
                    c.in_node == hid and c.enabled for c in genome.connections.values()
                )

                if not has_input:
                    in_id = random.choice(input_ids)
                    genome.connections[(in_id, hid)] = ConnectionGene(
                        in_id, hid, random.uniform(-0.5, 0.5), True, 0
                    )
                if not has_output:
                    out_id = random.choice(output_ids)
                    genome.connections[(hid, out_id)] = ConnectionGene(
                        hid, out_id, random.uniform(-0.5, 0.5), True, 0
                    )

    return genome


def main(
    task="xor",
    n_samples: int = 3000,
    visualize: bool = True,
    parallel: bool = True,
):

    if task not in [
        "xor",
        "two_circles",
        "two_gaussians",
        "spiral",
    ]:
        raise ValueError(
            f"Unknown task: {task}. Supported tasks: xor, two_circles, two_gaussians, spiral"
        )

    dataset_fn = None
    if task == "xor":
        dataset_fn = generate_xor
    elif task == "two_circles":
        dataset_fn = generate_two_circles
    elif task == "two_gaussians":
        dataset_fn = generate_two_gaussians
    elif task == "spiral":
        dataset_fn = generate_spiral

    """Main training loop for NEAT with backpropagation."""
    # Create directories for visualization
    if visualize:
        os.makedirs("results", exist_ok=True)
        # os.makedirs("results/decision_boundary", exist_ok=True)

    # Load dataset
    print(f"\nGenerating dataset using {dataset_fn.__name__}...")
    X, y = dataset_fn(n_samples=n_samples)

    # Split into train/val
    n_train = int(0.6666666667 * len(X))
    indices = np.random.permutation(len(X))
    X_train = X[indices[:n_train]]
    X_val = X[indices[n_train:]]
    y_train = y[indices[:n_train]]
    y_val = y[indices[n_train:]]

    print(f"Dataset size: {len(X)} samples")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Initialize population with optimized parameters
    pop_size = 100  # Increased for better exploration
    input_size = X.shape[1]
    output_size = 1

    # Initialize population with balanced parameters
    population = Population(
        size=pop_size,
        input_size=input_size,
        output_size=output_size,
        compatibility_threshold=2.0,  # More strict for better speciation
        elitism=1,  # Keep two best genomes
    )

    # Training parameters
    n_generations = 50
    generations_without_improvement = 0
    best_fitness = float("-inf")
    best_ever_genome = None
    min_species_size = 3  # Increased minimum species size
    target_n_species = 5  # Target more species
    species_adjustment_interval = 2  # Adjust more frequently
    patience = 15  # Increased patience

    # Encourage more initial diversity
    for genome in population.genomes[2:]:  # Skip first two
        # Encourage more structural mutations for XOR
        genome.mutate(
            weight_mutation_rate=0.85,
            add_node_rate=0.35,
            add_connection_rate=0.55,
        )

    # Main evolution loop
    for gen in range(n_generations):
        print(f"\nGeneration {gen}/{n_generations}")

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
        best_metrics_this_gen = None

        # Update genomes with results
        for genome, (fitness, weights, metrics) in zip(population.genomes, results):
            genome.fitness = fitness
            genome.metrics = metrics  # Store metrics for later reporting

            # Track best genome in this generation
            if fitness > best_fitness:
                best_fitness = fitness
                generations_without_improvement = 0
                best_ever_genome = genome
                best_ever_weights = weights
                best_ever_metrics = metrics
                generation_improved = True
                best_genome_this_gen = genome
                best_weights_this_gen = weights
                best_metrics_this_gen = metrics

        # After all genomes are evaluated, print/visualize ONCE if there was an improvement
        if generation_improved:
            print(f"\nGeneration {gen}: Improvement found!")
            print("Best genome metrics this generation:")
            for key, value in best_metrics_this_gen.items():
                print(f"  {key}: {value:.4f}")
            print(f"Best fitness: {best_fitness:.4f}")
            print(f"Generations without improvement: {generations_without_improvement}")
            print(f"Species in generation {gen}: {len(population.species)} total")
            for i, species in enumerate(population.species, 1):
                print(f"  Species {i}: {len(species.members)} members")

            if visualize and best_genome_this_gen:
                network = Network(best_genome_this_gen)
                # Ensure bias node exists in best_genome_this_gen before visualization
                bias_nodes = [
                    n for n in best_genome_this_gen.nodes.values() if n.type == "bias"
                ]
                if not bias_nodes:
                    bias_id = max(best_genome_this_gen.nodes.keys()) + 1
                    best_genome_this_gen.nodes[bias_id] = NodeGene(
                        bias_id, "bias", "linear"
                    )
                    for node in best_genome_this_gen.nodes.values():
                        if node.type in ("hidden", "output"):
                            conn_key = (bias_id, node.id)
                            if conn_key not in best_genome_this_gen.connections:
                                best_genome_this_gen.connections[conn_key] = (
                                    ConnectionGene(
                                        bias_id, node.id, random.uniform(-1, 1), True, 0
                                    )
                                )
                if best_weights_this_gen is not None:
                    network.set_weights(best_weights_this_gen)
                # Combined plot: network + raw decision boundary
                from src.visualization import plot_network_and_decision_boundary

                plot_network_and_decision_boundary(
                    best_genome_this_gen, network, X_val, y_val, task, gen
                )

        # Update stagnation counter
        if not generation_improved:
            generations_without_improvement += 1

        # Adjust compatibility threshold to maintain target number of species
        if gen % species_adjustment_interval == 0:
            n_species = len(population.species)
            if n_species == 0:
                # If no species, drastically lower the threshold to encourage speciation
                population.compatibility_threshold *= 0.5
            elif n_species < target_n_species:
                population.compatibility_threshold *= (
                    0.95  # Decrease threshold to create more species
                )
            elif n_species > target_n_species:
                population.compatibility_threshold *= (
                    1.05  # Increase threshold to reduce species
                )

        # Check species sizes and force diversity if needed
        species_sizes = [len(s.members) for s in population.species]
        need_diversity = len(species_sizes) == 0 or (
            len(species_sizes) > 0 and min(species_sizes) < min_species_size
        )

        if need_diversity:
            # Calculate adaptive mutation rates based on current fitness and stagnation
            base_weight_rate = 0.8
            base_node_rate = 0.3
            base_conn_rate = 0.5

            # Increase rates if we're stagnating
            stagnation_factor = min(2.0, 1.0 + (generations_without_improvement * 0.1))

            weight_rate = min(0.95, base_weight_rate * stagnation_factor)
            node_rate = min(0.5, base_node_rate * stagnation_factor)
            conn_rate = min(0.7, base_conn_rate * stagnation_factor)

            # Apply mutations with adaptive rates
            for genome in population.genomes:
                # --- PATCH: Prevent best genome from mutating ---
                if best_ever_genome is not None and genome is best_ever_genome:
                    continue  # Do not mutate the best genome
                if random.random() < 0.3:  # 30% chance for each genome
                    success = genome.mutate(
                        weight_mutation_rate=weight_rate,
                        add_node_rate=node_rate,
                        add_connection_rate=conn_rate,
                    )
                    # If mutation failed, try simpler mutation
                    if not success:
                        genome.mutate(
                            weight_mutation_rate=0.9,
                            add_node_rate=0.0,
                            add_connection_rate=0.3,
                        )

        # Enforce structural diversity by checking for duplicates
        structures = {}
        for genome in population.genomes:
            structure_hash = str(
                sorted([(n.type, n.activation) for n in genome.nodes.values()])
            ) + str(
                sorted(
                    [
                        (c.in_node, c.out_node)
                        for c in genome.connections.values()
                        if c.enabled
                    ]
                )
            )

            if structure_hash in structures:
                # If duplicate found, mutate this genome (but not the best)
                if best_ever_genome is not None and genome is best_ever_genome:
                    continue
                genome.mutate(
                    weight_mutation_rate=0.9, add_node_rate=0.3, add_connection_rate=0.5
                )
            structures[structure_hash] = True

        # Evolve population
        population.evolve()


if __name__ == "__main__":
    # Run with parallel processing for faster execution
    main(task="two_circles", n_samples=3000, visualize=True, parallel=True)
