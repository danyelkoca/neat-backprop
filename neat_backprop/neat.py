"""
Core NEAT implementation for topology evolution.
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random


@dataclass
class NodeGene:
    id: int
    type: str  # 'input', 'hidden', 'output'
    activation: str  # 'relu', 'sigmoid'


@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int


def get_distance(
    genome1: "Genome",
    genome2: "Genome",
    c1: float = 0.5,  # Reduced weight for disjoint connections
    c2: float = 0.5,  # Reduced weight for disjoint nodes
    c3: float = 1.0,  # Increased weight for weight differences
) -> float:
    """Compute genetic distance between two genomes with improved diversity measures."""
    # Get all innovation numbers and node IDs
    innovations1 = set(
        conn.innovation for conn in genome1.connections.values() if conn.enabled
    )
    innovations2 = set(
        conn.innovation for conn in genome2.connections.values() if conn.enabled
    )
    nodes1 = set(node.id for node in genome1.nodes.values() if node.type == "hidden")
    nodes2 = set(node.id for node in genome2.nodes.values() if node.type == "hidden")

    # Find matching and disjoint genes
    matching_conns = innovations1 & innovations2
    disjoint_conns = innovations1 ^ innovations2
    matching_nodes = nodes1 & nodes2
    disjoint_nodes = nodes1 ^ nodes2

    # Calculate average weight difference for matching genes
    weight_diff = 0.0
    activation_diff = 0.0
    if matching_conns:
        for innovation in matching_conns:
            conn1 = next(
                c for c in genome1.connections.values() if c.innovation == innovation
            )
            conn2 = next(
                c for c in genome2.connections.values() if c.innovation == innovation
            )
            weight_diff += abs(conn1.weight - conn2.weight)
        weight_diff /= len(matching_conns)

    # Calculate activation function differences for hidden nodes only
    if matching_nodes:
        for node_id in matching_nodes:
            node1 = genome1.nodes[node_id]
            node2 = genome2.nodes[node_id]
            if node1.activation != node2.activation:
                activation_diff += 1
        activation_diff /= len(matching_nodes)

    # Calculate distance with size normalization
    n = max(len(innovations1), len(innovations2))
    m = max(len(nodes1), len(nodes2))
    if n < 10:  # normalize by 1 for very small networks
        n = 1
    if m < 5:
        m = 1

    distance = (
        c1 * len(disjoint_conns) / n  # Connection topology difference
        + c2 * len(disjoint_nodes) / m  # Node topology difference
        + c3
        * (
            weight_diff + 0.5 * activation_diff
        )  # Weight and activation differences, reduced activation impact
    )

    return distance


from typing import Dict, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray


class Genome:
    """Genome class representing the structure and weights of a neural network."""

    # Class variables for innovation and node counting
    innovation_counter = 0
    node_counter = 0

    def __init__(
        self,
        nodes: Dict[int, NodeGene],
        connections: Dict[Tuple[int, int], ConnectionGene],
        fitness: float = float("-inf"),
        adjusted_fitness: float = float("-inf"),
        saved_weights: Optional[Dict[Tuple[int, int], Union[float, NDArray]]] = None,
    ):
        self.nodes = nodes
        self.connections = connections
        self.fitness = fitness
        self.adjusted_fitness = adjusted_fitness
        self.saved_weights = saved_weights

    def clone(self) -> "Genome":
        """Create a deep copy of the genome."""
        new_nodes = {
            k: NodeGene(v.id, v.type, v.activation) for k, v in self.nodes.items()
        }
        new_connections = {
            k: ConnectionGene(v.in_node, v.out_node, v.weight, v.enabled, v.innovation)
            for k, v in self.connections.items()
        }
        return Genome(
            new_nodes,
            new_connections,
            self.fitness,
            self.adjusted_fitness,
            self.saved_weights.copy() if self.saved_weights is not None else None,
        )

    def mutate(
        self,
        weight_mutation_rate: float = 0.8,
        weight_perturbation: float = 0.2,
        add_node_rate: float = 0.3,
        add_connection_rate: float = 0.5,
        toggle_rate: float = 0.1,
        activation_rate: float = 0.2,
    ):
        """Apply mutation operators."""
        # Mutate connection weights
        for conn in self.connections.values():
            if random.random() < weight_mutation_rate:
                if random.random() < 0.9:  # 90% chance of small perturbation
                    conn.weight += np.random.normal(0, weight_perturbation)
                else:  # 10% chance of random weight
                    conn.weight = np.random.normal(0, 1)

            # Toggle connection
            if random.random() < toggle_rate:
                conn.enabled = not conn.enabled

        # Mutate activation functions
        for node in self.nodes.values():
            if node.type == "hidden" and random.random() < activation_rate:
                node.activation = random.choice(
                    [
                        "relu",
                        "sigmoid",
                        "tanh",
                        "leaky_relu",
                        "elu",
                        "softplus",
                        "swish",
                        "gelu",
                    ]
                )

        # Add new node
        if random.random() < add_node_rate:
            self._mutate_add_node()

        # Add new connection
        if random.random() < add_connection_rate:
            self._mutate_add_connection()

    def _mutate_add_node(self):
        """Add a new node by splitting an existing connection."""
        # Choose a random enabled connection
        enabled_connections = [c for c in self.connections.values() if c.enabled]
        if not enabled_connections:
            return

        # Prefer connections between input and output
        io_connections = [
            c
            for c in enabled_connections
            if self.nodes[c.in_node].type == "input"
            and self.nodes[c.out_node].type == "output"
        ]

        if (
            io_connections and random.random() < 0.6
        ):  # 60% chance to split I/O connection
            conn = random.choice(io_connections)
        else:
            conn = random.choice(enabled_connections)

        conn.enabled = False

        # Create new node with random activation
        new_node_id = Genome.node_counter
        # Choose activation with bias towards successful ones
        activation_probs = {
            "relu": 0.4,  # ReLU is good for deep networks
            "sigmoid": 0.3,  # Sigmoid for binary classification
            "tanh": 0.3,  # Tanh for normalized outputs
        }
        activation = random.choices(
            list(activation_probs.keys()), list(activation_probs.values())
        )[0]
        self.nodes[new_node_id] = NodeGene(new_node_id, "hidden", activation)
        Genome.node_counter += 1

        # Create new connections with improved weight initialization
        # Input -> New node: Use Kaiming/He initialization for ReLU
        input_scale = np.sqrt(2.0) if activation == "relu" else 1.0
        w1 = np.random.normal(0, input_scale / np.sqrt(len(self.nodes)))
        self.connections[(conn.in_node, new_node_id)] = ConnectionGene(
            conn.in_node, new_node_id, w1, True, Genome.innovation_counter
        )
        Genome.innovation_counter += 1

        # New node -> Output: Preserve the approximate behavior
        # Scale the original weight to maintain output magnitude
        w2 = conn.weight * np.sqrt(2.0)  # Compensate for the new nonlinearity
        self.connections[(new_node_id, conn.out_node)] = ConnectionGene(
            new_node_id, conn.out_node, w2, True, Genome.innovation_counter
        )
        Genome.innovation_counter += 1

    def _mutate_add_connection(self):
        """Add a new connection between existing nodes."""
        # Get nodes by type
        input_nodes = [n.id for n in self.nodes.values() if n.type == "input"]
        hidden_nodes = [n.id for n in self.nodes.values() if n.type == "hidden"]
        output_nodes = [n.id for n in self.nodes.values() if n.type == "output"]

        # Define valid connection sources and targets
        sources = input_nodes + hidden_nodes  # Input and hidden nodes can be sources
        targets = hidden_nodes + output_nodes  # Hidden and output nodes can be targets

        if not hidden_nodes:  # If no hidden nodes, only allow input->output
            sources = input_nodes
            targets = output_nodes

        # Try to find an unconnected pair
        attempts = 0
        while attempts < 20:  # Limit attempts to prevent infinite loops
            in_node = random.choice(sources)
            out_node = random.choice(targets)

            # Skip if:
            # 1. Connection already exists
            # 2. Same node
            # 3. Would create a cycle
            # 4. Input to input
            # 5. Output to output
            if (
                (in_node, out_node) not in self.connections
                and in_node != out_node
                and not self._would_create_cycle(in_node, out_node)
            ):

                weight = np.random.normal(0, 1)
                self.connections[(in_node, out_node)] = ConnectionGene(
                    in_node, out_node, weight, True, Genome.innovation_counter
                )
                Genome.innovation_counter += 1
                break
            attempts += 1

        # After adding a new connection, ensure every output node has at least one incoming connection
        output_nodes = [n for n in self.nodes.values() if n.type == "output"]
        input_nodes = [n for n in self.nodes.values() if n.type == "input"]
        hidden_nodes = [n for n in self.nodes.values() if n.type == "hidden"]
        for output in output_nodes:
            incoming = [
                conn
                for conn in self.connections.values()
                if conn.out_node == output.id and conn.enabled
            ]
            if not incoming:
                possible_sources = input_nodes + hidden_nodes
                if possible_sources:
                    source = random.choice(possible_sources)
                    self.innovation_counter += 1
                    fan_in = len(input_nodes) + len(hidden_nodes)
                    fan_out = len(output_nodes) + len(hidden_nodes)
                    limit = (
                        np.sqrt(6 / (fan_in + fan_out))
                        if (fan_in + fan_out) > 0
                        else 1.0
                    )
                    weight = random.uniform(-limit, limit)
                    genome.connections[(source.id, output.id)] = ConnectionGene(
                        source.id,
                        output.id,
                        weight,
                        True,
                        self.innovation_counter,
                    )

    def _would_create_cycle(self, in_node: int, out_node: int) -> bool:
        """Check if adding a connection would create a cycle."""
        visited = set()

        def visit(node_id: int) -> bool:
            if node_id == in_node:  # Would create a cycle
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            # Check all outgoing connections
            for conn in self.connections.values():
                if conn.enabled and conn.in_node == node_id:
                    if visit(conn.out_node):
                        return True
            return False

        # Start DFS from the out_node
        return visit(out_node)

    @staticmethod
    def crossover(parent1: "Genome", parent2: "Genome") -> "Genome":
        """Create a new genome through crossover of two parents."""
        # Ensure parent1 is the fitter parent
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1

        # Create child genome with empty dicts that we'll fill in
        child = Genome({}, {})

        # Inherit saved weights from the fitter parent if available
        if parent1.saved_weights is not None:
            child.saved_weights = {
                k: v.copy() for k, v in parent1.saved_weights.items()
            }

        # Get all nodes from both parents
        all_nodes = set(parent1.nodes.keys())
        all_nodes.update(parent2.nodes.keys())

        for node_id in all_nodes:
            node1 = parent1.nodes.get(node_id)
            node2 = parent2.nodes.get(node_id)

            if node1 and node2:  # Node in both parents
                # Inherit randomly from either parent
                node = node1 if random.random() < 0.5 else node2
                child.nodes[node_id] = NodeGene(node.id, node.type, node.activation)
            elif node1:  # Node only in parent1
                child.nodes[node_id] = NodeGene(node1.id, node1.type, node1.activation)
            elif node2:  # Node only in parent2
                child.nodes[node_id] = NodeGene(node2.id, node2.type, node2.activation)

        # Inherit connections
        child.connections = {}

        # Get all innovation numbers
        all_innovations = set(conn.innovation for conn in parent1.connections.values())
        all_innovations.update(conn.innovation for conn in parent2.connections.values())

        for innovation in all_innovations:
            conn1 = next(
                (
                    conn
                    for conn in parent1.connections.values()
                    if conn.innovation == innovation
                ),
                None,
            )
            conn2 = next(
                (
                    conn
                    for conn in parent2.connections.values()
                    if conn.innovation == innovation
                ),
                None,
            )

            if conn1 and conn2:  # Matching gene
                # Inherit randomly from either parent
                conn = conn1 if random.random() < 0.5 else conn2
                child.connections[(conn.in_node, conn.out_node)] = ConnectionGene(
                    conn.in_node,
                    conn.out_node,
                    conn.weight,
                    conn.enabled,
                    conn.innovation,
                )
            elif conn1:  # Disjoint or excess from parent1 (fitter parent)
                # 80% chance to inherit from fitter parent
                if random.random() < 0.8:
                    child.connections[(conn1.in_node, conn1.out_node)] = ConnectionGene(
                        conn1.in_node,
                        conn1.out_node,
                        conn1.weight,
                        conn1.enabled,
                        conn1.innovation,
                    )
            elif conn2:  # Disjoint or excess from parent2
                # 20% chance to inherit from less fit parent
                if random.random() < 0.2:
                    child.connections[(conn2.in_node, conn2.out_node)] = ConnectionGene(
                        conn2.in_node,
                        conn2.out_node,
                        conn2.weight,
                        conn2.enabled,
                        conn2.innovation,
                    )

        return child

    def clone(self) -> "Genome":
        """Create a deep copy of the genome."""
        new_nodes = {
            k: NodeGene(v.id, v.type, v.activation) for k, v in self.nodes.items()
        }
        new_connections = {
            k: ConnectionGene(v.in_node, v.out_node, v.weight, v.enabled, v.innovation)
            for k, v in self.connections.items()
        }
        new_genome = Genome(new_nodes, new_connections, self.fitness)
        if self.saved_weights is not None:
            new_genome.saved_weights = self.saved_weights.copy()
        return new_genome

    def _prune_nonfunctional(self, genome: "Genome"):
        """Remove nodes and connections not on any path from input to output."""
        input_ids = [n.id for n in genome.nodes.values() if n.type == "input"]
        output_ids = [n.id for n in genome.nodes.values() if n.type == "output"]
        # Build adjacency list for enabled connections
        adj = {nid: [] for nid in genome.nodes}
        for conn in genome.connections.values():
            if conn.enabled:
                adj[conn.in_node].append(conn.out_node)
        # Find all nodes reachable from any input
        reachable = set()
        stack = list(input_ids)
        while stack:
            nid = stack.pop()
            if nid not in reachable:
                reachable.add(nid)
                stack.extend(adj[nid])
        # Find all nodes that can reach any output (reverse graph)
        rev_adj = {nid: [] for nid in genome.nodes}
        for conn in genome.connections.values():
            if conn.enabled:
                rev_adj[conn.out_node].append(conn.in_node)
        can_reach_output = set()
        stack = list(output_ids)
        while stack:
            nid = stack.pop()
            if nid not in can_reach_output:
                can_reach_output.add(nid)
                stack.extend(rev_adj[nid])
        # Keep only nodes and connections that are both reachable from input and can reach output
        valid_nodes = reachable & can_reach_output
        genome.nodes = {nid: n for nid, n in genome.nodes.items() if nid in valid_nodes}
        genome.connections = {
            (c.in_node, c.out_node): c
            for (c_in, c_out), c in genome.connections.items()
            if c.enabled and c.in_node in valid_nodes and c.out_node in valid_nodes
        }


@dataclass
class Species:
    id: int
    members: List[Genome]
    representative: Genome
    stagnation_generations: int = 0
    max_fitness: float = float("-inf")
    adjusted_fitness: float = 0.0


WeightsDict = Dict[Tuple[int, int], Union[float, NDArray]]


@dataclass
class Species:
    """A species in the NEAT population."""

    members: List[Genome]
    representative: Genome
    staleness: int = 0
    best_fitness: float = float("-inf")
    adjusted_fitness_sum: float = 0.0


class Population:
    """Population class for NEAT evolution."""

    def __init__(
        self,
        size: int,
        input_size: int,
        output_size: int,
        compatibility_threshold: float = 3.0,
        elitism: int = 2,
    ):
        """Initialize population with the given size and network dimensions."""
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        self.compatibility_threshold = compatibility_threshold
        self.elitism = elitism
        self.innovation_counter = 0
        # Initialize Genome class counters
        Genome.innovation_counter = 0
        Genome.node_counter = input_size + output_size  # Start after input/output nodes
        self.generation = 0
        self.min_species_size = 2

        # Initialize population with minimal networks
        self.genomes = self._create_diverse_population(size)
        self.species: List[Species] = []
        self.best_genome: Optional[Genome] = None

    def _create_diverse_population(self, size: int) -> List[Genome]:
        """Create initial population with structural diversity.
        Some networks start minimal (just input->output), others have hidden nodes."""
        genomes = []

        # Basic activation options
        activations = ["sigmoid", "tanh", "relu", "leaky_relu"]

        # Create genomes with different structures
        for i in range(size):
            nodes = {}
            connections = {}

            # Add input nodes
            for j in range(self.input_size):
                nodes[j] = NodeGene(j, "input", "linear")

            # Add output nodes (always sigmoid for classification)
            for j in range(self.output_size):
                nodes[self.input_size + j] = NodeGene(
                    self.input_size + j,
                    "output",
                    "sigmoid",  # Always sigmoid for classification tasks
                )

            # Randomly decide if this network starts with hidden nodes
            # 30% chance of no hidden nodes, 70% chance of 1-2 hidden nodes
            if random.random() < 0.7:
                n_hidden = random.randint(1, 2)
                for h in range(n_hidden):
                    node_id = self.input_size + self.output_size + h
                    nodes[node_id] = NodeGene(
                        node_id, "hidden", random.choice(activations)
                    )

            # Get node lists by type
            hidden_nodes = [n for n in nodes.values() if n.type == "hidden"]
            output_nodes = [n for n in nodes.values() if n.type == "output"]
            input_nodes = [n for n in nodes.values() if n.type == "input"]

            # Always connect inputs to outputs
            for in_node in input_nodes:
                for out_node in output_nodes:
                    self.innovation_counter += 1
                    # Initialize weights based on fan-in/fan-out
                    fan_in = len(input_nodes)
                    fan_out = len(output_nodes)
                    if hidden_nodes:
                        fan_in += len(hidden_nodes)
                        fan_out += len(hidden_nodes)

                    limit = np.sqrt(6.0 / (fan_in + fan_out))
                    weight = random.uniform(-limit, limit)

                    connections[(in_node.id, out_node.id)] = ConnectionGene(
                        in_node.id,
                        out_node.id,
                        weight,
                        True,  # enabled
                        self.innovation_counter,
                    )

            # If we have hidden nodes, connect them
            if hidden_nodes:
                # Connect inputs to hidden
                for in_node in input_nodes:
                    for hidden in hidden_nodes:
                        if random.random() < 0.8:  # 80% chance of connection
                            self.innovation_counter += 1
                            weight = random.uniform(-limit, limit)
                            connections[(in_node.id, hidden.id)] = ConnectionGene(
                                in_node.id,
                                hidden.id,
                                weight,
                                True,
                                self.innovation_counter,
                            )

                # Connect hidden to outputs
                for hidden in hidden_nodes:
                    for out_node in output_nodes:
                        if random.random() < 0.8:  # 80% chance of connection
                            self.innovation_counter += 1
                            weight = random.uniform(-limit, limit)
                            connections[(hidden.id, out_node.id)] = ConnectionGene(
                                hidden.id,
                                out_node.id,
                                weight,
                                True,
                                self.innovation_counter,
                            )

            genomes.append(Genome(nodes, connections))

        return genomes

    def _get_structure_hash(self, genome: Genome) -> str:
        """Get a hash representing the network structure (ignoring weights)."""
        # Get nodes by type
        input_nodes = sorted([n.id for n in genome.nodes.values() if n.type == "input"])
        hidden_nodes = sorted(
            [(n.id, n.activation) for n in genome.nodes.values() if n.type == "hidden"]
        )
        output_nodes = sorted(
            [n.id for n in genome.nodes.values() if n.type == "output"]
        )

        # Get enabled connections (sorted for consistency)
        connections = sorted(
            [
                (conn.in_node, conn.out_node)
                for conn in genome.connections.values()
                if conn.enabled
            ]
        )

        # Create structure string
        structure = f"i{input_nodes}h{hidden_nodes}o{output_nodes}c{connections}"
        return structure

    def _add_node_mutation(self, genome: Genome):
        """Add a new node by splitting an existing connection."""
        # Get list of enabled connections
        enabled_connections = [
            conn for conn in genome.connections.values() if conn.enabled
        ]
        if not enabled_connections:
            return

        # Choose a random connection to split
        conn = random.choice(enabled_connections)
        conn.enabled = False

        # Create new node
        new_node_id = self.node_counter = (
            max([node.id for node in genome.nodes.values()]) + 1
        )
        genome.nodes[new_node_id] = NodeGene(new_node_id, "hidden", "relu")

        # Create new connections
        self.innovation_counter += 1
        genome.connections[(conn.in_node, new_node_id)] = ConnectionGene(
            conn.in_node, new_node_id, 1.0, True, self.innovation_counter
        )

        self.innovation_counter += 1
        genome.connections[(new_node_id, conn.out_node)] = ConnectionGene(
            new_node_id, conn.out_node, conn.weight, True, self.innovation_counter
        )

    def _add_connection_mutation(self, genome: Genome):
        """Add a new connection between two unconnected nodes."""
        # Get list of all possible connections
        possible_connections = []
        node_ids = list(genome.nodes.keys())
        for i, node1_id in enumerate(node_ids):
            node1 = genome.nodes[node1_id]
            if node1.type == "output":
                continue  # Output nodes can't be source

            for node2_id in node_ids[i + 1 :]:
                node2 = genome.nodes[node2_id]
                if node2.type == "input":
                    continue  # Input nodes can't be target

                # Check if connection already exists
                if (node1_id, node2_id) not in genome.connections:
                    possible_connections.append((node1_id, node2_id))

        if possible_connections:
            # Add a random new connection
            in_node, out_node = random.choice(possible_connections)
            self.innovation_counter += 1
            genome.connections[(in_node, out_node)] = ConnectionGene(
                in_node, out_node, random.uniform(-1, 1), True, self.innovation_counter
            )

            # After adding a new connection, ensure every output node has at least one incoming connection
            output_nodes = [n for n in genome.nodes.values() if n.type == "output"]
            input_nodes = [n for n in genome.nodes.values() if n.type == "input"]
            hidden_nodes = [n for n in genome.nodes.values() if n.type == "hidden"]
            for output in output_nodes:
                incoming = [
                    conn
                    for conn in genome.connections.values()
                    if conn.out_node == output.id and conn.enabled
                ]
                if not incoming:
                    possible_sources = input_nodes + hidden_nodes
                    if possible_sources:
                        source = random.choice(possible_sources)
                        self.innovation_counter += 1
                        fan_in = len(input_nodes) + len(hidden_nodes)
                        fan_out = len(output_nodes) + len(hidden_nodes)
                        limit = (
                            np.sqrt(6 / (fan_in + fan_out))
                            if (fan_in + fan_out) > 0
                            else 1.0
                        )
                        weight = random.uniform(-limit, limit)
                        genome.connections[(source.id, output.id)] = ConnectionGene(
                            source.id,
                            output.id,
                            weight,
                            True,
                            self.innovation_counter,
                        )

    def evolve(self):
        """Evolve the population to create the next generation."""
        # Speciate the population
        self._speciate()

        # Compute adjusted fitness
        self._compute_adjusted_fitness()

        # Sort species by max fitness
        self.species.sort(key=lambda s: s.best_fitness, reverse=True)

        # Create new population
        new_population = []

        # Elitism: Keep best performing genomes
        if self.elitism > 0:
            best_genomes = sorted(self.genomes, key=lambda x: x.fitness, reverse=True)
            new_population.extend(best_genomes[: self.elitism])

        # Give each species a quota based on their performance
        total_adjusted_fitness = sum(s.adjusted_fitness_sum for s in self.species)
        remaining_slots = self.size - len(new_population)

        for species in self.species:
            if (
                species.staleness <= 15
            ):  # Allow species to survive for up to 15 generations
                # Calculate number of offspring
                if total_adjusted_fitness > 0:
                    quota = max(
                        2,  # Minimum 2 offspring if not stagnant
                        int(
                            remaining_slots
                            * species.adjusted_fitness_sum
                            / total_adjusted_fitness
                        ),
                    )
                else:
                    quota = 2

                # Create offspring
                for _ in range(quota):
                    if len(species.members) >= 2:
                        parent1 = self._select_from_species(species)
                        parent2 = self._select_from_species(species)
                        child = self._crossover(parent1, parent2)
                    else:
                        child = species.members[0].clone()

                    # Apply more mutation to stagnant species
                    mutation_rate = 1.0 + (
                        species.staleness * 0.1
                    )  # Increase mutation with staleness
                    self._mutate(child, rate_multiplier=mutation_rate)
                    new_population.append(child)

        # Fill remaining slots with new random genomes
        while len(new_population) < self.size:
            new_population.append(self._create_diverse_population(1)[0])

        # Update population
        self.genomes = new_population
        self.generation += 1

    def _speciate(self):
        """Divide population into species based on similarity."""
        # Clear current species
        self.species = []

        # Try to add each genome to an existing species
        for genome in self.genomes:
            added = False
            for species in self.species:
                if (
                    get_distance(genome, species.representative)
                    < self.compatibility_threshold
                ):
                    species.members.append(genome)
                    added = True
                    break

            # If no matching species found, create new one
            if not added:
                self.species.append(
                    Species(members=[genome], representative=genome, staleness=0)
                )

    def _compute_adjusted_fitness(self):
        """Compute adjusted fitness using fitness sharing within species."""
        for species in self.species:
            # Update species fitness
            species_fitness = max(m.fitness for m in species.members)
            if species_fitness > species.best_fitness:
                species.best_fitness = species_fitness
                species.staleness = 0
            else:
                species.staleness += 1

            # Compute shared fitness
            for member in species.members:
                member.adjusted_fitness = member.fitness / len(species.members)
            species.adjusted_fitness_sum = sum(
                m.adjusted_fitness for m in species.members
            )

    def _select_from_species(self, species: Species) -> Genome:
        """Tournament selection within a species."""
        tournament_size = min(3, len(species.members))
        selected = random.sample(species.members, tournament_size)
        return max(selected, key=lambda x: x.fitness)

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Perform crossover between two parent genomes."""
        # Inherit structure primarily from the fitter parent
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1

        child_nodes = {}
        child_connections = {}

        # Inherit nodes
        for node_id, node in parent1.nodes.items():
            child_nodes[node_id] = NodeGene(node.id, node.type, node.activation)

        # Inherit connections
        for conn_key, conn1 in parent1.connections.items():
            if conn_key in parent2.connections:
                conn2 = parent2.connections[conn_key]
                # Randomly choose connection properties
                enabled = conn1.enabled if random.random() < 0.5 else conn2.enabled
                weight = conn1.weight if random.random() < 0.5 else conn2.weight
            else:
                enabled = conn1.enabled
                weight = conn1.weight

            child_connections[conn_key] = ConnectionGene(
                conn1.in_node, conn1.out_node, weight, enabled, conn1.innovation
            )

        # Create child genome
        child = Genome(child_nodes, child_connections)

        # Inherit weights from the fitter parent if available
        if parent1.saved_weights is not None:
            child.saved_weights = parent1.saved_weights.copy()

        # Prune any disconnected nodes
        prune_nonfunctional(child)
        return child

    def _mutate(self, genome: Genome, rate_multiplier: float = 1.0):
        """Apply mutation operators to a genome."""
        # Get nodes by type for later use
        input_nodes = [n.id for n in genome.nodes.values() if n.type == "input"]
        hidden_nodes = [n.id for n in genome.nodes.values() if n.type == "hidden"]
        output_nodes = [n.id for n in genome.nodes.values() if n.type == "output"]

        # Weight mutation (95% chance per genome for XOR)
        if random.random() < 0.95 * rate_multiplier:
            for conn in genome.connections.values():
                if (
                    random.random() < 0.2 * rate_multiplier
                ):  # 20% chance per connection (increased for XOR)
                    if (
                        random.random() < 0.2
                    ):  # 20% chance of random reset (increased for XOR)
                        conn.weight = random.uniform(-2, 2)  # Wider range for XOR
                    else:  # 80% chance of perturbation
                        conn.weight += random.gauss(
                            0, 0.2 * rate_multiplier
                        )  # Larger perturbation
                        conn.weight = max(-3, min(3, conn.weight))  # Wider weight range

        # Structural mutations - more aggressive for XOR
        if (
            random.random() < 0.08 * rate_multiplier
        ):  # 8% chance of add node (increased for XOR)
            # Choose a random enabled connection to split
            enabled_conns = [c for c in genome.connections.values() if c.enabled]
            if enabled_conns:
                conn = random.choice(enabled_conns)
                conn.enabled = False

                # Add new node
                new_node_id = max(n.id for n in genome.nodes.values()) + 1
                genome.nodes[new_node_id] = NodeGene(new_node_id, "hidden", "relu")

                # Add new connections
                self.innovation_counter += 1
                genome.connections[(conn.in_node, new_node_id)] = ConnectionGene(
                    conn.in_node, new_node_id, 1.0, True, self.innovation_counter
                )

                self.innovation_counter += 1
                genome.connections[(new_node_id, conn.out_node)] = ConnectionGene(
                    new_node_id,
                    conn.out_node,
                    conn.weight,
                    True,
                    self.innovation_counter,
                )

                prune_nonfunctional(genome)

        if (
            random.random() < 0.12 * rate_multiplier
        ):  # 12% chance of add connection (increased for XOR)
            # Get all possible new connections
            possible_connections = []
            # Get nodes by type (to properly connect hidden->output)
            input_nodes = [n.id for n in genome.nodes.values() if n.type == "input"]
            hidden_nodes = [n.id for n in genome.nodes.values() if n.type == "hidden"]
            output_nodes = [n.id for n in genome.nodes.values() if n.type == "output"]

            # Find all valid sources and targets
            sources = input_nodes + hidden_nodes
            targets = hidden_nodes + output_nodes

            # Prioritize hidden->output connections to ensure hidden nodes connect to outputs
            if (
                hidden_nodes and output_nodes and random.random() < 0.3
            ):  # 30% chance to focus on hidden->output
                for h_id in hidden_nodes:
                    for o_id in output_nodes:
                        if (h_id, o_id) not in genome.connections:
                            possible_connections.append((h_id, o_id))

            # If no hidden->output connections possible, try all valid connections
            if not possible_connections:
                for source_id in sources:
                    for target_id in targets:
                        if (
                            source_id != target_id
                            and (source_id, target_id) not in genome.connections
                        ):
                            # Skip input->input, output->output, output->anything
                            if (
                                source_id in input_nodes and target_id in input_nodes
                            ) or (source_id in output_nodes):
                                continue
                            possible_connections.append((source_id, target_id))

            # Add a random new connection
            if possible_connections:
                in_node, out_node = random.choice(possible_connections)
                self.innovation_counter += 1
                genome.connections[(in_node, out_node)] = ConnectionGene(
                    in_node,
                    out_node,
                    random.uniform(-1, 1),
                    True,
                    self.innovation_counter,
                )

                prune_nonfunctional(genome)

        # Additional mutation: explicitly try adding hidden->output connections
        # This ensures hidden nodes can influence outputs
        if hidden_nodes and random.random() < 0.2:  # 20% chance
            for hidden_id in hidden_nodes:
                for output_id in output_nodes:
                    # If connection doesn't exist yet, add it with some probability
                    if (
                        hidden_id,
                        output_id,
                    ) not in genome.connections and random.random() < 0.3:
                        self.innovation_counter += 1
                        genome.connections[(hidden_id, output_id)] = ConnectionGene(
                            hidden_id,
                            output_id,
                            random.uniform(-1, 1),
                            True,
                            self.innovation_counter,
                        )

    def _select_parent(self, tournament_size: int = 3):
        """Select a parent genome using tournament selection."""
        # Randomly select tournament_size genomes
        candidates = random.sample(
            self.genomes, min(tournament_size, len(self.genomes))
        )
        # Return the genome with the highest fitness
        return max(candidates, key=lambda g: g.fitness)


def prune_nonfunctional(genome: Genome):
    """Remove nodes and connections not on any path from input to output.
    This is critical for XOR networks to ensure all nodes contribute to the solution."""
    input_ids = [n.id for n in genome.nodes.values() if n.type == "input"]
    output_ids = [n.id for n in genome.nodes.values() if n.type == "output"]
    hidden_ids = [n.id for n in genome.nodes.values() if n.type == "hidden"]

    # Build adjacency list for enabled connections
    adj = {nid: [] for nid in genome.nodes}
    for conn in genome.connections.values():
        if conn.enabled:
            adj[conn.in_node].append(conn.out_node)

    # Find all nodes reachable from any input
    reachable = set()
    stack = list(input_ids)
    while stack:
        nid = stack.pop()
        if nid not in reachable:
            reachable.add(nid)
            stack.extend(adj[nid])

    # Find all nodes that can reach any output (reverse graph)
    rev_adj = {nid: [] for nid in genome.nodes}
    for conn in genome.connections.values():
        if conn.enabled:
            rev_adj[conn.out_node].append(conn.in_node)
    can_reach_output = set()
    stack = list(output_ids)
    while stack:
        nid = stack.pop()
        if nid not in can_reach_output:
            can_reach_output.add(nid)
            stack.extend(rev_adj[nid])

    # Keep only nodes and connections that are both reachable from input and can reach output
    valid_nodes = reachable & can_reach_output

    # Update genome with pruned nodes and connections
    genome.nodes = {nid: n for nid, n in genome.nodes.items() if nid in valid_nodes}
    genome.connections = {
        (c.in_node, c.out_node): c
        for (c_in, c_out), c in genome.connections.items()
        if c.enabled and c.in_node in valid_nodes and c.out_node in valid_nodes
    }

    return genome
