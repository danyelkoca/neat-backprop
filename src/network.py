"""
JAX-based neural network implementation for backpropagation.
"""

from typing import Dict, List, Tuple, Union, TypeVar
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
import random
from numpy.typing import NDArray

from .neat import Genome

# Define a type variable for numeric types
Number = TypeVar("Number", float, int)
WeightsDict = Dict[Tuple[int, int], Union[jnp.ndarray, NDArray, Number]]


class Network:
    def __init__(self, genome: Genome, learning_rate: float = 0.1):
        self.genome = genome
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.epoch = 0  # Initialize weights dictionary
        self.weights: WeightsDict = {}

        import numpy as np

        # Try to use saved weights first
        if genome.saved_weights is not None:
            for key, value in genome.saved_weights.items():
                # Convert any type of value to jax array
                if isinstance(value, (float, int)):
                    self.weights[key] = jnp.array([float(value)], dtype=jnp.float32)
                else:
                    self.weights[key] = jnp.array(value, dtype=jnp.float32)
            return

        # If no saved weights, use Xavier initialization
        import numpy as np

        n_input = sum(1 for n in genome.nodes.values() if n.type == "input")
        n_hidden = sum(1 for n in genome.nodes.values() if n.type == "hidden")
        n_output = sum(1 for n in genome.nodes.values() if n.type == "output")
        for conn in genome.connections.values():
            if conn.enabled:
                # Determine fan-in and fan-out for proper scaling
                from_type = genome.nodes[conn.in_node].type
                to_type = genome.nodes[conn.out_node].type

                if from_type == "input" and to_type == "output":
                    # Direct input->output: use He initialization with scaling
                    scale = np.sqrt(2.0 / n_input)
                    init_weight = np.random.normal(0, scale)
                elif from_type == "hidden" or to_type == "hidden":
                    # Hidden layer connections: use combination of He and Xavier
                    fan_in = n_input if from_type == "input" else n_hidden
                    fan_out = n_output if to_type == "output" else n_hidden
                    scale = np.sqrt(4.0 / (fan_in + fan_out))
                    init_weight = np.random.normal(0, scale)
                else:
                    # Default with small random initialization
                    init_weight = np.random.normal(0, 0.1)

                # Ensure non-zero initialization to break symmetry
                if abs(init_weight) < 0.01:
                    init_weight = 0.01 if init_weight >= 0 else -0.01

                self.weights[(conn.in_node, conn.out_node)] = jnp.array([init_weight])

    def _get_sorted_nodes(self) -> List[int]:
        """Get nodes in topological order."""
        inputs = [n.id for n in self.genome.nodes.values() if n.type == "input"]
        outputs = [n.id for n in self.genome.nodes.values() if n.type == "output"]
        hidden = [n.id for n in self.genome.nodes.values() if n.type == "hidden"]
        return inputs + hidden + outputs

    def _activate(self, x: jnp.ndarray, activation: str) -> jnp.ndarray:
        """Apply activation function with balanced non-linearity for general classification."""
        if activation == "sigmoid":
            # Balanced sigmoid with moderate temperature for good gradients
            return jax.nn.sigmoid(x * 1.5)
        elif activation == "relu":
            # ReLU with slight bend to avoid dead neurons and improve gradients
            small_bend = jax.nn.sigmoid(x) - 0.5
            return jax.nn.relu(x) + small_bend * 0.2
        elif activation == "tanh":
            # Standard tanh with slight scaling for better gradient flow
            return jnp.tanh(x * 1.5)
        elif activation == "leaky_relu":
            # Leaky ReLU with balanced slope for better gradient flow
            return jax.nn.leaky_relu(x, negative_slope=0.1)
        elif activation == "elu":
            # Standard ELU with slight scaling
            return jax.nn.elu(x * 1.2)
        elif activation == "swish":
            # Swish with balanced beta, good for deep networks
            beta = 1.5
            return x * jax.nn.sigmoid(x * beta)
        elif activation == "gelu":
            # Standard GELU, good for various tasks
            return jax.nn.gelu(x)
        # Default to a slightly enhanced tanh for unknown activations
        return jnp.tanh(x * 1.2)

    def _sync_weights_with_genome(self):
        """Ensure all enabled connections in the genome have weights as float32."""
        for conn in self.genome.connections.values():
            key = (conn.in_node, conn.out_node)
            if conn.enabled and key not in self.weights:
                # Always cast to float32
                self.weights[key] = jnp.array([float(conn.weight)], dtype=jnp.float32)
            elif conn.enabled:
                # Ensure existing weights are float32
                self.weights[key] = jnp.array(self.weights[key], dtype=jnp.float32)

    def forward_impl(self, weights: WeightsDict, x: jnp.ndarray) -> jnp.ndarray:
        self._sync_weights_with_genome()

        # Initialize node values
        node_values = {
            node_id: (
                jnp.array([1.0])
                if self.genome.nodes[node_id].type == "bias"
                else jnp.array([0.0])
            )
            for node_id in self.genome.nodes.keys()
        }

        # Set input values
        input_nodes = [n.id for n in self.genome.nodes.values() if n.type == "input"]
        for i, node_id in enumerate(input_nodes):
            node_values[node_id] = jnp.array([x[i]])

        # Process nodes in topological order
        sorted_nodes = self._get_sorted_nodes()

        for node_id in sorted_nodes:
            if node_id not in input_nodes:  # Skip input nodes
                node = self.genome.nodes[node_id]
                # Sum incoming connections
                incoming_sum = jnp.array([0.0])

                for conn in self.genome.connections.values():
                    if conn.enabled and conn.out_node == node_id:
                        weight = weights[(conn.in_node, conn.out_node)]
                        contribution = node_values[conn.in_node] * weight
                        incoming_sum += contribution

                # Apply activation function
                pre_activation = incoming_sum
                post_activation = self._activate(incoming_sum, node.activation)
                node_values[node_id] = post_activation

        # Get output values
        output_nodes = [n.id for n in self.genome.nodes.values() if n.type == "output"]
        outputs = jnp.array([node_values[node_id][0] for node_id in output_nodes])

        return outputs

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with current weights. Handles both single and batch inputs."""
        # If x is 1D, treat as single sample
        if x.ndim == 1:
            return self.forward_impl(self.weights, x)
        # If x is 2D, treat as batch
        return jax.vmap(lambda xi: self.forward_impl(self.weights, xi))(x)

    def compute_loss(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute binary cross entropy loss."""
        predictions = self.forward(x)
        predictions = jnp.clip(predictions, 1e-7, 1.0 - 1e-7)
        return float(
            -jnp.mean(y * jnp.log(predictions) + (1 - y) * jnp.log(1 - predictions))
        )

    def _update_learning_rate(self):
        """Update learning rate using a cosine decay schedule with warm restarts."""
        # Cosine annealing with warm restarts
        cycle_length = 50  # Shorter cycles for more frequent restarts
        min_lr = self.initial_learning_rate * 0.01
        cycle = self.epoch // cycle_length
        t = (self.epoch % cycle_length) / cycle_length
        # Cosine decay with warm restarts
        cos_decay = 0.5 * (1 + jnp.cos(jnp.pi * t))
        # Add gradual decay across cycles
        cycle_decay = 0.9**cycle
        self.learning_rate = (
            min_lr + (self.initial_learning_rate - min_lr) * cos_decay * cycle_decay
        )
        self.epoch += 1

    def train_step(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """Perform one training step using backpropagation with binary cross entropy loss."""
        self._sync_weights_with_genome()
        self._update_learning_rate()

        # Define loss function for batch
        def batch_loss(
            weights: WeightsDict, x_batch: jnp.ndarray, y_batch: jnp.ndarray
        ) -> jnp.ndarray:
            predictions = jax.vmap(lambda x: self.forward_impl(weights, x))(x_batch)
            # Binary cross entropy loss
            preds_clipped = jnp.clip(predictions, 1e-7, 1.0 - 1e-7)
            return -jnp.mean(
                y_batch * jnp.log(preds_clipped)
                + (1 - y_batch) * jnp.log(1 - preds_clipped)
            )

        # Compute gradients
        loss_val, gradients = jax.value_and_grad(batch_loss)(self.weights, x, y)

        # Update weights with gradient descent
        for key in self.weights:
            grad = gradients[key]

            # Basic gradient clipping to prevent explosions
            grad = jnp.clip(grad, -1.0, 1.0)

            # Apply learning rate
            self.weights[key] -= self.learning_rate * grad

            # Basic weight clipping to maintain stability
            self.weights[key] = jnp.clip(self.weights[key], -5.0, 5.0)

        return float(loss_val)

    def get_weights(self) -> WeightsDict:
        """Get a copy of the network weights."""
        return {k: v.copy() for k, v in self.weights.items()}

    def set_weights(self, weights):
        """Set network weights.
        Can handle both JAX arrays and Python floats/dicts."""
        self.weights = {}
        for k, v in weights.items():
            # If v is a JAX array, copy it directly
            if isinstance(v, jnp.ndarray):
                self.weights[k] = v.copy()
            # If v is a single float (from serialized weights), wrap it in a JAX array
            elif isinstance(v, (float, int)):
                self.weights[k] = jnp.array([float(v)], dtype=jnp.float32)
            # If v is a numpy array
            elif isinstance(v, np.ndarray):
                self.weights[k] = jnp.array(v, dtype=jnp.float32)
            # For any other case, try to convert and wrap
            else:
                try:
                    self.weights[k] = jnp.array([float(v)], dtype=jnp.float32)
                except (TypeError, ValueError):
                    raise TypeError(
                        f"Cannot convert weight of type {type(v)} to JAX array"
                    )

    def predict_binary(self, x: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
        """Make binary predictions with threshold."""
        predictions = self.forward(x)
        # For XOR, use a cleaner threshold with some margin
        binary_output = (predictions >= threshold).astype(jnp.float32)

        return binary_output

    def compute_binary_accuracy(
        self, x: jnp.ndarray, y: jnp.ndarray, threshold: float = 0.5
    ) -> float:
        """Compute binary classification accuracy."""
        binary_preds = self.predict_binary(x, threshold)
        accuracy = float(jnp.mean((binary_preds == y).astype(jnp.float32)))

        return accuracy

    def compute_binary_cross_entropy(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute binary cross entropy loss."""
        predictions = self.forward(x)
        # Clip predictions to prevent log(0)
        predictions = jnp.clip(predictions, 1e-7, 1.0 - 1e-7)

        # Add focal loss weighting for XOR
        # Give more weight to hard examples (predictions close to 0.5)
        # This helps to focus on the difficult examples that the model is uncertain about
        confidence = jnp.abs(predictions - 0.5) * 2  # 0 for p=0.5, 1 for p=0 or p=1
        focal_weight = (1 - confidence) ** 2  # Higher weight for uncertain predictions

        # Apply focal weighting
        weighted_bce = -jnp.mean(
            focal_weight
            * (y * jnp.log(predictions) + (1 - y) * jnp.log(1 - predictions))
        )
        return float(weighted_bce)
