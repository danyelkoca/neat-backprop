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

from .neat import Genome, prune_nonfunctional

# Define a type variable for numeric types
Number = TypeVar("Number", float, int)
WeightsDict = Dict[Tuple[int, int], Union[jnp.ndarray, NDArray, Number]]


class Network:
    def __init__(
        self, genome: Genome, learning_rate: float = 0.05, debug: bool = False
    ):
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
                    # Direct input->output connections: need larger weights
                    scale = 2.0 / (n_input + n_output)
                    init_weight = np.random.uniform(-2.0, 2.0) * scale
                elif from_type == "hidden" or to_type == "hidden":
                    # Connections to/from hidden: standard Xavier
                    scale = np.sqrt(6.0 / (n_input + n_output + n_hidden))
                    init_weight = np.random.uniform(-1.0, 1.0) * scale
                else:
                    # Default
                    init_weight = conn.weight

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
        """Apply activation function with extreme non-linearity for XOR."""
        if activation == "sigmoid":
            # Extreme sigmoid for XOR to push values far from 0.5
            temp = 3.0  # Much higher temperature for sharper decision boundaries
            return jax.nn.sigmoid(x * temp)
        elif activation == "relu":
            # For XOR, plain ReLU often fails due to collapsing to a linear function
            # Add a curvature component for non-linearity
            sigmoid_component = jax.nn.sigmoid(x * 2.0) - 0.5  # -0.5 to 0.5 range
            return (
                jax.nn.relu(x) + sigmoid_component * 0.5
            )  # Modified ReLU with sigmoid bent
        elif activation == "tanh":
            # Very steep tanh for strong XOR discrimination
            return jnp.tanh(x * 3.0)  # Much steeper slope for better decision boundary
        elif activation == "leaky_relu":
            # Non-linear leaky ReLU with sigmoid bent
            base = jax.nn.leaky_relu(x, negative_slope=0.25)
            sigmoid_bend = jax.nn.sigmoid(x * 2.5) - 0.5
            return base + sigmoid_bend * 0.4
        elif activation == "elu":
            # Stronger ELU with extra non-linearity
            base = jax.nn.elu(x * 2.0)
            return base + 0.2 * jnp.tanh(x * 3.0)
        elif activation == "softplus":
            # Modified softplus with sharp transition
            return jax.nn.softplus(x * 2.0) + 0.1 * jnp.tanh(x * 5.0)
        elif activation == "swish":
            # Enhanced Swish with extra non-linearity - great for XOR
            beta = 3.0  # Much steeper
            return x * jax.nn.sigmoid(x * beta) + 0.1 * jnp.tanh(x * 4.0)
        elif activation == "gelu":
            # Enhanced GELU with extra non-linearity
            return jax.nn.gelu(x * 2.0) + 0.15 * jnp.tanh(x * 3.0)
        # For "linear" or any other activation, add strong non-linearity
        # This is crucial for XOR problem
        return x + 0.25 * jnp.tanh(x * 3.0) + 0.1 * jax.nn.sigmoid(x * 4.0) - 0.05

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

        # Create flag for debugging - only debug specific XOR corner cases
        # Use numpy for debugging checks to avoid JAX tracing errors
        debug_mode = False

        # Only perform this check outside of JAX tracing
        # Check if this is a JAX tracer object without directly accessing jax.core
        is_jax_tracer = hasattr(x, "_trace") and hasattr(x, "aval")
        if not is_jax_tracer and len(x) == 2:
            # We need to convert this to a regular Python value to avoid JAX tracing issues
            x_np = np.array(x)
            # If input is close to one of the XOR corners
            corners = [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
            for corner in corners:
                if abs(x_np[0] - corner[0]) < 0.3 and abs(x_np[1] - corner[1]) < 0.3:
                    debug_mode = True
                    break

        # Initialize node values
        node_values = {
            node_id: jnp.array([0.0]) for node_id in self.genome.nodes.keys()
        }

        # Set input values
        input_nodes = [n.id for n in self.genome.nodes.values() if n.type == "input"]
        for i, node_id in enumerate(input_nodes):
            node_values[node_id] = jnp.array([x[i]])

        if debug_mode:
            print(f"\n--- DETAILED NETWORK TRACE for input {x} ---")
            print(f"Input nodes: {input_nodes}")
            print(f"Input values: {[node_values[n][0] for n in input_nodes]}")

        # Process nodes in topological order
        sorted_nodes = self._get_sorted_nodes()

        if debug_mode:
            print(f"Node processing order: {sorted_nodes}")

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
        """Compute MSE loss."""
        predictions = jax.vmap(lambda x: self.forward(x))(x)
        return float(jnp.mean((predictions - y) ** 2))

    def _update_learning_rate(self):
        """Update learning rate using a simple decay schedule."""
        # Decay learning rate every 100 epochs
        if self.epoch % 100 == 0 and self.epoch > 0:
            self.learning_rate = self.initial_learning_rate * (
                0.5 ** (self.epoch // 100)
            )
        self.epoch += 1

    def train_step(self, x: jnp.ndarray, y: jnp.ndarray, use_bce: bool = True) -> float:
        """Perform one training step using backpropagation."""
        self._sync_weights_with_genome()
        self._update_learning_rate()

        # Define loss function for batch
        def batch_loss(
            weights: WeightsDict, x_batch: jnp.ndarray, y_batch: jnp.ndarray
        ) -> jnp.ndarray:
            predictions = jax.vmap(lambda x: self.forward_impl(weights, x))(x_batch)

            if use_bce:
                # Binary cross entropy for classification tasks
                preds_clipped = jnp.clip(predictions, 1e-7, 1.0 - 1e-7)
                return -jnp.mean(
                    y_batch * jnp.log(preds_clipped)
                    + (1 - y_batch) * jnp.log(1 - preds_clipped)
                )
            else:
                # MSE loss for regression tasks
                return jnp.mean((predictions - y_batch) ** 2)

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

        # If the input is small (for debugging), print inputs and outputs
        if isinstance(x, np.ndarray) and len(x) <= 10:
            print("\nDebug predict_binary:")
            for i in range(len(x)):
                print(
                    f"Input: {x[i]}, Raw: {predictions[i][0]:.4f}, Binary: {binary_output[i][0]}"
                )

        return binary_output

    def compute_binary_accuracy(
        self, x: jnp.ndarray, y: jnp.ndarray, threshold: float = 0.5
    ) -> float:
        """Compute binary classification accuracy."""
        binary_preds = self.predict_binary(x, threshold)
        accuracy = float(jnp.mean((binary_preds == y).astype(jnp.float32)))

        # For debugging XOR accuracy issues
        if len(x) <= 20:  # Only for small test sets
            raw_preds = self.forward(x)
            print("Debug - Sample predictions:")
            for i in range(min(5, len(x))):
                print(
                    f"  Input: {x[i]}, Expected: {y[i][0]:.1f}, Raw pred: {raw_preds[i][0]:.4f}, Binary: {binary_preds[i][0]:.1f}"
                )

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
