"""
Visualization utilities for networks and results.
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import numpy as np
from neat_backprop.neat import Genome
from typing import Optional


def plot_network(genome: Genome, title: Optional[str] = None):
    """Visualize network architecture using networkx."""
    G = nx.DiGraph()

    # Add nodes
    pos = {}
    input_nodes = []
    hidden_nodes = []
    output_nodes = []

    # Group nodes by activation function for shape assignment
    relu_nodes = []
    sigmoid_nodes = []
    linear_nodes = []
    tanh_nodes = []
    other_nodes = []

    for node_id, node in genome.nodes.items():
        G.add_node(node_id)
        # Categorize by node type
        if node.type == "input":
            input_nodes.append(node_id)
        elif node.type == "hidden":
            hidden_nodes.append(node_id)
        else:
            output_nodes.append(node_id)

        # Categorize by activation function
        if node.activation == "relu":
            relu_nodes.append(node_id)
        elif node.activation == "sigmoid":
            sigmoid_nodes.append(node_id)
        elif node.activation == "linear":
            linear_nodes.append(node_id)
        elif node.activation == "tanh":
            tanh_nodes.append(node_id)
        else:
            other_nodes.append(node_id)

    # Position nodes in layers
    layer_spacing = 2.0
    node_spacing = 1.0

    # Position input nodes
    for i, node_id in enumerate(input_nodes):
        pos[node_id] = (0, (i - len(input_nodes) / 2) * node_spacing)

    # Position hidden nodes
    for i, node_id in enumerate(hidden_nodes):
        pos[node_id] = (layer_spacing, (i - len(hidden_nodes) / 2) * node_spacing)

    # Position output nodes
    for i, node_id in enumerate(output_nodes):
        pos[node_id] = (2 * layer_spacing, (i - len(output_nodes) / 2) * node_spacing)

    # Add connections
    for conn in genome.connections.values():
        if conn.enabled:
            G.add_edge(conn.in_node, conn.out_node, weight=conn.weight)

    # Draw the network
    plt.figure(figsize=(10, 8))

    # Define colors for each activation and node type
    activation_colors = {
        "relu": "#9C27B0",  # Vibrant purple
        "sigmoid": "#2ECC71",  # Bright green
        "tanh": "#E74C3C",  # Bright red
        "linear": "#3498DB",  # Sky blue
        "leaky_relu": "#F39C12",  # Golden orange
    }

    type_colors = {"input": "#1B1464", "output": "#6F1E51"}  # Deep navy  # Deep magenta

    # Draw input nodes
    if input_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=input_nodes,
            node_color=type_colors["input"],
            node_shape="o",
            node_size=500,
        )

    # Draw output nodes
    if output_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=output_nodes,
            node_color=type_colors["output"],
            node_shape="o",
            node_size=500,
        )

    # Draw nodes by activation function (for hidden nodes)
    for activation, color in activation_colors.items():
        nodes = [
            n.id
            for n in genome.nodes.values()
            if n.type == "hidden" and n.activation == activation
        ]
        if nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes,
                node_color=color,
                node_shape="o",
                node_size=500,
            )

    # Draw all connections in gray
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#808080",  # Medium gray
        arrows=True,
        width=1.0,
        alpha=0.6,
        arrowsize=10,
    )

    # Add node labels with white text
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_color="white")

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=type_colors["input"],
            label="Input",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=type_colors["output"],
            label="Output",
            markersize=10,
        ),
    ]

    # Add activation functions to legend
    for activation, color in activation_colors.items():
        if any(
            n.activation == activation and n.type == "hidden"
            for n in genome.nodes.values()
        ):
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    label=activation.capitalize(),
                    markersize=10,
                )
            )

    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    # Add connection legend
    legend_elements.append(Line2D([0], [0], color="#808080", label="Connection"))

    plt.axis("off")

    # Save or display
    if title:
        plt.savefig(f"{title}.png", bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_decision_boundary(model, X, y, title: Optional[str] = None):
    """Plot the decision boundary of the network with both raw and binary outputs."""
    h = 0.02  # Step size in the mesh

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Create points for prediction
    grid_points = np.array([[x, y] for x, y in zip(xx.ravel(), yy.ravel())])

    # Make predictions on mesh
    Z_raw = model.forward(grid_points)
    Z_raw = Z_raw.reshape(xx.shape)

    # Get binary predictions
    Z_binary = model.predict_binary(grid_points, threshold=0.5)
    Z_binary = Z_binary.reshape(xx.shape)

    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Calculate accuracy
    accuracy = model.compute_binary_accuracy(X, y)

    # Plot raw outputs
    cs_raw = axes[0].contourf(xx, yy, Z_raw, cmap="coolwarm", alpha=0.8, levels=20)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="black", s=40)
    axes[0].set_title(f"Raw Network Output (Accuracy: {accuracy:.2f})", fontsize=14)
    axes[0].set_xlabel("X1", fontsize=12)
    axes[0].set_ylabel("X2", fontsize=12)
    fig.colorbar(cs_raw, ax=axes[0], label="Raw Output")

    # Add decision boundary line
    axes[0].contour(
        xx, yy, Z_raw, levels=[0.5], colors="k", linewidths=2, linestyles="--"
    )

    # Plot binary classification
    cs_bin = axes[1].contourf(xx, yy, Z_binary, cmap="coolwarm", alpha=0.8, levels=2)
    axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="black", s=40)
    axes[1].set_title(f"Binary Predictions (Accuracy: {accuracy:.2f})", fontsize=14)
    axes[1].set_xlabel("X1", fontsize=12)
    axes[1].set_ylabel("X2", fontsize=12)
    fig.colorbar(cs_bin, ax=axes[1], label="Binary Output (0/1)")

    # Save or display based on title
    if title:
        plt.savefig(f"{title}.png", bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
