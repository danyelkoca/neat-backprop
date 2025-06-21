"""
Visualization utilities for networks and results.
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import numpy as np
from src.neat import Genome
from typing import Optional


def plot_network_and_decision_boundary(genome, network, X, y, task, gen):
    """Plot network architecture and raw decision boundary side by side."""
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    from matplotlib.lines import Line2D

    # --- Plot network architecture (left subplot) ---
    G = nx.DiGraph()
    pos = {}
    input_nodes = []
    hidden_nodes = []
    output_nodes = []
    bias_nodes = []
    relu_nodes = []
    sigmoid_nodes = []
    linear_nodes = []
    tanh_nodes = []
    other_nodes = []
    connected_nodes = set()
    for conn in genome.connections.values():
        if conn.enabled:
            connected_nodes.add(conn.in_node)
            connected_nodes.add(conn.out_node)
    for node_id, node in genome.nodes.items():
        if node_id not in connected_nodes:
            continue
        G.add_node(node_id)
        if node.type == "input":
            input_nodes.append(node_id)
        elif node.type == "hidden":
            hidden_nodes.append(node_id)
        elif node.type == "bias":
            bias_nodes.append(node_id)
        else:
            output_nodes.append(node_id)
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
    layer_spacing = 2.0
    node_spacing = 1.0
    for i, node_id in enumerate(input_nodes):
        pos[node_id] = (0, (i - len(input_nodes) / 2) * node_spacing)
    for i, node_id in enumerate(bias_nodes):
        pos[node_id] = (0, -((len(input_nodes) / 2) + 1 + i) * node_spacing)
    for i, node_id in enumerate(hidden_nodes):
        pos[node_id] = (layer_spacing, (i - len(hidden_nodes) / 2) * node_spacing)
    for i, node_id in enumerate(output_nodes):
        pos[node_id] = (2 * layer_spacing, (i - len(output_nodes) / 2) * node_spacing)
    for conn in genome.connections.values():
        if conn.enabled:
            G.add_edge(conn.in_node, conn.out_node, weight=conn.weight)
    activation_colors = {
        "relu": "#9C27B0",
        "sigmoid": "#2ECC71",
        "tanh": "#E74C3C",
        "linear": "#3498DB",
        "leaky_relu": "#F39C12",
    }
    type_colors = {
        "input": "#1B1464",
        "output": "#6F1E51",
        "bias": "#E67E22",
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    # Draw input nodes
    if input_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=input_nodes,
            node_color=type_colors["input"],
            node_shape="o",
            node_size=500,
            ax=axes[0],
        )
    if bias_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=bias_nodes,
            node_color=type_colors["bias"],
            node_shape="s",
            node_size=500,
            ax=axes[0],
        )
    if output_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=output_nodes,
            node_color=type_colors["output"],
            node_shape="o",
            node_size=500,
            ax=axes[0],
        )
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
                ax=axes[0],
            )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#808080",
        arrows=True,
        width=1.0,
        alpha=0.6,
        arrowsize=10,
        ax=axes[0],
    )
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_color="white", ax=axes[0])
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
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=type_colors["bias"],
            label="Bias",
            markersize=10,
        ),
    ]
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
    legend_elements.append(Line2D([0], [0], color="#808080", label="Connection"))
    axes[0].legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))
    axes[0].set_title("Network Architecture")
    axes[0].axis("off")

    # --- Plot raw decision boundary (right subplot) ---
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.array([[x, y] for x, y in zip(xx.ravel(), yy.ravel())])
    Z_raw = network.forward(grid_points)
    Z_raw = Z_raw.reshape(xx.shape)
    accuracy = network.compute_binary_accuracy(X, y)
    cs_raw = axes[1].contourf(xx, yy, Z_raw, cmap="coolwarm", alpha=0.8, levels=20)
    axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="black", s=40)
    axes[1].set_title(f"Raw Network Output (Accuracy: {accuracy:.4f})", fontsize=14)
    axes[1].set_xlabel("X1", fontsize=12)
    axes[1].set_ylabel("X2", fontsize=12)
    fig.colorbar(cs_raw, ax=axes[1], label="Raw Output")
    axes[1].contour(
        xx, yy, Z_raw, levels=[0.5], colors="k", linewidths=2, linestyles="--"
    )
    plt.tight_layout()
    plt.savefig(f"results/{task}_gen_{gen}.png", bbox_inches="tight", dpi=150)
    plt.close()
