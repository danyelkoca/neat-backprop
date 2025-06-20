# NEAT with Backpropagation

A hybrid neural network system that combines NEAT (NeuroEvolution of Augmenting Topologies) with backpropagation using JAX. The algorithm evolves network architectures while training weights through gradient descent.

## Features

- JAX-based neural networks with automatic differentiation
- NEAT for topology evolution
- Built-in 2D classification tasks:
  - XOR
  - Two Circles
  - Spiral
- Real-time visualization of:
  - Network architecture
  - Decision boundaries
  - Training progress

## Quick Start

1. Set up environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run with default settings (XOR task):

```bash
python main.py
```

3. Try different datasets:

```python
# In main.py, change the dataset:
from neat_backprop.datasets import generate_xor, generate_two_circles, generate_spiral

main(dataset_fn=generate_spiral)  # Try spiral dataset
main(dataset_fn=generate_two_circles)  # Try two circles dataset
```

## Structure

```
neat_backprop/
├── neat.py         # NEAT implementation
├── network.py      # Neural network with JAX
├── datasets.py     # Classification tasks
└── visualization.py # Network/boundary plotting

main.py            # Example usage
requirements.txt   # Dependencies
```
