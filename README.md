# NEAT-Backprop: Neuroevolution with Backpropagation

![xor_evolution.gif](xor_evolution.gif)

This project implements a hybrid neuroevolution algorithm that combines NEAT (NeuroEvolution of Augmenting Topologies) with backpropagation for evolving neural networks on classic classification tasks. The system evolves both the topology and weights of neural networks, and further optimizes weights using gradient descent.

## Example Results

### XOR

![xor_gen_32.png](results/xor_gen_32.png)

### Two Circles

![two_circles_gen_12.png](results/two_circles_gen_12.png)

### Two Gaussians

![two_gaussians_gen_0.png](results/two_gaussians_gen_0.png)

### Spiral

![spiral_gen_31.png](results/spiral_gen_31.png)

## Features

- Evolve neural network architectures and weights using NEAT
- Backpropagation for fine-tuning weights within each generation
- Supports multiple 2D classification tasks: XOR, two circles, two gaussians, spiral
- Visualizes both the network architecture and decision boundary during evolution

## How to Run

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run training:**

   ```bash
   python train.py
   ```

   You can edit `train.py` to change the task (e.g., 'xor', 'two_circles', 'two_gaussians', 'spiral') and other parameters.

3. **Results:**
   - Combined plots of network architecture and decision boundary are saved in the `results/` directory for each task and generation.
