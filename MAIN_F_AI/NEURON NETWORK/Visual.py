# Fait par chatGPT

import matplotlib.pyplot as plt

def visualize_network(nn):
    layer_sizes = [nn.layers[0].weights.shape[1]]  # nombre d'entrées
    layer_sizes += [layer.weights.shape[0] for layer in nn.layers]

    _, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off') # Disable axis lines and labels for a cleaner neural network visualization

    # Placement des couches
    x_spacing = 1 / (len(layer_sizes) + 1)

    positions = []
    for i, size in enumerate(layer_sizes):
        x = (i+1) * x_spacing
        y_positions = []
        for j in range(size):
            y = 1 - (j+1)/(size+1)
            y_positions.append((x, y))
            ax.scatter(x, y, s=500)
        positions.append(y_positions)

    # Connexions
    for i in range(len(layer_sizes)-1):
        for x1, y1 in positions[i]:
            for x2, y2 in positions[i+1]:
                ax.plot([x1, x2], [y1, y2], 'gray')

    plt.show()