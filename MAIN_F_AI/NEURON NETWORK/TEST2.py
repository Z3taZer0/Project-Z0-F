from C_Layer import Layer
from C_Network import NeuralNetwork, train
from SaveSystem import save_network, load_network
import numpy as np
from Visual import visualize_network

X = np.array([[0.6, -1.5, 2.5, -1.4],
              [1.0, 0.5, -0.5, 2.0]])

y = np.array([1, 0])

nn = NeuralNetwork([
    Layer(n_inputs=4, n_neurons=6),
    Layer(n_inputs=6, n_neurons=7),
    Layer(n_inputs=7, n_neurons=5),
    Layer(n_inputs=5, n_neurons=4),
    Layer(n_inputs=4, n_neurons=1)
])

load_network(nn, "network_data2.npz")

print("Sortie du réseau de neurones avant entraînement :", nn.forward(X[0]))
    
train(nn, X, y, lr=0.5, epochs=10000)

save_network(nn, "network_data2.npz")

print("Sortie du réseau de neurones après entraînement :", nn.forward(X[0]))

visualize_network(nn)