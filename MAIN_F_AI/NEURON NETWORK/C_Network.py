import numpy as np


class NeuralNetwork: # Definition de la classe NeuralNetwork, représentant un réseau de neurones composé de plusieurs couches
    def __init__(self, layers): # Initialisation du réseau avec une liste de couches
        
        self.layers = layers # Liste des couches du réseau (que l'on stock pour permettre de parcourir chaque couche dans l'ordre)

    def forward(self, inputs): # Propagation avant à travers le réseau pour retourner la sortie
        self.activations = [inputs] # Stocke les activations de chaque couche pour le backpropagation
        output = inputs # Initialisation de la sortie avec les entrées (pour la première couche)
        for layer in self.layers:
            output = layer.forward(output) # Mise à jour de la sortie à chaque couche pour la suivante
            self.activations.append(output) # Stocke l'activation de la couche courante pour le backpropagation
        return output # Retourne la sortie finale du réseau
    
def train(nn, X, y, lr=0.1, epochs=1000): # Fonction d'entraînement du réseau de neurones (nn = NeuralNetwork)
    for _ in range(epochs): # Boucle sur le nombre d'itérations d'entraînement (epochs)
        for inputs, target in zip(X, y): # Boucle sur chaque paire d'entrée et de cible dans les données d'entraînement

            # Forward pass
            output = nn.forward(inputs) # Calcul de la sortie du réseau

            # Erreur de la sortie 
            delta = output - target  # Utilisation de la dérivée de la fonction de perte Log Loss avec sigmoïde

            # Backprop couche par couche
            for i in reversed(range(len(nn.layers))): # Parcours des couches en sens inverse (Backpropagation)
                layer = nn.layers[i] # Récupération de la couche courante (pour faire dans l'ordre)

                a_prev = nn.activations[i] # Récupération de l'activation de la couche précédente
                
                # Gradients
                grad_w = np.outer(delta, a_prev) # Calcul du gradient des poids
                grad_b = delta # Calcul du gradient des biais

                # Mise à jour des poids et biais
                layer.weights -= lr * grad_w # Mise à jour des poids
                layer.biases  -= lr * grad_b # Mise à jour des biais

                # delta pour la couche précédente (si c'est pas la première couche (input layer))
                if i > 0:
                    delta = (layer.weights.T @ delta) * (a_prev * (1 - a_prev)) # Calcul du delta pour la couche précédente
                    
                # En gros on propage l'erreur en arrière à travers le réseau en mettant à jour les poids et biais de chaque couche
                # pour minimiser l'erreur globale du réseau.