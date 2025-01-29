#tutaj tworzymy model naszej prostej sieci neuronowej
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.weights = 2*np.random.random((3,1))-1

    # def __new__(cls, *args, **kwargs):
    #     return object.__new__(SimpleNeuralNetwork)

    def __repr__(self):
        return (f"nowa sieć neuronowa oparata na klasie: {self.__class__.__name__}\n"
                f"Wylosowano wagi:\n{self.weights}\n")
