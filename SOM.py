import numpy as np
import matplotlib.pyplot as plt

class SOM:
    POINTS = None
    NN     = None
    def __init__(self, inputPoints: np.ndarray, nnSize: int, learningRate: float, epochs: int=10000) -> None:
        self.POINTS = self.normalize(inputPoints)
        self.nnSize = nnSize
        self.n = nnSize
        self.learningRate = learningRate
        self.epochs = epochs

    def normalize(self, array: np.ndarray) -> np.ndarray:
        # return (array - np.min(array)) / (np.max(array) - np.min(array))
        self.norms = np.linalg.norm(array, axis=0)
        return array / self.norms
        # return array / np.linalg.norm(array, axis=0)

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        return array * self.norms

    def _initNetwork(self) -> np.ndarray:
        self.NN = np.random.rand(self.nnSize, 2)
        return self.NN
    
    def train(self):
        self._initNetwork()
        for i in range(self.epochs):
            print(f"\t Epoch {i}/{self.epochs}", end="\r")
            point = self.randomSample()
            winnerIndex = self.selectWinner(point)
            gaussian = self.getNeighborhood(winnerIndex, self.n//8)
            self.NN += gaussian[:,np.newaxis] * self.learningRate * (point - self.NN)

            # print(f"\t NN {self.NN[0][0]}/{self.NN[0][1]}", end="\r")
            # Decay the variables
            self.learningRate = self.learningRate * 0.99997
            self.n = self.n * 0.9997

            if self.learningRate < 0.001:
                break

            # if i%500 == 0:
            #     plt.plot(self.NN[:,0], self.NN[:,1])
            #     plt.scatter(self.POINTS[:,0], self.POINTS[:,1], edgecolors='r', marker='o',s=3)
            #     plt.show(block=True)


        # plt.plot(self.NN[:,0], self.NN[:,1])
        # plt.scatter(self.POINTS[:,0], self.POINTS[:,1], edgecolors='r', marker='o',s=3)
        # plt.show(block=True)
        return self.NN

    def randomSample(self) -> np.array:
        return self.POINTS[np.random.choice(len(self.POINTS), size=1, replace=False)]
    

    def selectWinner(self, point: np.array) -> int:
        return np.linalg.norm(self.NN - point, axis=1).argmin()
    

    def getNeighborhood(self, index: int, radix: int):
        # Impose an upper bound on the radix to prevent NaN and blocks
        if radix < 1:
            radix = 1

        # Compute the circular network distance to the center
        deltas = np.absolute(index - np.arange(self.nnSize))
        distances = np.minimum(deltas, self.nnSize - deltas)

        return np.exp((-1*distances**2) / (2*(radix**2)))


    def getCitiesNormalize(self) -> np.ndarray: # FIXME:  
        return self.POINTS
    
    def getNetwork(self) -> np.ndarray:
        return self.NN



    def getShape(self):
        return(self.NN.shape[0])
