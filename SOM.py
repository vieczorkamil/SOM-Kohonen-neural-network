import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shutil 
import glob
import os
import re


class SOM:
    POINTS = None
    NN     = None
    def __init__(self, inputPoints: np.ndarray, nnSize: int, learningRate: float, epochs: int=10000) -> None:
        self.POINTS = self._normalize(inputPoints)
        self.nnSize = nnSize
        self.n = nnSize
        self.learningRate = learningRate
        self.epochs = epochs

    
    def train(self, report: bool=False, reportPath: str="./report"):
        if report:
            if os.path.exists(reportPath):
                shutil.rmtree(reportPath) 
            os.makedirs(reportPath)
            name = f"{reportPath}/epoch 0.png"
            self._plotPoints(name)

        self._initNetwork()
        for i in range(1, self.epochs + 1):
            print(f"\t Training in progress. Epoch {i}/{self.epochs}", end="\r")

            point = self._randomSample()
            winnerIndex = self._selectWinner(point)
            gaussian = self._getNeighborhood(winnerIndex, self.n//8)
            self.NN += gaussian[:,np.newaxis] * self.learningRate * (point - self.NN)

            # Decay the variables
            self.learningRate = self.learningRate * 0.99997
            self.n = self.n * 0.9997

            if self.learningRate < 0.001:
                break

            if report:
                if i%1000 == 0:
                    name = f"{reportPath}/epoch {i}.png"
                    self._plotResult(name)

        if report:

            # self._makeGif(frameFolder="report/*.png", saveName=f"{reportPath}/solution.gif")
            self._makeGif(saveName=f"{reportPath}/solution.gif")
            self._printReport()

        return self.NN
    

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        self.norms = np.linalg.norm(array, axis=0)
        return array / self.norms
    

    def _initNetwork(self) -> np.ndarray:
        self.NN = np.random.rand(self.nnSize, 2)
        return self.NN
    

    def _randomSample(self) -> np.array:
        return self.POINTS[np.random.choice(len(self.POINTS), size=1, replace=False)]
    

    def _selectWinner(self, point: np.array) -> int:
        return np.linalg.norm(self.NN - point, axis=1).argmin()
    

    def _getNeighborhood(self, index: int, radix: int):
        # Impose an upper bound on the radix to prevent NaN and blocks
        if radix < 1:
            radix = 1

        # Compute the circular network distance to the center
        deltas = np.absolute(index - np.arange(self.nnSize))
        distances = np.minimum(deltas, self.nnSize - deltas)

        return np.exp((-1*distances**2) / (2*(radix**2)))
    

    def _plotPoints(self, saveName: str) -> None:
        plt.axis('off')
        # plt.plot(self.POINTS[:,0], self.POINTS[:,1], color="purple", marker='.', markersize=1.5)
        plt.scatter(self.POINTS[:,0], self.POINTS[:,1], edgecolors='purple', marker='o',s=2.5)
        plt.savefig(saveName, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()


    def _plotResult(self, saveName: str) -> None:
        plt.axis('off')
        plt.plot(self.NN[:,0], self.NN[:,1], color="cyan")
        plt.scatter(self.POINTS[:,0], self.POINTS[:,1], edgecolors='purple', marker='o',s=2.5)
        plt.savefig(saveName, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()


    def _printReport(self) -> None:
        print()
        print("Done!!!")
    

    def _makeGif(self, frameFolder: str="report/*.png", saveName: str="solution.gif") -> None:
        fileNames = [name for name in glob.glob(frameFolder)]
        fileNames = sorted(fileNames, key=lambda ts : int(re.findall('\d+', ts)[0]))
        frames = [Image.open(image) for image in fileNames]
        frame_one = frames[0]
        frame_one.save(saveName, format="GIF", append_images=frames,
                       save_all=True, duration=200, loop=1)