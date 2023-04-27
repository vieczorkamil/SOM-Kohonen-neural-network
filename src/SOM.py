import shutil
import math
import glob
import os
import re
from mpl_toolkits.basemap import Basemap
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class SOM:
    NN = None
    POINTS = None
    CITIES = None

    def __init__(self, inputPoints: np.ndarray, nnSize: int, learningRate: float, epochs: int = 10000) -> None:
        self.CITIES = inputPoints
        self.POINTS = self._normalize(inputPoints)
        self.nnSize = nnSize
        self.n = nnSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.progress = 0
        self.NN_norm = None

    def train(self, report: bool = False, reportPath: str = "./report"):
        if os.path.exists(reportPath):
            shutil.rmtree(reportPath)
        os.makedirs(reportPath)
        os.makedirs(f"{reportPath}/solution")
        if report:
            name = f"{reportPath}/epoch 0.png"
            self._plotResultMap(name, plotNeurons=False)
        name = f"{reportPath}/solution/start points.png"
        self._plotResultMap(name, plotNeurons=False)

        self._initNetwork()
        j = 0
        for i in range(1, self.epochs + 1):
            point = self._randomSample()
            winnerIndex = self._selectWinner(point)
            gaussian = self._getNeighborhood(winnerIndex, self.n // 8)
            self.NN += gaussian[:,np.newaxis] * self.learningRate * (point - self.NN)

            # Decay the variables
            self.learningRate = self.learningRate * 0.99997
            self.n = self.n * 0.9997

            if self.learningRate < 0.001:
                break

            # Keep only 15 photos to save gif to optimize memory
            if int(self.epochs / 1000) > 15:
                temp = math.floor(self.epochs / 15)
            else:
                temp = 1000

            if report:
                if i % temp == 0:
                    j += 1
                    if j == 15:
                        name = f"{reportPath}/epoch {self.epochs}.png"
                        self._plotResultMap(name, title=f" - after {self.epochs} epochs")
                    else:
                        name = f"{reportPath}/epoch {temp * j}.png"
                        self._plotResultMap(name, title=f" - after {temp * j} epochs")

            if i == self.epochs:
                name = f"{reportPath}/solution/final epoch.png"
                self._plotResultMap(name, title=f" - final epoch {i}")

            self.progress = (i / self.epochs) * 100
            if self.progress == 100.0:
                self.progress = 99.99  # Just to indicate that not everything is done - still wait for gif creation

        if report:
            self._makeGif(saveName=f"{reportPath}/solution.gif")
            self._printReport()

        self.progress = 100.0  # Now everything is done

        return self.NN

    def get_progress(self) -> float:
        return round(self.progress, 2)

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        self.norms = np.linalg.norm(array, axis=0)
        return array / self.norms

    def _denormalize(self, array: np.ndarray) -> np.ndarray:
        return array * self.norms

    def _initNetwork(self) -> np.ndarray:
        self.NN = np.random.rand(self.nnSize, 2)
        return self.NN

    def _randomSample(self) -> np.array:
        return self.POINTS[np.random.choice(len(self.POINTS), size=1, replace=False)]

    def _selectWinner(self, point: np.array) -> int:
        return np.linalg.norm(self.NN - point, axis=1).argmin()

    def _getNeighborhood(self, index: int, radix: int):
        # Impose an upper bound on the radix to prevent NaN and blocks
        radix = max(radix, 1)

        # Compute the circular network distance to the center
        deltas = np.absolute(index - np.arange(self.nnSize))
        distances = np.minimum(deltas, self.nnSize - deltas)

        return np.exp((-1 * distances ** 2) / (2 * (radix ** 2)))

    def _plotResult(self, saveName: str, plotNeurons: bool = True) -> None:
        plt.axis('off')
        if plotNeurons:
            plt.scatter(self.POINTS[:,0], self.POINTS[:,1], edgecolors='cyan', marker='o',s=2.5)
        plt.plot(self.NN[:,0], self.NN[:,1], color="purple")
        plt.savefig(saveName, bBSox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    def _printReport(self) -> None:
        ''' A good place to add some functionality in the future '''
        print("Done !!!")

    def _plotResultMap(self, saveName: str, plotNeurons: bool = True, title: str = ""):
        # How much to zoom from coordinates (in degrees)
        zoomScale = 1

        minX = min(self.CITIES[:,0])
        minY = min(self.CITIES[:,1])
        maxX = max(self.CITIES[:,0])
        maxY = max(self.CITIES[:,1])

        # Setup the bounding box for the zoom and bounds of the map
        bBSox = [minX - zoomScale, maxX + zoomScale, minY - zoomScale, maxY + zoomScale]

        # Define the projection, scale, the corners of the map, and the resolution.
        m = Basemap(projection='merc', llcrnrlat=bBSox[0], urcrnrlat=bBSox[1], llcrnrlon=bBSox[2], urcrnrlon=bBSox[3], lat_ts=10, resolution='i')

        # Draw coastlines and fill continents and water with color
        m.drawcoastlines()
        m.fillcontinents(color='gray', lake_color='dodgerblue')

        # draw parallels, meridians, and color boundaries
        m.drawparallels(np.arange(bBSox[0], bBSox[1], (bBSox[1] - bBSox[0]) / 5), labels=[1,0,0,0], linewidth=0.15)
        m.drawmeridians(np.arange(bBSox[2],bBSox[3], (bBSox[3] - bBSox[2]) / 5), labels=[0,0,0,1], rotation=45, linewidth=0.15)
        m.drawmapboundary(fill_color='dodgerblue')

        if plotNeurons:
            # build and plot neurons coordinates onto map
            self.NN_norm = self._denormalize(self.NN)

            x_line, yLine = m(self.NN_norm[:,1], self.NN_norm[:,0])
            m.plot(x_line, yLine, '-', markersize=5, linewidth=1.1, color="orange")

        # build and plot coordinates onto map
        x,y = m(self.CITIES[:,1],self.CITIES[:,0])
        m.scatter(x, y, marker='.', color='yellow', s=8, zorder=1)

        plt.title(f"SOM{title}")
        plt.savefig(saveName, format='png', dpi=500, bbox_inches='tight')
        plt.close()

    def _makeGif(self, frameFolder: str = "report/*.png", saveName: str = "solution.gif") -> None:
        fileNames = [name for name in glob.glob(frameFolder)]
        fileNames = sorted(fileNames, key=lambda ts: int(re.findall(r'\d+', ts)[0]))
        frames = [Image.open(image) for image in fileNames]
        frame_one = frames[0]
        frame_one.save(saveName, format="GIF", append_images=frames, save_all=True, duration=220, loop=1)
