from mpl_toolkits.basemap import Basemap
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
    CITIES = None
    def __init__(self, inputPoints: np.ndarray, nnSize: int, learningRate: float, epochs: int=10000) -> None:
        self.CITIES = inputPoints
        self.POINTS = self._normalize(inputPoints)
        self.nnSize = nnSize
        self.n = nnSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.progress = 0

    
    def train(self, report: bool=False, reportPath: str="./report"):
        global progress
        if report:
            if os.path.exists(reportPath):
                shutil.rmtree(reportPath) 
            os.makedirs(reportPath)
            name = f"{reportPath}/epoch 0.png"
            # self._plotResult(name, plotNeurons=False)
            self._plotResultMap(name, plotNeurons=False)

        self._initNetwork()
        for i in range(1, self.epochs + 1):
            self.progress = i/self.epochs*100
            # print(f"Progress {progress} %")
            # print(f"\t Training in progress. Epoch {i}/{self.epochs}", end="\r")

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
                    # self._plotResult(name)
                    self._plotResultMap(name, title=f" - after {i} epochs")

        if report:
            self._makeGif(saveName=f"{reportPath}/solution.gif")
            self._printReport()

        return self.NN
    

    def get_progress(self) -> float:
        return self.progress


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
        if radix < 1:
            radix = 1

        # Compute the circular network distance to the center
        deltas = np.absolute(index - np.arange(self.nnSize))
        distances = np.minimum(deltas, self.nnSize - deltas)

        return np.exp((-1*distances**2) / (2*(radix**2)))


    def _plotResult(self, saveName: str, plotNeurons: bool=True) -> None:
        plt.axis('off')
        if plotNeurons:
            plt.scatter(self.POINTS[:,0], self.POINTS[:,1], edgecolors='cyan', marker='o',s=2.5)
        plt.plot(self.NN[:,0], self.NN[:,1], color="purple")
        plt.savefig(saveName, bBSox_inches='tight', pad_inches=0, dpi=200)
        plt.close()


    def _printReport(self) -> None:
        ''' A good place to add some functionality in the future '''
        print("Done !!!")
    

    def _plotResultMap(self, saveName: str, plotNeurons: bool=True, title: str=""):
        # How much to zoom from coordinates (in degrees)
        zoomScale = 1

        minX = min(self.CITIES[:,0])
        minY = min(self.CITIES[:,1])
        maxX = max(self.CITIES[:,0])
        maxY = max(self.CITIES[:,1])

        # Setup the bounding box for the zoom and bounds of the map
        bBSox = [minX-zoomScale,maxX+zoomScale, minY-zoomScale,maxY+zoomScale]

        # plt.figure(figsize=(12,12))
        # Define the projection, scale, the corners of the map, and the resolution.
        m = Basemap(projection='merc',llcrnrlat=bBSox[0],urcrnrlat=bBSox[1],\
                    llcrnrlon=bBSox[2],urcrnrlon=bBSox[3],lat_ts=10,resolution='i')
        
        # Draw coastlines and fill continents and water with color
        m.drawcoastlines()
        m.fillcontinents(color='gray',lake_color='dodgerblue')

        # draw parallels, meridians, and color boundaries
        m.drawparallels(np.arange(bBSox[0],bBSox[1],(bBSox[1]-bBSox[0])/5),labels=[1,0,0,0], linewidth=0.15)
        m.drawmeridians(np.arange(bBSox[2],bBSox[3],(bBSox[3]-bBSox[2])/5),labels=[0,0,0,1],rotation=45, linewidth=0.15)
        m.drawmapboundary(fill_color='dodgerblue')


        if plotNeurons:
            # build and plot neurons coordinates onto map
            self.NN_norm = self._denormalize(self.NN)

            x_line, yLine = m(self.NN_norm[:,1], self.NN_norm[:,0])
            m.plot(x_line, yLine, '-', markersize=5, linewidth=1.1, color="orange") 
        
        # build and plot coordinates onto map
        x,y = m(self.CITIES[:,1],self.CITIES[:,0])
        m.scatter(x,y,marker='.',color='yellow',s=8, zorder=1)

        plt.title(f"SOM{title}")
        plt.savefig(saveName, format='png', dpi=500, bbox_inches='tight')
        plt.close()
    
    
    def _makeGif(self, frameFolder: str="report/*.png", saveName: str="solution.gif") -> None:
        fileNames = [name for name in glob.glob(frameFolder)]
        fileNames = sorted(fileNames, key=lambda ts : int(re.findall('\d+', ts)[0]))
        frames = [Image.open(image) for image in fileNames]
        frame_one = frames[0]
        frame_one.save(saveName, format="GIF", append_images=frames,
                       save_all=True, duration=220, loop=1)
        