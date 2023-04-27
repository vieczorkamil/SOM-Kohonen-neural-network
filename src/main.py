import numpy as np
from SOM import SOM

DATA_PATH = "data/Usa.csv" 

def loadData(path: str) ->np.ndarray:
    # Read data from CSV
    data = np.loadtxt(path, delimiter=',', encoding="utf8")
    index = data[:,0]
    x_coordinates = data[:,1]
    y_coordinates = data[:,2]

    return np.transpose(np.array([x_coordinates, y_coordinates]))


def main():
    cities = loadData(DATA_PATH)

    # print(type(cities))
    # print(cities)
    # print(cities[0])
    # exit()

    network = SOM(cities, len(cities) * 8, 0.9997, 30000)
    network.train(report=True)


if __name__ == "__main__":
    main()