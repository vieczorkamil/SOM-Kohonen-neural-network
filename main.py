import numpy as np
import matplotlib.pyplot as plt
import random
import math

from SOM import SOM

# Wczytanie pliku CSV
data = np.loadtxt('uruguay.csv', delimiter=',')

# Wyświetlenie zawartości tablicy NumPy
# print(data)

index = data[:,0]
x_coordinates = data[:,1]
y_coordinates = data[:,2]


cities = np.transpose(np.array([x_coordinates, y_coordinates]))

# print(len(cities))

plt.scatter(cities[:,0], cities[:,1], edgecolors='r', marker='o',s=3)
plt.show(block=True)

network = SOM(cities, len(cities) * 8, 0.9997,30000)
solution = network.train()
normCities = network.getCitiesNormalize()

solution = network.denormalize(solution)

plt.plot(solution[:,0], solution[:,1])
plt.scatter(cities[:,0], cities[:,1], edgecolors='r', marker='o',s=3)
plt.show(block=True)