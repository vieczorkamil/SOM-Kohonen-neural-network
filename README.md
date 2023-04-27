
# Self organizing maps - Kohonen neural network
## Kohonen network
The Kohonen network is an example of a self-organizing map (SOM), which is a popular unsupervised learning algorithm used for data processing. The Kohonen network allows for dimensionality reduction of data and finding the internal structure of data by grouping them based on their similarity. However, it can also be applied to solve the Traveling Salesman Problem (TSP).

The Traveling Salesman Problem is to find the shortest route connecting a set of cities, visiting each city exactly once and returning to the starting point.

When applying the SOM algorithm to the TSP problem, each city is represented by a feature vector that defines its location on a plane. Then, these feature vectors are subjected to a learning process to organize them in a topology resembling a map.
## Algorithm
1. Initialization of network weights:
    - Random initial weight values are assigned to each network unit.
2. Assignment of input to the closest network unit:
    - For each input, the Euclidean distance from each network unit is calculated.
    - The network unit whose weights are closest to the input is chosen as the winner.
3. Update of the winner and its neighbors' weights:
    - The winner and its neighbors' weights are adjusted towards the input.
    - The winner receives a larger weight change than its neighbors who are further from the winner.
4. Repeat steps 2-3:
    - Steps 2-3 are repeated for each input in the training data.

## Result
The Kohonen network gives us a sorted list of network units representing clusters of input data, which can be used to determine the optimal Traveling Salesman route. To do this, the order of visiting cities on the SOM map is determined, and then this order is converted into the order of visiting cities in the real world.
## Summary
The Kohonen network allows for dimensionality reduction of data and finding the internal structure of data by grouping them based on their similarity. This algorithm is used in many fields. Using it to solve the TSP problem seems to be simpler than traditional approaches, which involve heuristics or genetic algorithms. However, it should be noted that the result of the SOM algorithm will not always be the optimal solution.

Nevertheless, the use of SOM for TSP can be a good choice for problems with a smaller number of cities, where it is difficult to apply more complicated algorithms.
# Software usage
## First steps
```
$ python -m venv venv
$ ./venv/Scripts/activate.bat
$ pip install -r requirements.txt
```
## Basic local usage
```
$ python src/main.py
```
## Fast api usage
```
$ python src/api.py
```
## Docker
```
$ docker-compose up -d --build
```
# Software usage
## Deploy
Fast api deployed [Link](https://kohonen-tsp.herokuapp.com/)
[![ci_cd](https://github.com/vieczorkamil/SOM-Kohonen-neural-network/actions/workflows/ci_cd.yaml/badge.svg)](https://github.com/vieczorkamil/SOM-Kohonen-neural-network/actions/workflows/ci_cd.yaml)
# TSP
Example results:
- USA 600 cities
<p align="center">
<img src="docs/USA_solution.gif" width="650"/>
</p>
- Poland 340 cities 
<p align="center">
<img src="docs/Poland_solution.gif" width="650"/>
</p>
- Brazil 106 cities
<p align="center">
<img src="docs/Brazil_solution.gif" width="650"/>
</p>

### Example data source
https://simplemaps.com/data/au-cities
