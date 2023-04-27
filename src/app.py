from os import environ
from typing import List 
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi import Response
from pydantic import BaseModel
import uvicorn
import numpy as np

from SOM import SOM


app = FastAPI(title="KOHONEN SOM - TSP") 
coordinates_array = None
network = None
result = None


class Coordinates(BaseModel):
    x: float
    y: float


class CoordinateData(BaseModel):
    coordinates: List[Coordinates]


class EpochNumber(BaseModel):
    number: int


def backgroud_task() -> None:
    print("????????????????????????????????????????????????????? start background task")
    app.result = "In progress ... "
    app.network.train(report=True)
    app.result = "Done !!!"
    print("????????????????????????????????????????????????????? end background task")


@app.on_event("startup")
async def startup_event():
    app.coordinates_array = None
    app.network = None
    app.result = "Not started yet"


@app.get("/", tags=["Default"])
async def read_root():
    return {"Hello world": "TSP solution using Koheonen network"}


@app.post("/setPoints")
async def set_points(data: CoordinateData):
    results = []
    coordinates_list = []
    for i, coordinate in enumerate(data.coordinates):
        coordinates_list.append([coordinate.x, coordinate.y])
        result = f"{i+1}: x={coordinate.x}, y={coordinate.y}"
        results.append(result)

    app.coordinates_array = np.array(coordinates_list)
    # print(type(app.coordinates_array))
    # print(app.coordinates_array)
    # print(app.coordinates_array[0])
    return {"results": results}


@app.post("/uploadPoints")
async def upload_points(file: UploadFile = File(...)):
    results = []
    contents = file.file.read().decode("utf-8")
    data = np.loadtxt(contents.splitlines(), delimiter=',', encoding="utf8")
    index = data[:,0]
    x_coordinates = data[:,1]
    y_coordinates = data[:,2]

    for i, x, y in zip(index, x_coordinates, y_coordinates):
        result = f"{i}: x={x}, y={y}"
        results.append(result)

    app.coordinates_array = np.transpose(np.array([x_coordinates, y_coordinates]))

    # print(type(app.coordinates_array))
    # print(app.coordinates_array)
    # print(app.coordinates_array[0])

    return {"results": results}


@app.post("/findSolution")
async def find_solution(background_tasks: BackgroundTasks, number: EpochNumber):
    app.network = SOM(app.coordinates_array, len(app.coordinates_array) * 8, 0.9997, int(number.number))
    app.result = "Started ..."
    background_tasks.add_task(backgroud_task)

    return {"Process": app.result}


@app.get("/getResult")
async def get_result():
    print(app.network.get_progress())
    if app.network.get_progress() == 100.0:
        return {f"Process: {app.result}"}
    else:
        return {f"Process: {app.result} {app.network.get_progress()}%"}


@app.get('/getSolution', status_code=200)
async def get_solution():
     with open('report/solution.gif', 'rb') as f:
         response_img = f.read()
     headers = {'Content-Disposition': 'inline; filename="solution.gif"'}
     return Response(response_img, headers=headers, media_type='image/gif')


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=int(environ.get("PORT", 5000)))