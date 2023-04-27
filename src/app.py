from os import environ
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, Response
from pydantic import BaseModel
import uvicorn
import numpy as np
from SOM import SOM


app = FastAPI(title="KOHONEN SOM - TSP")
coordinates_array = None
network = None
result = None
report = None


class Coordinates(BaseModel):
    x: float
    y: float


class CoordinateData(BaseModel):
    coordinates: List[Coordinates]


def backgroud_task() -> None:
    print("Start background task")
    app.result = "In progress ... "
    app.network.train(report=app.report)
    app.result = "Done !!!"
    print("End background task")


@app.on_event("startup")
async def startup_event():
    app.coordinates_array = None
    app.network = None
    app.result = "Not started yet"
    app.report = True


@app.get("/", tags=["Default"], status_code=200)
async def read_root():
    return {"Hello world": "TSP solution using Koheonen network"}


@app.post("/setPoints", status_code=200)
async def set_points(data: CoordinateData):
    results = []
    coordinates_list = []
    for i, coordinate in enumerate(data.coordinates):
        coordinates_list.append([coordinate.x, coordinate.y])
        result = f"{i+1}: x={coordinate.x}, y={coordinate.y}"
        results.append(result)

    app.coordinates_array = np.array(coordinates_list)
    return {"results": results}


@app.post("/uploadPoints", status_code=200)
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
    return {"results": results}


@app.post("/findSolution", status_code=200)
async def find_solution(background_tasks: BackgroundTasks, number_of_epoch: int = Form(), full_report: List[bool] = Form(...)):
    app.network = SOM(app.coordinates_array, len(app.coordinates_array) * 8, 0.9997, int(number_of_epoch))
    app.result = "Started ..."
    app.report = full_report[0]
    background_tasks.add_task(backgroud_task)

    return {"Process": app.result}


@app.get("/getResult", status_code=200)
async def get_result():
    print(app.network.get_progress())
    if app.result != "Done !!!":
        return {f"Process: {app.network.get_progress()}%"}
    return {f"Process: {app.result}"}


@app.get("/getStartPoints", status_code=200)
def get_start_points():
    if app.result == "Done !!!":
        with open('report/solution/start points.png', 'rb') as f:
            response_img = f.read()
        headers = {'Content-Disposition': 'inline; filename="start points.gif"'}
        return Response(response_img, headers=headers, media_type='image/png')
    return {"Error": "The search for a solution has not started"}


@app.get("/getFinalPoints", status_code=200)
def get_final_points():
    if app.result == "Done !!!":
        with open('report/solution/final epoch.png', 'rb') as f:
            response_img = f.read()
        headers = {'Content-Disposition': 'inline; filename="final epoch.gif"'}
        return Response(response_img, headers=headers, media_type='image/png')
    return {"Error": "The search for a solution has not started"}


@app.get('/getSolution', status_code=200)
async def get_solution():
    if app.result == "Done !!!":
        if app.report is True:
            with open('report/solution.gif', 'rb') as f:
                response_img = f.read()
            headers = {'Content-Disposition': 'inline; filename="solution.gif"'}
            return Response(response_img, headers=headers, media_type='image/gif')
        return {"Error": "A faster (not full) report was selected"}
    return {"Error": "The search for a solution has not started"}


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=int(environ.get("PORT", 5000)))
