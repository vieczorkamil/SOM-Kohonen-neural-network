FROM python:3.11-slim

WORKDIR /src
COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["python", "app.py"]
EXPOSE 5000