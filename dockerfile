FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models /app/data /app/logs

COPY ./src /app/src
COPY ./models /app/models

ENV PYTHONPATH=/app

EXPOSE 4222

CMD ["python", "src/app.py"]