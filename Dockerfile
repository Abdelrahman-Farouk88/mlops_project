FROM python:3.9-slim

WORKDIR /app

COPY req_for_docker/req_fastapi.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY API/main.py /app/API/main.py

EXPOSE 8000

CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]