FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=1000 fastapi uvicorn
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]