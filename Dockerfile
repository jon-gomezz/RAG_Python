# Imagen base ligera con Python.
FROM python:3.12-slim

# Evita ficheros .pyc y fuerza salida sin búfer (logs en tiempo real).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias primero para aprovechar la caché de capas de Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación.
COPY app ./app

EXPOSE 8000

# Arranca el servidor accesible desde fuera del contenedor.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
