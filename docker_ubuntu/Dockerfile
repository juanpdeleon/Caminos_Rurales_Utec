# Imagen base oficial de NVIDIA con soporte CUDA y Ubuntu 22.04
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Evita la interacción con apt
ARG DEBIAN_FRONTEND=noninteractive

# Instala Python 3.10, pip y otras dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Alias para python y pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Directorio de trabajo
WORKDIR /camineria

# Copia archivos necesarios
COPY requirements.txt .
COPY inicio.py .
COPY libreria.py .
COPY axis.pt .
COPY vehiculos.pt .
COPY config.json .

# Instala dependencias de Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Comando por defecto
CMD ["python", "inicio.py"]
