FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0

COPY requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

COPY . ./

ENTRYPOINT ["python", "export_dataset.py"]
CMD ["--dataset-dir", "/data/I24", "--output-dir", "/output"]
