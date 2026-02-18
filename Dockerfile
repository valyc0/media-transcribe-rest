FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG INSTALL_DIARIZATION=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY requirements-diarization.txt /app/requirements-diarization.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && if [ "$INSTALL_DIARIZATION" = "true" ]; then pip install -r /app/requirements-diarization.txt; fi

COPY app.py /app/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
