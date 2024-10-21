FROM python:3.11

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

RUN groupadd -r user && useradd -r -g user -m user

RUN mkdir -p /tmp/whisper_cache && chmod 777 /tmp/whisper_cache

RUN chown -R user:user /app

ENV NUMBA_CACHE_DIR /tmp

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

USER user

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
