FROM ghcr.io/open-webui/open-webui:latest-cuda

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr ghostscript && \
    rm -rf /var/lib/apt/lists/*

CMD ["bash", "start.sh"]
