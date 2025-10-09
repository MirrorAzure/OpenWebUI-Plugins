FROM ghcr.io/open-webui/open-webui:latest-cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus ghostscript && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt


CMD ["bash", "start.sh"]
