# Build context: the directory containing all four downloaded example files.
# Use an operator-approved base matching the worker CPU architecture.
ARG OBJVIEW_BASE_IMAGE=registry.mthreads.com/presale/devtech/torchcodec:d0922
FROM ${OBJVIEW_BASE_IMAGE}
WORKDIR /opt/objview
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Configure an internal Python package mirror or preinstall this dependency.
RUN python -m pip install --no-cache-dir httpx==0.28.1
COPY submission_client.py algorithm.py run.sh /opt/objview/
USER 65532:65532
ENTRYPOINT ["sh", "/opt/objview/run.sh"]
