# Development example. Release builder must resolve/record the base-image digest.
# Build context: repository root, not this directory.
FROM python:3.10-slim
LABEL org.objview.smoke.role="algorithm"
WORKDIR /app
RUN pip install --no-cache-dir httpx==0.28.1
COPY benchmark/submission_client.py /app/submission_client.py
COPY benchmark/examples/submission/algorithm.py benchmark/examples/submission/run.sh /app/
USER 65532:65532
ENTRYPOINT ["sh", "/app/run.sh"]
