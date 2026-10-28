# Development example. Release builder must resolve/record the base-image digest.
# Build context: repository root, not this directory.
ARG OBJVIEW_BASE_IMAGE=python:3.10-slim
FROM ${OBJVIEW_BASE_IMAGE}
LABEL org.objview.smoke.role="algorithm"
WORKDIR /opt/objview
RUN pip install --no-cache-dir httpx==0.28.1
COPY benchmark/submission_client.py /opt/objview/submission_client.py
COPY benchmark/examples/submission/algorithm.py benchmark/examples/submission/random_algorithm.py /opt/objview/
COPY benchmark/examples/submission/run.sh benchmark/examples/submission/run-random.sh /opt/objview/
USER 65532:65532
ENTRYPOINT ["sh", "/opt/objview/run.sh"]
