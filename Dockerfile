# syntax=docker/dockerfile:1
#
# Five stages, arranged so the two expensive ones never invalidate each other:
#
#   frontend ──┐
#              ├─> builder ──┐
#   pysrc ─────┤             ├─> runtime
#              └─> demomodel ┘
#
# `demomodel` trains and exports the demo model and depends on `pysrc` alone, so
# editing the React app cannot retrain it; `frontend` reaches only `builder`, so
# a Python change does not reinstall npm packages. The final image contains
# neither Node nor PyTorch.

ARG PYTHON_IMAGE=python:3.13-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------- frontend
FROM node:22-slim AS frontend

WORKDIR /workspace
# Manifests first: this layer is the npm install, and it should survive every
# change to the application source.
COPY demo/package.json demo/package-lock.json ./demo/
RUN npm --prefix demo ci

COPY demo ./demo
# vite writes to ../src/football_insights/serving/static, which is why the demo
# lives under a workspace root rather than being built in place.
RUN npm --prefix demo run build \
 && test -f src/football_insights/serving/static/index.html

# ---------------------------------------------------------------- python source
FROM ${PYTHON_IMAGE} AS pysrc

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# ---------------------------------------------------------------- demo model
FROM pysrc AS demomodel
ARG TORCH_INDEX

# CPU-only torch: this stage trains for well under a minute on a few thousand
# generated windows, and the default wheel would pull several GB of CUDA runtime
# to do it. None of this reaches the runtime image either way.
RUN pip install --extra-index-url "${TORCH_INDEX}" ".[train]"

# Deterministic for its seed, and parity between PyTorch and ONNX Runtime is
# asserted inside the command — a graph that disagrees with the model it came
# from fails the build rather than shipping.
RUN football-insights demo-model --out /artifacts/registry \
 && test -f /artifacts/registry/demo-synthetic-gru.onnx

# ---------------------------------------------------------------- wheel + venv
FROM pysrc AS builder

COPY --from=frontend /workspace/src/football_insights/serving/static \
     ./src/football_insights/serving/static

RUN pip install build setuptools wheel

# One wheel, and the runtime installs exactly it. The `find` resolves the file
# explicitly rather than handing pip a glob, and the zipfile listing fails the
# build if the page did not make it into the artifact — the alternative is an
# image that starts happily and 404s its own front page.
RUN python -m build --wheel --no-isolation -o /dist \
 && WHEEL="$(find /dist -maxdepth 1 -name 'football_insights-*.whl' -print -quit)" \
 && test -n "${WHEEL}" \
 && python -m zipfile -l "${WHEEL}" | grep -q 'serving/static/index.html' \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir "${WHEEL}"

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.source="https://github.com/grounzero/live-football-insight-engine"
LABEL org.opencontainers.image.title="Live Football Insight Engine"
LABEL org.opencontainers.image.description="Public demo: synthetic replay, ONNX scoring, qualified insights"
LABEL org.opencontainers.image.licenses="MIT"

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# The venv is built at this exact path in `builder` and copied verbatim, so the
# runtime layer carries no pip cache and no build tooling.
COPY --from=builder /opt/venv /opt/venv
# The whole directory, not just the .onnx: the exporter writes weights to a
# sibling .onnx.data file that ONNX Runtime resolves by name at load time.
COPY --from=demomodel /artifacts/registry /opt/artifacts/registry
COPY configs ./configs

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Bind every interface: the loopback default is right on a development
    # machine and makes a container unreachable from outside itself.
    FI_SERVICE__HOST=0.0.0.0 \
    # A hosted demo: generated fixture, read-only looping replay, no dataset.
    FI_SERVICE__PUBLIC_DEMO=1 \
    FI_REPLAY__LOOP=1 \
    FI_MODEL__MODEL_NAME=demo-synthetic-gru \
    FI_PATHS__REGISTRY_DIR=/opt/artifacts/registry \
    # A default only. Railway, Fly, Heroku and Cloud Run all inject PORT, and
    # `serve` reads it above every other configuration layer.
    PORT=8000

USER app
EXPOSE 8000

# Liveness only. Readiness is a separate endpoint and reports rather more, so it
# gates traffic (see railway.toml) but must not decide whether to restart.
#
# The port is read from the environment in Python: exec-form CMD does not go
# through a shell, so a literal $PORT here would never expand and the probe
# would check 8000 while the service listened somewhere else.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request; port=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+port+'/health', timeout=2).status==200 else 1)"]

# CMD without an ENTRYPOINT, deliberately. An entrypoint of `football-insights`
# would swallow the first argument of every `docker run … python -c` or
# `… sh -c`, so inspecting the image would need `--entrypoint` on each call and
# the documented acceptance checks would not run as written.
CMD ["football-insights", "serve"]
