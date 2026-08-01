# NOTE: not built locally — no Docker daemon on the development machine.
# CI builds this on push; see the verification matrix in README.md.

FROM python:3.13-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only torch: the service scores one window at a time, where a device
# transfer would cost more than the forward pass. The default wheel would pull
# in several GB of CUDA runtime for no benefit.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu . \
 && pip install --target=/deps --extra-index-url https://download.pytorch.org/whl/cpu .

FROM python:3.13-slim AS runtime

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /deps /usr/local/lib/python3.13/site-packages
COPY --from=builder /build/src ./src
COPY configs ./configs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    FI_SERVICE__HOST=0.0.0.0

USER app
EXPOSE 8000

# Liveness only. Readiness is a separate endpoint and is false until a model is
# loaded, so it must not gate container health.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

ENTRYPOINT ["python", "-m", "football_insights.cli"]
CMD ["serve", "--match", "Sample_Game_2", "--speed", "8"]
