# SPDX-License-Identifier: MIT

FROM python:3.13-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on \
    VIRTUAL_ENV=/opt/tradepulse/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /build

FROM base AS builder
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock ./
RUN python -m venv "$VIRTUAL_ENV" \
    && "$VIRTUAL_ENV/bin/pip" install --upgrade pip setuptools wheel \
    && "$VIRTUAL_ENV/bin/pip" install --no-deps -r requirements.lock
COPY pyproject.toml README.md VERSION sample.csv ./
COPY data/sample.csv ./data/sample.csv
COPY analytics ./analytics
COPY application ./application
COPY backtest ./backtest
COPY core ./core
COPY domain ./domain
COPY execution ./execution
COPY interfaces ./interfaces
COPY libs ./libs
COPY markets ./markets
COPY nfpro ./nfpro
COPY observability ./observability
COPY scripts ./scripts
COPY src ./src
COPY tools ./tools
COPY configs ./configs
COPY conf ./conf
RUN "$VIRTUAL_ENV/bin/pip" install --no-deps --no-build-isolation .

FROM base AS runtime
WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system tradepulse \
    && useradd --system --create-home --gid tradepulse tradepulse
COPY --from=builder /opt/tradepulse/venv /opt/tradepulse/venv
COPY --from=builder /build/nfpro ./nfpro
COPY --from=builder /build/application ./application
COPY --from=builder /build/core ./core
COPY --from=builder /build/execution ./execution
COPY --from=builder /build/markets ./markets
COPY --from=builder /build/domain ./domain
COPY --from=builder /build/analytics ./analytics
COPY --from=builder /build/backtest ./backtest
COPY --from=builder /build/interfaces ./interfaces
COPY --from=builder /build/tools ./tools
COPY --from=builder /build/configs ./configs
COPY --from=builder /build/conf ./conf
COPY --from=builder /build/libs ./libs
COPY --from=builder /build/observability ./observability
COPY --from=builder /build/scripts ./scripts
COPY --from=builder /build/src ./src
COPY --from=builder /build/README.md ./README.md
COPY --from=builder /build/VERSION ./VERSION
COPY --from=builder /build/sample.csv ./sample.csv
COPY --from=builder /build/data/sample.csv ./data/sample.csv
USER tradepulse
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8001/metrics || exit 1
CMD ["python", "-m", "nfpro", "--mode", "paper"]
