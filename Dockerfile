# SPDX-License-Identifier: MIT

FROM python:3.13-slim AS builder

ARG APP_HOME=/app
ARG VENV_PATH=/opt/venv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY requirements.lock ./

RUN set -euxo pipefail \
    && apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && umask 027 \
    && python -m venv "${VENV_PATH}" \
    && "${VENV_PATH}/bin/pip" install --no-cache-dir --upgrade pip \
    && "${VENV_PATH}/bin/pip" install --no-cache-dir --require-hashes -r requirements.lock \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY nfpro/ "${APP_HOME}/nfpro/"
COPY container_entrypoint.py "${APP_HOME}/container_entrypoint.py"

RUN set -euxo pipefail \
    && if ! getent group 65532 >/dev/null; then groupadd --system --gid 65532 nonroot; fi \
    && if ! getent passwd 65532 >/dev/null; then \
        useradd --system --no-log-init --uid 65532 --gid 65532 --create-home --home-dir /home/nonroot nonroot; \
    fi \
    && umask 027 \
    && chown -R nonroot:nonroot "${APP_HOME}" "${VENV_PATH}" /home/nonroot \
    && chmod 0550 "${APP_HOME}/container_entrypoint.py"

FROM gcr.io/distroless/python3-debian12

ARG APP_HOME=/app
ARG VENV_PATH=/opt/venv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME="${APP_HOME}" \
    VENV_PATH="${VENV_PATH}" \
    APP_ENV_FILE=/run/secrets/tradepulse.env \
    APP_UMASK=027 \
    PATH="${VENV_PATH}/bin:/usr/local/bin:/usr/bin:/bin"

WORKDIR "${APP_HOME}"

COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group
COPY --from=builder "${VENV_PATH}" "${VENV_PATH}"
COPY --from=builder --chown=65532:65532 "${APP_HOME}" "${APP_HOME}"

USER nonroot:nonroot

ENTRYPOINT ["/opt/venv/bin/python", "/app/container_entrypoint.py"]
CMD ["-m", "nfpro", "--mode", "paper"]
