# SPDX-License-Identifier: MIT

FROM python:3.13-slim AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /tmp/build
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

FROM python:3.13-slim

ENV VIRTUAL_ENV=/opt/venv \
    PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_UID=1000 \
    APP_GID=1000 \
    APP_USER=nfpro \
    APP_GROUP=nfpro \
    READ_ONLY_ROOT=1

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

RUN set -eu \ 
    && groupadd --system --gid "${APP_GID}" "${APP_GROUP}" \ 
    && useradd --system --no-log-init --uid "${APP_UID}" --gid "${APP_GID}" --home /home/"${APP_USER}" --create-home "${APP_USER}" \ 
    && mkdir -p /app /var/lib/nfpro \ 
    && chown -R "${APP_UID}:${APP_GID}" /app /var/lib/nfpro /home/"${APP_USER}"

WORKDIR /app
COPY --chown=${APP_UID}:${APP_GID} nfpro/ ./nfpro/

RUN chmod -R a-w /app

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

VOLUME ["/var/lib/nfpro"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD ["python", "-m", "nfpro", "--mode", "healthcheck"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "nfpro", "--mode", "paper"]
