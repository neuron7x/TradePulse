# SPDX-License-Identifier: MIT

FROM python:3.13-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && python -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.lock ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock \
    && python -m pip uninstall -y pip

FROM python:3.13-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV

WORKDIR /app

COPY nfpro/ ./nfpro/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "nfpro", "--mode", "paper"]
