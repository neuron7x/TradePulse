# SPDX-License-Identifier: MIT
FROM python:3.14-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY nfpro/ ./nfpro/
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "nfpro", "--mode", "paper"]
