# SPDX-License-Identifier: MIT
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nfpro/ ./nfpro/
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "nfpro", "--mode", "paper"]
