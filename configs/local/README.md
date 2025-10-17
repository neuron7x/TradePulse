# Local Development Assets

This directory contains certificates and supporting files for the TLS-enabled Kafka stack defined in `docker-compose.yml`.

- `kafka/ca.crt` – self-signed certificate authority for local brokers and clients.
- `kafka/server.crt` / `kafka/server.key` – server identity mounted into the Kafka container.
- `kafka/client.crt` / `kafka/client.key` – client identity mounted into the TradePulse container.

The material is **not** intended for staging or production. Replace it with environment-specific certificates when generating secrets for Kubernetes overlays.
