# Kafka Component

This Kustomize component wraps the Bitnami Kafka Helm chart with TLS and SASL/SCRAM enabled for TradePulse. It is intended to be layered into staging and production overlays to supply a secure ingestion cluster.

## Secrets

The component expects the following Kubernetes secrets:

- `tradepulse-kafka-broker-tls` – contains `ca.crt`, `tls.crt`, and `tls.key` used by brokers.
- `tradepulse-kafka-sasl` – contains `client-passwords`, `inter-broker-password`, and `controller-password` entries required by the Helm chart.
- `tradepulse-kafka-client` – bundles the CA and client certificate material for TradePulse workloads.

See `deploy/kustomize/overlays/staging/kafka-secrets.yaml` for a complete example wired to the local development certificates under `configs/local/kafka`.

## External Access

The component enables KRaft mode with both internal and external TLS listeners. Each overlay should pair the Helm release with a `Service` policy (for example, LoadBalancer or NodePort) that aligns with the environment. The staging overlay exposes a LoadBalancer for smoke tests.
