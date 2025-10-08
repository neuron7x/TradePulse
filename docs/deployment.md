# Deployment Guide

This guide describes how to promote TradePulse from staging to production while preserving observability and rollback guarantees.

## 1. Deployment Targets
- **Staging:** Mirrors production configuration with sandbox exchange credentials.
- **Production:** Uses live exchange keys, hardened networking, and audited observability pipelines.

## 2. Pre-Deployment Checklist
- ✅ All tests pass (`pytest -q` and coverage ≥ 98 %).
- ✅ Docker images built and scanned (Trivy/Grype).
- ✅ Secrets stored in the target secret manager (HashiCorp Vault, AWS Secrets Manager, or Kubernetes Secrets).
- ✅ Monitoring dashboards updated with current metric names.
- ✅ Release notes reviewed by engineering and operations leads.

## 3. Build & Package
```bash
make docker-build
make docker-push REGISTRY=registry.example.com/tradepulse TAG=v1.0.0
```
Artifacts:
- `tradepulse-core` – Python orchestration layer
- `tradepulse-exec` – Go execution engine
- `tradepulse-web` – Next.js dashboard (optional)

## 4. Infrastructure Configuration
| Component          | Recommendation |
| ------------------ | -------------- |
| Container Runtime  | Kubernetes, Nomad, or Docker Swarm |
| Database           | PostgreSQL 14+ with point-in-time recovery |
| Cache/Queue        | Redis 7+ for streaming and stateful indicators |
| Observability      | Prometheus, Grafana, Loki (or OpenTelemetry collector) |
| Secret Management  | Vault/AWS Secrets Manager/GCP Secret Manager |

## 5. Deployment Steps (Kubernetes Example)
1. **Create namespaces** for `tradepulse-staging` and `tradepulse-prod`.
2. **Apply secrets** using sealed secrets or external secrets operator.
3. **Deploy core services:**
   ```bash
   kubectl apply -f deploy/k8s/core.yaml
   kubectl apply -f deploy/k8s/execution.yaml
   ```
4. **Deploy web dashboard (optional):**
   ```bash
   kubectl apply -f deploy/k8s/web.yaml
   ```
5. **Run smoke tests:** execute `python -m interfaces.cli backtest configs/backtests/sample.yaml` against the deployed cluster via a job or port-forwarding session.

## 6. Post-Deployment Verification
- Confirm Prometheus scrapes all services and alerts are green.
- Inspect structured logs for error spikes.
- Validate trade execution latency against SLOs.
- Run canary strategies to ensure risk controls behave as expected.

## 7. Rollback Strategy
- Maintain at least two tagged images in the registry (current and previous stable).
- Use Kubernetes deployments with `maxSurge=1` and `maxUnavailable=0` for safe rollouts.
- Automate rollback via `kubectl rollout undo deployment/tradepulse-core` if health checks fail.
- Document incident details in the runbook and update monitoring dashboards accordingly.

## 8. Continuous Improvement
- Schedule quarterly chaos drills to validate failover paths.
- Capture metrics for deployment frequency, lead time, and mean time to recovery.
- Update this guide whenever infrastructure or tooling changes.

For detailed quality gates and testing workflows, consult the [Quality Assurance Playbook](quality-assurance.md) and [Testing Guide](../TESTING.md).
