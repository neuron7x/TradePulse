# Deployment Guide

This guide outlines how to run TradePulse with Docker Compose for workstation and staging setups and how to prepare a Helm-based deployment for Kubernetes clusters. It also explains how to manage sensitive configuration and wire health checks into your automation pipelines. For a visual overview of the services involved in each environment, refer to the [architecture diagrams](docs/architecture/system_overview.md) and [feature-store topology](docs/architecture/feature_store.md) before proceeding with environment-specific steps.

## Prerequisites

Before deploying, install the following tools:

- Docker Engine 20.10+ and Docker Compose v2 for container orchestration.
- Access to a container registry where you can push the TradePulse image, if you plan to deploy to Kubernetes.
- `kubectl` configured to talk to your cluster and Helm 3.12+ for chart management.

## Configuration Management

1. Copy the sample environment file and adjust it for your target environment:
   ```bash
   cp .env.example .env
   ```
2. Populate the `.env` file with database settings, exchange API credentials, and provider keys (`POSTGRES_*`, `BINANCE_*`, `COINBASE_*`, `KRAKEN_*`, `ALPHA_VANTAGE_API_KEY`, `IEX_CLOUD_API_KEY`, etc.).【F:.env.example†L19-L64】
3. Set observability and logging configuration so that metrics and logs are exposed on the expected ports (`METRICS_PORT`, `PROMETHEUS_PORT`, `LOG_*`).【F:.env.example†L103-L133】
4. Replace placeholder application secrets (`SECRET_KEY`, `JWT_SECRET`, OAuth tokens, SMTP/Slack/Telegram credentials) with secure values before deploying anywhere outside of local development.【F:.env.example†L135-L183】
5. Never commit populated `.env` files—store them in your secret manager or CI/CD vault and inject them during deployment.

### Secure Database Connectivity

- Provision client TLS material (root CA, client certificate, and private key) for each environment and store them in your
  secret manager. Mount them into the container at runtime or distribute them via Kubernetes Secrets/ConfigMaps as read-only
  files.
- Export the corresponding environment variables that the Hydra experiments expect (`PROD_DB_CA_FILE`, `PROD_DB_CERT_FILE`,
  `PROD_DB_KEY_FILE`, and their staging equivalents). These defaults map to `/etc/tradepulse/db/*.pem` paths in the sample
  configuration so mounting the directory at that location keeps the templates working out of the box.【F:conf/experiment/prod.yaml†L2-L6】【F:conf/experiment/stage.yaml†L2-L6】
- Ensure your database accepts only TLS-authenticated connections and requires the `verify-full` (or stronger) `sslmode` so
  hostname and certificate validation protect against downgrade attacks. The configuration loader now rejects weaker modes,
  causing application startup to fail fast if misconfigured.【F:core/config/postgres.py†L6-L43】

## Docker Compose Deployment

The repository ships with a production-ready Compose stack that builds the TradePulse application container and provisions the full observability toolchain (Prometheus, Alertmanager, Grafana, node-exporter, and cAdvisor).【F:docker-compose.yml†L1-L86】

1. **Compile observability artefacts** whenever dashboards or alert definitions change:
   ```bash
   python -m tools.observability.builder --output-dir observability/generated
   ```
2. **Build images** (required after modifying application code or dependencies):
   ```bash
   docker compose build tradepulse
   ```
3. **Provide alert routing secrets** before booting the stack:
   ```bash
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
   export PAGERDUTY_ROUTING_KEY="pd_routing_key"
   ```
4. **Start the stack** using your `.env` file:
   ```bash
   docker compose --env-file .env up -d
   ```
5. **Verify runtime state**:
   ```bash
   docker compose ps
   docker compose logs -f tradepulse
   docker compose logs -f prometheus
   ```
6. **Stop and remove** the stack when done:
   ```bash
   docker compose down -v
   ```

The Compose configuration exposes Prometheus on `http://localhost:9090`, Grafana on `http://localhost:3000`, Alertmanager on `http://localhost:9093`, and cAdvisor on `http://localhost:8080`. Grafana auto-imports all dashboards generated under `observability/generated/dashboards`, so rebuilding the bundle keeps the UI in sync with Git.【F:docker-compose.yml†L33-L86】【F:deploy/grafana/provisioning/dashboards/dashboards.yaml†L1-L9】

### Compose Health Check

The `tradepulse` container image includes a baked-in health check that hits `/metrics`. When running in Compose, the same command is duplicated in the service definition so failures surface through `docker compose ps` and Prometheus alerting quickly.【F:Dockerfile†L53-L66】【F:docker-compose.yml†L6-L15】

## Kubernetes Deployment with Helm

Although the repository does not include a packaged chart, you can scaffold one under `deploy/helm/tradepulse` using:

```bash
mkdir -p deploy/helm
helm create deploy/helm/tradepulse
```

Update the generated files as follows:

- **`values.yaml`** – set the container image (`image.repository`, `image.tag`), replica count, and service type. Define sensible resource requests/limits so the scheduler can reserve capacity for steady-state traffic.
- **`templates/deployment.yaml`** – mount environment configuration via Secrets/ConfigMaps, open the metrics port, and wire HTTP probes for `/healthz` and `/readyz` to surface pod status to the control plane.

Example patches for the Deployment template:

```yaml
envFrom:
  - secretRef:
      name: tradepulse-secrets
ports:
  - name: metrics
    containerPort: 8001
livenessProbe:
  httpGet:
    path: /metrics
    port: metrics
  initialDelaySeconds: 30
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /metrics
    port: metrics
  initialDelaySeconds: 10
  periodSeconds: 10
```

The repository ships with a production-ready manifest (`deploy/tradepulse-deployment.yaml`) that demonstrates these patterns: the Deployment constrains rollouts to one unavailable pod at a time, advertises resource guarantees, and exposes liveness/readiness/startup probes along with a dedicated metrics port.【F:deploy/tradepulse-deployment.yaml†L1-L74】 A companion `Service` (`deploy/tradepulse-service.yaml`) targets the HTTP and metrics endpoints, while an accompanying PodDisruptionBudget keeps at least two replicas available during voluntary disruptions to preserve availability guarantees.【F:deploy/tradepulse-service.yaml†L1-L24】

Install or upgrade the release once the chart is configured and the container image is available in your registry:

```bash
helm upgrade --install tradepulse deploy/helm/tradepulse \
  --namespace tradepulse \
  --create-namespace \
  --values deploy/helm/tradepulse/values.yaml \
  --set image.tag=$(git rev-parse --short HEAD)
```

## Managing Secrets

- **Docker Compose** – export sensitive values via a `.env` file stored outside of version control. In production automation, load them from your CI/CD vault (`docker compose --env-file /path/to/rendered.env up`).
- **Kubernetes** – create Secrets straight from the same environment file:
  ```bash
  kubectl create secret generic tradepulse-secrets \
    --from-env-file=.env \
    --namespace tradepulse
  ```
  Reference the secret with `envFrom` in your Deployment so that the application receives identical configuration in every environment.
- Rotate API keys and credentials regularly. Update the Secret object and restart the workloads (`kubectl rollout restart deployment tradepulse`).

### Vault/KMS-backed exchange credentials

Define a `secret_backend` block inside each venue credential stanza to source API keys from Vault or a managed KMS instead of long-lived environment variables. The adapter selects the backend implementation while `path` or `path_env` points to the credential bundle. Optional `field_mapping` entries let you translate the payload into the uppercase keys expected by the connectors:

```toml
[[venues]]
name = "binance"
class = "execution.adapters.BinanceRESTConnector"

  [venues.credentials]
  env_prefix = "BINANCE"
  required = ["API_KEY", "API_SECRET"]

    [venues.credentials.secret_backend]
    adapter = "vault"
    path_env = "BINANCE_VAULT_PATH"

      [venues.credentials.secret_backend.field_mapping]
      API_KEY = "api_key"
      API_SECRET = "api_secret"
```

At runtime register backend resolvers on the `LiveTradingRunner`. For example, when HashiCorp Vault agents render JSON secrets locally you can expose a resolver that reads and parses the file, while a cloud KMS adapter might call the vendor SDK and return a decoded dictionary:

```python
import json
from pathlib import Path

from interfaces.live_runner import LiveTradingRunner

def resolve_vault(path: str) -> dict[str, str]:
    return json.loads(Path(path).read_text())

runner = LiveTradingRunner(
    config_path=Path("configs/live/default.toml"),
    secret_backends={"vault": resolve_vault},
)
```

Connectors inheriting from `AuthenticatedRESTExecutionConnector` automatically reuse the resolver for credential rotations so a Vault/KMS rotation triggers a fresh fetch before the next REST call.【F:configs/live/default.toml†L8-L36】【F:interfaces/live_runner.py†L73-L140】【F:interfaces/execution/common.py†L52-L147】

## Health Checks and Observability

- **HTTP probes** – reuse the metrics endpoint for readiness and liveness, or expose a lightweight `/healthz` endpoint that validates downstream dependencies before returning `200`.
- **Prometheus** – keep the scrape configuration aligned with your target service names (`tradepulse:8001` in Docker Compose, `<service-name>:8001` in Kubernetes).【F:deploy/prometheus.yml†L2-L7】
- **Dashboards** – point Grafana or your preferred UI to the Prometheus instance and alert on failed health probes, high error rates, or scrape gaps.

Following these practices keeps deployments reproducible across environments while giving operations teams the hooks they need for automation, alerting, and incident response.

## Release Automation

TradePulse releases are automated through GitHub Actions:

1. **Drafting** – The `Release Drafter` workflow updates a draft release on every push to `main`, grouping merged PRs by labels via `.github/release-drafter.yml`.
2. **Changelog fragments** – Every change must include a Towncrier fragment under `newsfragments/` (see `newsfragments/README.md`). When the release workflow runs it stitches the fragments into `CHANGELOG.md` and generates the GitHub release notes.
3. **SemVer tags only** – Publishing requires annotated tags that follow `vMAJOR.MINOR.PATCH`. The workflow verifies that the tag matches the version recorded in the `VERSION` file before continuing.
4. **Green CI gate** – Releases are blocked unless all checks on the tagged commit have completed successfully. The workflow inspects the commit status and fails fast if required jobs (tests, lint, build) are pending or red.
5. **Signature enforcement** – Tags must carry a cryptographic signature, and build artifacts (wheels and sdists) are signed with Sigstore before they are uploaded to the release and PyPI.

To perform a release:

```bash
# ensure CI is green and fragments exist
# create a signed SemVer tag and push it
 git tag -s vX.Y.Z -m "TradePulse vX.Y.Z"
 git push origin vX.Y.Z
```

GitHub Actions will take over from there, generate the changelog, attach the signed artifacts, and publish to PyPI once the release is approved.
