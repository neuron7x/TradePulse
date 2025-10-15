# Deployment Guide

This guide outlines how to run TradePulse with Docker Compose for workstation and staging setups and how to prepare a Helm-based deployment for Kubernetes clusters. It also explains how to manage sensitive configuration and wire health checks into your automation pipelines. For a visual overview of the services involved in each environment, refer to the [architecture diagrams](docs/architecture/system_overview.md) and [feature-store topology](docs/architecture/feature_store.md) before proceeding with environment-specific steps.

## Prerequisites

Before deploying, install the following tools:

- Docker Engine 20.10+ and Docker Compose v2 for container orchestration.
- Access to a container registry where you can push the TradePulse image, if you plan to deploy to Kubernetes.
- `kubectl` configured to talk to your cluster and Helm 3.12+ for chart management.

## Configuration Management

TradePulse reads runtime configuration exclusively from environment variables. The `.env.example` file documents the required keys and safe placeholder values but should not be copied or populated directly. Instead:

1. Mirror the keys that carry sensitive data (`POSTGRES_*`, `BINANCE_*`, `COINBASE_*`, `KRAKEN_*`, `ALPHA_VANTAGE_API_KEY`, `IEX_CLOUD_API_KEY`, etc.) inside your Vault/KMS hierarchy. For Vault KV v2 this typically resembles:
   ```bash
   vault kv put secret/tradepulse/workstation \
     POSTGRES_PASSWORD=s3cr3t \
     BINANCE_API_KEY=... \
     BINANCE_API_SECRET=...
   ```
2. Store non-secret toggles (metrics ports, feature flags, sandbox switches) in ConfigMaps or parameter stores so they can be reviewed and rotated independently of secrets.【F:.env.example†L103-L183】
3. Surface dashboard credentials through the same backend by providing `DASHBOARD_ADMIN_USERNAME`, `DASHBOARD_ADMIN_PASSWORD_HASH`, and cookie parameters. The Streamlit interface automatically resolves these values via the configured `SecretManager` or Vault resolver, falling back to environment variables only when no backend is defined.【F:interfaces/dashboard_streamlit.py†L1-L154】

### Injecting Secrets from Vault/KMS

There are two common ways to inject secrets at runtime:

1. **Sidecar/agent templates** – run a Vault agent or cloud secret-sync sidecar that renders key/value data to an in-memory file (e.g., `/run/secrets/tradepulse.env`). Point Compose (`--env-file`) or your Kubernetes Secret manifest at this path. The agent keeps the file fresh so rotations propagate without restarts.
2. **Ephemeral CLI renders** – for CI/CD pipelines, export the secret bundle just-in-time and feed it directly to the orchestrator:
   ```bash
   vault kv get -mount=secret -field=data -format=json tradepulse/workstation \
     | jq -r 'to_entries[] | "\(.key)=\(.value)"' > /tmp/tradepulse.env
   docker compose --env-file /tmp/tradepulse.env up -d
   shred -u /tmp/tradepulse.env
   ```

For Kubernetes, the rendered file can be uploaded as a Secret (`kubectl create secret generic tradepulse --from-env-file=/tmp/tradepulse.env --dry-run=client -o yaml | kubectl apply -f -`). The application code transparently consumes the same variables both locally and in clusters.

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

The repository ships with a lightweight Compose stack that builds the TradePulse container and runs Prometheus for metrics scraping.【F:docker-compose.yml†L1-L12】

1. **Build images** (only required when you change the application code):
   ```bash
   docker compose build tradepulse
   ```
2. **Start the stack** using a rendered secret bundle from Vault/KMS (for example, a Vault agent template or CLI export saved to `/tmp/tradepulse.env`):
   ```bash
   docker compose --env-file /tmp/tradepulse.env up -d
   ```
3. **Verify runtime state**:
   ```bash
   docker compose ps
   docker compose logs -f tradepulse
   ```
4. **Stop and remove** the stack when done:
   ```bash
   docker compose down -v
   ```

### Compose Health Check

Expose an HTTP health endpoint from the TradePulse service (e.g., `/metrics` or `/healthz` on port 8001) and add the following to the `tradepulse` service to integrate with Compose status reporting:

```yaml
services:
  tradepulse:
    # ...existing settings...
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/metrics"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
```

Prometheus is preconfigured to scrape the TradePulse metrics endpoint on port 8001, so a failing health check will surface quickly in dashboards.【F:deploy/prometheus.yml†L2-L7】

## Kubernetes Deployment with Helm

Although the repository does not include a packaged chart, you can scaffold one under `deploy/helm/tradepulse` using:

```bash
mkdir -p deploy/helm
helm create deploy/helm/tradepulse
```

Update the generated files as follows:

- **`values.yaml`** – set the container image (`image.repository`, `image.tag`), replica count, service type, and resource requests/limits.
- **`templates/deployment.yaml`** – mount environment configuration via Secrets/ConfigMaps and open the metrics port.

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

Install or upgrade the release once the chart is configured and the container image is available in your registry:

```bash
helm upgrade --install tradepulse deploy/helm/tradepulse \
  --namespace tradepulse \
  --create-namespace \
  --values deploy/helm/tradepulse/values.yaml \
  --set image.tag=$(git rev-parse --short HEAD)
```

## Managing Secrets

- **Docker Compose** – render secrets to a temporary `.env` file via your Vault/KMS tooling (`vault kv get -format=json ... | jq ... > /tmp/tradepulse.env`) and pass it to Compose (`docker compose --env-file /tmp/tradepulse.env up`). Clean up the file after startup or mount it as an in-memory volume during CI/CD runs.
- **Kubernetes** – create Secrets straight from the rendered environment bundle so containers receive the same variables regardless of platform:
  ```bash
  kubectl create secret generic tradepulse-secrets \
    --from-env-file=/tmp/tradepulse.env \
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
