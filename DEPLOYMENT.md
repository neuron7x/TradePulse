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

The repository ships with a lightweight Compose stack that builds the TradePulse container and runs Prometheus for metrics scraping.【F:docker-compose.yml†L1-L12】

1. **Build images** (only required when you change the application code):
   ```bash
   docker compose build tradepulse
   ```
2. **Start the stack** using your `.env` file:
   ```bash
   docker compose --env-file .env up -d
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

## Kubernetes Infrastructure with Terraform and Kustomize

TradePulse now ships with first-class infrastructure code for Amazon EKS alongside Kustomize overlays for staging and production workloads. Use Terraform to provision the cluster(s) and managed node groups, then deploy the application manifests with the provided overlays.

### Provisioning EKS

1. Review the Terraform module under `infra/terraform/eks/`. It provisions a multi-AZ VPC, EKS control plane, managed node groups, and an optional Cluster Autoscaler with IRSA.【F:infra/terraform/eks/main.tf†L1-L203】
2. Initialise Terraform with the desired backend (configure S3/DynamoDB or Terraform Cloud before the first apply):

   ```bash
   terraform -chdir=infra/terraform/eks init
   ```

3. Select the workspace for the target environment and apply the corresponding variables file:

   ```bash
   terraform -chdir=infra/terraform/eks workspace select staging || terraform -chdir=infra/terraform/eks workspace new staging
   terraform -chdir=infra/terraform/eks apply -var-file=environments/staging.tfvars
   ```

   The `production.tfvars` file contains higher-capacity defaults and an additional SPOT node group tailored for mission-critical workloads.【F:infra/terraform/eks/environments/production.tfvars†L1-L27】 Stage-specific sizing lives in `staging.tfvars` for parity testing without the full production footprint.【F:infra/terraform/eks/environments/staging.tfvars†L1-L20】

4. Export AWS credentials securely (e.g., via IAM roles, AWS SSO, or Vault) before applying Terraform. Never embed static keys inside the codebase.

5. The Kubernetes and Helm providers rely on the cluster outputs created in the same plan, so Terraform waits for the control plane to stabilise before installing add-ons like the Cluster Autoscaler.【F:infra/terraform/eks/main.tf†L132-L203】

### Staging and Production Manifests

- Base manifests reside in `deploy/kustomize/base/` and encapsulate shared deployment traits, probes, and service wiring.【F:deploy/kustomize/base/deployment.yaml†L1-L74】【F:deploy/kustomize/base/service.yaml†L1-L17】【F:deploy/kustomize/base/pdb.yaml†L1-L11】
- Environment overlays extend the base with namespace scoping, image tags, scheduling policies, and topology constraints:
  - `deploy/kustomize/overlays/staging` targets the `tradepulse-staging` namespace, preserves mTLS requirements, and spreads pods across zones while staying right-sized for testing.【F:deploy/kustomize/overlays/staging/kustomization.yaml†L1-L14】【F:deploy/kustomize/overlays/staging/patches/deployment-resources.yaml†L1-L36】
  - `deploy/kustomize/overlays/production` introduces a high-priority class, strict topology distribution, and rate limiting tuned for live trading traffic in the `tradepulse-production` namespace.【F:deploy/kustomize/overlays/production/kustomization.yaml†L1-L14】【F:deploy/kustomize/overlays/production/patches/deployment-high-availability.yaml†L1-L43】
- Namespaces are declaratively managed in `deploy/kustomize/namespaces/` and should be applied before or together with the workload overlays.【F:deploy/kustomize/namespaces/staging/namespace.yaml†L1-L8】【F:deploy/kustomize/namespaces/production/namespace.yaml†L1-L8】
- Production overlays install a dedicated `PriorityClass` so the API keeps scheduling headroom even during large-scale cluster events.【F:deploy/kustomize/overlays/production/priorityclass.yaml†L1-L7】

Apply manifests directly with `kubectl` once your kubeconfig contexts are configured:

```bash
kubectl apply -k deploy/kustomize/overlays/staging
kubectl rollout status deployment/tradepulse-api -n tradepulse-staging

kubectl apply -k deploy/kustomize/overlays/production
kubectl rollout status deployment/tradepulse-api -n tradepulse-production
```

Secrets referenced by the deployments (`tradepulse-secrets`, `tradepulse-mtls-client`) must be provisioned outside of source control via your secret management workflow.

### Continuous Delivery Pipeline

The `Deploy TradePulse Environments` GitHub Actions workflow automates validation and rollouts for both environments.【F:.github/workflows/deploy-environments.yml†L1-L139】 Key characteristics:

- On every push to `main`, Terraform formatting/validation and Kustomize builds are executed before any cluster writes.
- Staging deploys automatically after validation. Production deploys once staging succeeds and the protected `production` environment gate is approved inside GitHub.
- Workflow dispatch supports ad-hoc rollouts via GitHub OIDC → AWS IAM federation. Configure short-lived access by supplying the `AWS_REGION`, `AWS_STAGING_ROLE_ARN`, `AWS_STAGING_CLUSTER_NAME`, `AWS_PRODUCTION_ROLE_ARN`, and `AWS_PRODUCTION_CLUSTER_NAME` secrets so the workflow can assume scoped roles and call `aws eks update-kubeconfig` on demand.
- `kubectl diff` runs before each apply to surface configuration drift without failing the run for expected changes.

Before enabling the workflow, create an IAM OIDC identity provider in your cloud account for `token.actions.githubusercontent.com` (or the equivalent endpoint on your platform). Bind environment-specific roles to that provider with trust policies that limit access to this repository, branch, and workflow so that every job receives ephemeral credentials with least privilege.

Document required environment reviewers inside your repository settings so production deployments remain a two-person control.

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
- **CI/CD cluster access** – rely on GitHub's OpenID Connect integration with your cloud provider instead of storing kubeconfigs in secrets. Create dedicated IAM roles with the minimum permissions needed to call `eks:DescribeCluster` (and supporting `sts:AssumeRole`), scope their trust policy to your repository, and allow automatic rotation by issuing fresh tokens per workflow run.

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
