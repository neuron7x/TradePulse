# Secret Management Runbook

This runbook describes how TradePulse operators provision, rotate, and audit
access credentials across environments.

## Sources of truth

* **Google Secret Manager** – preferred for cloud-deployed services.
* **HashiCorp Vault** – used for on-premise deployments with dynamic database
  credentials.
* **AWS Secrets Manager** – optional backend when deploying to AWS-native
  infrastructure.

Each environment must declare a `SECRET_BACKEND` variable within its deployment
configuration. Supported values are `gcp`, `vault`, and `aws`. The default local
configuration falls back to `.env` files for developer productivity.

## Bootstrap steps

1. Create a service account or IAM role with permission to read the relevant
   secrets path.
2. Store TLS material for Postgres in the secret backend under the key
   `tradepulse/postgres-tls`. The value should be a JSON object containing
   `ca`, `cert`, and `key` fields with PEM encoded content.
3. Create API key secrets for each exchange under
   `tradepulse/exchanges/<venue>/api-key` and `api-secret`.
4. Populate `tradepulse/mlflow` and `tradepulse/wandb` entries with the tracking
   credentials used by experiment pipelines.

## Rotation procedure

1. Update the secret in the backend.
2. Trigger the `tradepulse-secrets-refresh` workflow. The workflow runs
   `scripts/rotate_secrets.py` which syncs credentials into Kubernetes secrets
   and restarts affected deployments.
3. Verify success via the Grafana dashboard `Secrets / Refresh Status`. The
   dashboard reads from the `secret_rotation_status` metric exposed by the
   orchestration service.

## Emergency revocation

* For exchange credentials, revoke keys via the exchange console and remove the
  secret entry. Deployments will fail closed and retry once new credentials are
  provisioned.
* For database credentials managed by Vault, revoke the lease using
  `vault lease revoke <lease-id>` and rotate the root token if compromise is
  suspected.

## Local development tips

* Copy `.env.example` to `.env` and fill in placeholders with throwaway keys.
* When using Vault locally, export `VAULT_ADDR` and `VAULT_TOKEN` before
  launching services.
* The helper `scripts/secrets/download.py` retrieves secrets into `.secrets/`
  for inspection. Ensure the directory remains `.gitignore`d.
