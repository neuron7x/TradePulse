# TradePulse EKS Infrastructure

This Terraform configuration provisions a production-ready Amazon EKS foundation for TradePulse. It prioritises
fault tolerance, security, and operational excellence for latency-sensitive trading workloads.

## Features

- Highly-available VPC with private worker subnets and managed NAT.
- Hardened EKS cluster with managed node groups and IRSA enabled by default.
- Opinionated defaults for staging and production with environment-specific `*.tfvars`.
- Optional Cluster Autoscaler deployment configured through Helm with IRSA and conservative scaling flags.
- Tagging strategy aligned with FinOps best practices for cost attribution.

## Directory Layout

```
infra/terraform/eks/
├── environments/
│   ├── production.tfvars
│   └── staging.tfvars
├── main.tf
├── outputs.tf
├── providers.tf
├── variables.tf
└── versions.tf
```

## Usage

1. Initialise the backend and providers:

   ```bash
   terraform -chdir=infra/terraform/eks init
   ```

2. Select the target environment via Terraform workspaces or by passing the relevant `tfvars` file. Workspaces keep
   state isolation between staging and production:

   ```bash
   terraform -chdir=infra/terraform/eks workspace new staging   # first time only
   terraform -chdir=infra/terraform/eks workspace select staging
   terraform -chdir=infra/terraform/eks apply -var-file=environments/staging.tfvars
   ```

   For production:

   ```bash
   terraform -chdir=infra/terraform/eks workspace new production   # first time only
   terraform -chdir=infra/terraform/eks workspace select production
   terraform -chdir=infra/terraform/eks apply -var-file=environments/production.tfvars
   ```

3. Provide AWS credentials via environment variables, IAM roles, or other secure mechanisms supported by Terraform.

4. The Kubernetes and Helm providers depend on the cluster being created in the same apply operation. Terraform waits
   for the control plane to become available before attempting to configure add-ons.

### Managed Kafka (MSK)

Set the optional `msk_config` object in your environment `tfvars` to provision an AWS MSK cluster with the [modules/msk](../modules/msk/README.md) module. The module emits TLS and SASL bootstrap endpoints via the Terraform outputs:

```hcl
msk_config = {
  cluster_name           = "tradepulse-staging-msk"
  number_of_broker_nodes = 3
  broker_subnet_ids      = module.vpc.private_subnets
  client_tls_certificate_authority_arns = [aws_acmpca_certificate_authority.kafka.arn]
  client_sasl_scram_secret_arns         = [aws_secretsmanager_secret.kafka_scram.arn]
}
```

After `terraform apply`, capture the `kafka_bootstrap_brokers_tls` or `kafka_bootstrap_brokers_sasl_scram` outputs and feed them into the application configuration (`configs/live/default.toml` or the `KAFKA_BOOTSTRAP_SERVERS` environment variable).

### Cluster Autoscaler

The Cluster Autoscaler installation can be disabled by setting `enable_cluster_autoscaler = false`. The IAM role for
service accounts (IRSA) configuration ensures the autoscaler operates with least-privilege access scoped to the EKS
cluster only.

### Production Hardening Checklist

- Integrate with your preferred secret management solution (e.g., AWS Secrets Manager or External Secrets Operator).
- Enable control plane logging destinations (CloudWatch, S3, Kinesis) via the EKS module inputs as required.
- Configure AWS WAF and Shield if exposing services publicly through an ingress controller.
- Pin a Terraform backend (e.g., S3 + DynamoDB) before first apply to guarantee state durability.

## State Management

It is strongly recommended to configure a remote backend (S3 with DynamoDB locking or Terraform Cloud) for HA state
management before executing `terraform apply` in collaborative settings. Update `terraform` block with a backend stanza
as appropriate for your organisation.
