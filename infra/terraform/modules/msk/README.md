# MSK Module

This module provisions an AWS Managed Streaming for Apache Kafka (MSK) cluster with TLS and optional SASL/SCRAM authentication. It creates a managed configuration object, brokers, CloudWatch logging, and Open Monitoring integration so that TradePulse services can consume secure Kafka endpoints.

## Inputs

| Name | Description | Type | Default |
| ---- | ----------- | ---- | ------- |
| `cluster_name` | Name for the MSK cluster. | `string` | n/a |
| `kafka_version` | Kafka version to deploy. | `string` | n/a |
| `number_of_broker_nodes` | Number of broker nodes (>=2). | `number` | n/a |
| `instance_type` | Broker instance type. | `string` | n/a |
| `broker_subnet_ids` | Private subnet IDs for brokers. | `list(string)` | n/a |
| `security_group_ids` | Security groups to attach to brokers. | `list(string)` | `[]` |
| `configuration_properties` | Map of additional server properties merged with safe defaults. | `map(string)` | `{}` |
| `encryption_in_transit_client_broker` | MSK client-broker encryption mode (`TLS`, `TLS_PLAINTEXT`, or `PLAINTEXT`). | `string` | `"TLS"` |
| `encryption_at_rest_kms_key_arn` | Optional KMS key for volume encryption. | `string` | `null` |
| `client_tls_certificate_authority_arns` | ACM PCA CA ARNs enabling mutual TLS. | `list(string)` | `[]` |
| `client_sasl_scram_secret_arns` | Secrets Manager ARNs that hold SCRAM credentials. | `list(string)` | `[]` |
| `cloudwatch_logs_enabled` | Enable broker log shipping to CloudWatch Logs. | `bool` | `true` |
| `cloudwatch_logs_log_group` | Explicit log group name. When omitted a group is created automatically. | `string` | `null` |
| `open_monitoring_prometheus` | Toggle MSK Open Monitoring/Prometheus exporters. | `bool` | `true` |
| `enhanced_monitoring` | Enhanced monitoring granularity. | `string` | `"PER_TOPIC_PER_PARTITION"` |
| `tags` | Tags applied to all MSK resources. | `map(string)` | `{}` |

## Outputs

| Name | Description |
| ---- | ----------- |
| `cluster_arn` | ARN of the MSK cluster. |
| `bootstrap_brokers` | Bootstrap brokers for PLAINTEXT listeners (empty when disabled). |
| `bootstrap_brokers_tls` | Bootstrap brokers for TLS listeners. |
| `bootstrap_brokers_sasl_scram` | Bootstrap brokers for SASL/SCRAM listeners. |
| `zookeeper_connect_string` | ZooKeeper connection string for legacy tooling. |

## Example

```hcl
module "msk" {
  source = "../modules/msk"

  cluster_name           = "tradepulse-staging-msk"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = 3
  instance_type          = "kafka.m7g.large"
  broker_subnet_ids      = module.vpc.private_subnets
  security_group_ids     = [module.vpc.default_security_group_id]

  client_tls_certificate_authority_arns = [aws_acmpca_certificate_authority.kafka.arn]
  client_sasl_scram_secret_arns         = [aws_secretsmanager_secret.msk_scram.arn]
  tags                                  = local.tags
}
```
