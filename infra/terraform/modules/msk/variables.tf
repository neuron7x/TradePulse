variable "cluster_name" {
  description = "Name of the MSK cluster"
  type        = string
}

variable "kafka_version" {
  description = "Kafka version for the cluster"
  type        = string
}

variable "number_of_broker_nodes" {
  description = "Number of brokers in the cluster"
  type        = number
  validation {
    condition     = var.number_of_broker_nodes >= 2
    error_message = "number_of_broker_nodes must be at least 2 for availability."
  }
}

variable "instance_type" {
  description = "Instance type for MSK brokers"
  type        = string
}

variable "broker_subnet_ids" {
  description = "Subnets for the broker nodes"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security groups applied to broker ENIs"
  type        = list(string)
  default     = []
}

variable "configuration_properties" {
  description = "Additional Kafka configuration overrides"
  type        = map(string)
  default     = {}
}

variable "encryption_in_transit_client_broker" {
  description = "Encryption in transit mode for client connections"
  type        = string
  default     = "TLS"
}

variable "encryption_at_rest_kms_key_arn" {
  description = "Optional CMK ARN for encryption at rest"
  type        = string
  default     = null
}

variable "client_tls_certificate_authority_arns" {
  description = "List of ACM PCA certificate authority ARNs for client TLS auth"
  type        = list(string)
  default     = []
}

variable "client_sasl_scram_secret_arns" {
  description = "List of Secrets Manager secret ARNs providing SASL/SCRAM credentials"
  type        = list(string)
  default     = []
}

variable "cloudwatch_logs_enabled" {
  description = "Whether broker logs should flow to CloudWatch Logs"
  type        = bool
  default     = true
}

variable "cloudwatch_logs_log_group" {
  description = "Name of the CloudWatch Logs group for broker logs"
  type        = string
  default     = null
}

variable "open_monitoring_prometheus" {
  description = "Whether to enable MSK open monitoring with Prometheus"
  type        = bool
  default     = true
}

variable "enhanced_monitoring" {
  description = "MSK enhanced monitoring level"
  type        = string
  default     = "PER_TOPIC_PER_PARTITION"
}

variable "tags" {
  description = "Tags applied to MSK resources"
  type        = map(string)
  default     = {}
}
