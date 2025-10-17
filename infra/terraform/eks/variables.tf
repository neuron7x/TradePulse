variable "aws_region" {
  description = "AWS region to provision EKS resources in."
  type        = string
}

variable "cluster_name" {
  description = "Name of the EKS cluster."
  type        = string
}

variable "environment" {
  description = "Environment name (e.g. staging, production) used for tagging and namespacing."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.40.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "List of CIDR blocks for private subnets."
  type        = list(string)
  default     = ["10.40.0.0/19", "10.40.32.0/19", "10.40.64.0/19"]

  validation {
    condition     = length(var.private_subnet_cidrs) >= 2
    error_message = "private_subnet_cidrs must define at least two subnets."
  }
}

variable "public_subnet_cidrs" {
  description = "List of CIDR blocks for public subnets."
  type        = list(string)
  default     = ["10.40.96.0/20", "10.40.112.0/20", "10.40.128.0/20"]

  validation {
    condition     = length(var.public_subnet_cidrs) >= 2
    error_message = "public_subnet_cidrs must define at least two subnets."
  }
}

variable "availability_zones" {
  description = "Availability zones to spread nodes across. Defaults to the first three in the region."
  type        = list(string)
  default     = []
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS control plane."
  type        = string
  default     = "1.29"
}

variable "node_groups" {
  description = "Managed node group definitions keyed by name."
  type = map(object({
    min_size       = number
    max_size       = number
    desired_size   = number
    instance_types = list(string)
    capacity_type  = string
    labels         = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {}
}

variable "enable_cluster_autoscaler" {
  description = "Whether to install the Kubernetes Cluster Autoscaler via Helm."
  type        = bool
  default     = true
}

variable "tags" {
  description = "Map of additional tags to apply to all resources."
  type        = map(string)
  default     = {}
}

variable "msk_config" {
  description = <<-DESC
    Optional configuration for provisioning an AWS MSK (Managed Streaming for Apache Kafka) cluster alongside EKS.
    When null, no Kafka resources are created. See modules/msk/README.md for supported settings.
  DESC
  type = object({
    cluster_name                         = string
    kafka_version                        = optional(string, "3.6.0")
    number_of_broker_nodes               = optional(number, 3)
    instance_type                        = optional(string, "kafka.m7g.large")
    broker_subnet_ids                    = optional(list(string), [])
    security_group_ids                   = optional(list(string), [])
    configuration_properties             = optional(map(string), {})
    encryption_in_transit_client_broker  = optional(string, "TLS")
    encryption_at_rest_kms_key_arn       = optional(string)
    client_tls_certificate_authority_arns = optional(list(string), [])
    client_sasl_scram_secret_arns        = optional(list(string), [])
    cloudwatch_logs_enabled              = optional(bool, true)
    cloudwatch_logs_log_group            = optional(string)
    open_monitoring_prometheus           = optional(bool, true)
    enhanced_monitoring                  = optional(string, "PER_TOPIC_PER_PARTITION")
  })
  default = null
}
