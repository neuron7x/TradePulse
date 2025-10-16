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

variable "db_engine_version" {
  description = "Aurora PostgreSQL engine version for the managed cluster."
  type        = string
  default     = "15.4"
}

variable "db_instance_class" {
  description = "Instance class to use for Aurora PostgreSQL instances."
  type        = string
  default     = "db.r6g.large"
}

variable "db_reader_instance_count" {
  description = "Number of reader instances to provision alongside the writer."
  type        = number
  default     = 1
}

variable "db_admin_username" {
  description = "Admin username for the Aurora PostgreSQL cluster."
  type        = string
  default     = "tradepulse_admin"
}

variable "db_admin_password" {
  description = "Admin password for the Aurora PostgreSQL cluster."
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.db_admin_password) >= 12
    error_message = "db_admin_password must be at least 12 characters long."
  }
}

variable "db_name" {
  description = "Default database name to create in the Aurora PostgreSQL cluster."
  type        = string
  default     = "tradepulse"
}

variable "db_backup_retention_period" {
  description = "Number of days to retain automated backups."
  type        = number
  default     = 7
}

variable "db_backup_window" {
  description = "Preferred backup window in UTC (format hh24:mi-hh24:mi)."
  type        = string
  default     = "03:00-05:00"
}

variable "db_maintenance_window" {
  description = "Preferred maintenance window in UTC (format ddd:hh24:mi-ddd:hh24:mi)."
  type        = string
  default     = "sun:06:00-sun:07:00"
}

variable "db_deletion_protection" {
  description = "Enable deletion protection on the Aurora PostgreSQL cluster."
  type        = bool
  default     = true
}

variable "db_skip_final_snapshot" {
  description = "Skip the final snapshot when destroying the cluster (not recommended for production)."
  type        = bool
  default     = false
}

variable "db_performance_insights_enabled" {
  description = "Enable Amazon RDS Performance Insights for the PostgreSQL instances."
  type        = bool
  default     = true
}

variable "db_performance_insights_retention_period" {
  description = "Retention period (in days) for Performance Insights metrics."
  type        = number
  default     = 7
}

variable "db_max_connections" {
  description = "Upper bound for allowed PostgreSQL connections (applied via parameter group)."
  type        = number
  default     = 500
}
