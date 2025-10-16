output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate authority data"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "oidc_provider_arn" {
  description = "OIDC provider ARN for the cluster"
  value       = module.eks.oidc_provider_arn
}

output "node_group_roles" {
  description = "IAM roles associated with managed node groups"
  value       = { for name, group in module.eks.eks_managed_node_groups : name => group.iam_role_name }
}

output "db_cluster_endpoint" {
  description = "Writer endpoint for the Aurora PostgreSQL cluster"
  value       = aws_rds_cluster.db.endpoint
}

output "db_cluster_reader_endpoint" {
  description = "Reader endpoint for the Aurora PostgreSQL cluster"
  value       = aws_rds_cluster.db.reader_endpoint
}

output "db_cluster_port" {
  description = "Port exposed by the Aurora PostgreSQL cluster"
  value       = aws_rds_cluster.db.port
}

output "db_security_group_id" {
  description = "Security group protecting the Aurora PostgreSQL cluster"
  value       = aws_security_group.db.id
}

output "db_subnet_group_name" {
  description = "Subnet group used by the Aurora PostgreSQL cluster"
  value       = aws_db_subnet_group.db.name
}
