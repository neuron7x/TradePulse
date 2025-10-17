aws_region  = "us-east-1"
environment = "production"
cluster_name = "tradepulse-production"

node_groups = {
  system = {
    min_size       = 3
    max_size       = 9
    desired_size   = 3
    instance_types = ["m6i.xlarge"]
    capacity_type  = "ON_DEMAND"
    labels = {
      "workload" = "system"
    }
    taints = []
  }
  spot = {
    min_size       = 2
    max_size       = 12
    desired_size   = 4
    instance_types = ["m6i.xlarge", "m6a.xlarge"]
    capacity_type  = "SPOT"
    labels = {
      "workload" = "spot"
    }
    taints = []
  }
}

tags = {
  "CostCenter" = "tradepulse-production"
  "Availability" = "mission-critical"
}

# To provision the production Kafka cluster, uncomment and populate the block
# below with production subnet IDs and secret ARNs. Keep credentials segregated
# per environment.
# msk_config = {
#   cluster_name           = "tradepulse-production-msk"
#   number_of_broker_nodes = 4
#   broker_subnet_ids      = ["subnet-prod-aaaa", "subnet-prod-bbbb", "subnet-prod-cccc"]
#   client_tls_certificate_authority_arns = ["arn:aws:acm-pca:us-east-1:123456789012:certificate-authority/prod"]
#   client_sasl_scram_secret_arns         = ["arn:aws:secretsmanager:us-east-1:123456789012:secret:kafka/prod/scram"]
# }
