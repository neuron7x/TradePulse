aws_region  = "us-east-1"
environment = "staging"
cluster_name = "tradepulse-staging"

node_groups = {
  general = {
    min_size       = 2
    max_size       = 6
    desired_size   = 3
    instance_types = ["m6i.large"]
    capacity_type  = "ON_DEMAND"
    labels = {
      "workload" = "api"
    }
    taints = []
  }
}

tags = {
  "CostCenter" = "tradepulse-staging"
}

# To provision the staging Kafka cluster, uncomment and fill the block below with
# environment-specific subnet IDs and secret ARNs generated for TLS/SASL.
# msk_config = {
#   cluster_name           = "tradepulse-staging-msk"
#   number_of_broker_nodes = 3
#   broker_subnet_ids      = ["subnet-aaaaaaaa", "subnet-bbbbbbbb", "subnet-cccccccc"]
#   client_tls_certificate_authority_arns = ["arn:aws:acm-pca:us-east-1:123456789012:certificate-authority/abcd"]
#   client_sasl_scram_secret_arns         = ["arn:aws:secretsmanager:us-east-1:123456789012:secret:kafka/scram"]
# }
