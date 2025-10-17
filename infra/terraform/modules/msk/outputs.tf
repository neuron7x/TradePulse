output "cluster_arn" {
  description = "ARN of the MSK cluster"
  value       = aws_msk_cluster.this.arn
}

output "bootstrap_brokers" {
  description = "PLAINTEXT bootstrap brokers (if enabled)"
  value       = aws_msk_cluster.this.bootstrap_brokers
}

output "bootstrap_brokers_tls" {
  description = "TLS bootstrap brokers"
  value       = aws_msk_cluster.this.bootstrap_brokers_tls
}

output "bootstrap_brokers_sasl_scram" {
  description = "SASL/SCRAM bootstrap brokers"
  value       = aws_msk_cluster.this.bootstrap_brokers_sasl_scram
}

output "zookeeper_connect_string" {
  description = "ZooKeeper connect string (for Kafka < 3.4 compatibility)"
  value       = aws_msk_cluster.this.zookeeper_connect_string
}
