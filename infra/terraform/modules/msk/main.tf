locals {
  log_group_name        = coalesce(var.cloudwatch_logs_log_group, "/aws/msk/${var.cluster_name}/brokers")
  server_properties_map = length(var.configuration_properties) > 0 ? var.configuration_properties : {
    "auto.create.topics.enable" = "false"
  }
  server_properties = join("\n", [for key in sort(keys(local.server_properties_map)) : "${key}=${local.server_properties_map[key]}"])
  merged_tags       = var.tags
}

resource "aws_cloudwatch_log_group" "this" {
  count             = var.cloudwatch_logs_enabled && var.cloudwatch_logs_log_group == null ? 1 : 0
  name              = local.log_group_name
  retention_in_days = 30
  kms_key_id        = var.encryption_at_rest_kms_key_arn
  tags              = local.merged_tags
}

resource "aws_msk_configuration" "this" {
  name            = "${var.cluster_name}-config"
  kafka_versions  = [var.kafka_version]
  server_properties = trimspace(local.server_properties)
}

resource "aws_msk_cluster" "this" {
  cluster_name           = var.cluster_name
  kafka_version          = var.kafka_version
  number_of_broker_nodes = var.number_of_broker_nodes
  enhanced_monitoring    = var.enhanced_monitoring

  broker_node_group_info {
    client_subnets = var.broker_subnet_ids
    instance_type  = var.instance_type
    security_groups = length(var.security_group_ids) > 0 ? var.security_group_ids : null

    storage_info {
      ebs_storage_info {
        volume_size = 1000
      }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.this.arn
    revision = aws_msk_configuration.this.latest_revision
  }

  encryption_info {
    encryption_at_rest_kms_key_arn = var.encryption_at_rest_kms_key_arn
    encryption_in_transit {
      client_broker = var.encryption_in_transit_client_broker
      in_cluster    = true
    }
  }

  dynamic "client_authentication" {
    for_each = length(var.client_tls_certificate_authority_arns) > 0 || length(var.client_sasl_scram_secret_arns) > 0 ? [1] : []
    content {
      dynamic "tls" {
        for_each = length(var.client_tls_certificate_authority_arns) > 0 ? [1] : []
        content {
          certificate_authority_arns = var.client_tls_certificate_authority_arns
        }
      }
      dynamic "sasl" {
        for_each = length(var.client_sasl_scram_secret_arns) > 0 ? [1] : []
        content {
          scram {
            secret_arn_list = var.client_sasl_scram_secret_arns
          }
        }
      }
    }
  }

  logging_info {
    broker_logs {
      dynamic "cloudwatch_logs" {
        for_each = var.cloudwatch_logs_enabled ? [1] : []
        content {
          enabled   = true
          log_group = local.log_group_name
        }
      }
    }
  }

  dynamic "open_monitoring" {
    for_each = var.open_monitoring_prometheus ? [1] : []
    content {
      prometheus {
        jmx_exporter {
          enabled_in_broker = true
        }
        node_exporter {
          enabled_in_broker = true
        }
      }
    }
  }

  tags = local.merged_tags
}
