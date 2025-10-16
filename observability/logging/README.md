# TradePulse Log Shipping

This directory contains configuration required to forward structured TradePulse
logs into the Elastic Stack when running via `docker-compose`.

## Components

* **Filebeat** autodiscovers containers labelled with
  `co.elastic.logs/enabled=true` and streams their JSON logs to Logstash.
* **Logstash** normalises the payload, moves TradePulse metadata to stable
  fields, and writes the events to Elasticsearch.
* **Elasticsearch** stores the log indices (`tradepulse-logs-*`).
* **Kibana** exposes the data for analysis and dashboarding.

## Usage

```bash
docker compose up tradepulse prometheus elasticsearch logstash kibana filebeat
```

The default pipeline expects the application to emit JSON logs to stdout (the
existing `core.utils.logging` module already provides that). Kibana will be
available on <http://localhost:5601> with an index pattern of
`tradepulse-logs-*`.

## Customisation

* Adjust `service.name` and `environment` fields in `filebeat.docker.yml` to
  align with your deployment naming.
* Extend `logstash.conf` to enrich the payload or route to additional
  destinations (e.g. S3, Kafka).

