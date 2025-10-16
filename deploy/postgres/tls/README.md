# TradePulse Docker TLS Material

This directory holds development TLS assets that enable mutual TLS between the
HA PostgreSQL stack (Bitnami `postgresql-repmgr`) and the TradePulse API when
running via `docker-compose`. The bundled PEM files are *examples only* and
should be replaced with environment-specific certificates before running in
staging or production.

Files:

- `server/root-ca.pem` – root certificate authority used by the PostgreSQL
  primary/replica nodes and Pgpool.
- `server/server.crt` / `server/server.key` – node certificate and key installed
  on the PostgreSQL primary/replica and Pgpool containers.
- `server/client.crt` / `server/client.key` – replication client certificate and
  key trusted by PostgreSQL for intra-cluster checks.
- `client/root-ca.pem`, `client/client.crt`, `client/client.key` – client bundle
  mounted by the TradePulse application container.

Replace the sample files by dropping in your own PEM-encoded materials. When you
update any of the files restart the dependent containers so the new credentials
are loaded.
