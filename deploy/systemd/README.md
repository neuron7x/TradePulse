# Systemd deployment for the TradePulse application server

This guide describes how to run the TradePulse FastAPI service as a long-lived
systemd unit. The configuration focuses on stability: automatic restarts,
process supervision, and health checks that can be wired into observability
pipelines.

## Prerequisites

1. **Host preparation**
   - Ubuntu 22.04 LTS (or any systemd-based distribution).
   - Python 3.11 runtime and build tools (`python3-venv`, `build-essential`).
   - `git`, `curl`, and `ufw` (firewall configured to allow inbound API traffic).
2. **Dedicated user**
   ```bash
   sudo useradd --system --home /opt/tradepulse --shell /usr/sbin/nologin tradepulse
   ```
3. **Directory layout**
   ```bash
   sudo mkdir -p /opt/tradepulse/{current,venv}
   sudo chown -R tradepulse:tradepulse /opt/tradepulse
   ```

## Install the application

```bash
sudo -u tradepulse git clone https://example.com/TradePulse.git /opt/tradepulse/current
sudo -u tradepulse python3 -m venv /opt/tradepulse/venv
sudo -u tradepulse /opt/tradepulse/venv/bin/pip install --upgrade pip wheel
sudo -u tradepulse /opt/tradepulse/venv/bin/pip install -r /opt/tradepulse/current/requirements.txt
```

Copy your prepared environment file (rendered from `.env.example`) to a secure
location readable by systemd:

```bash
sudo install -o root -g tradepulse -m 640 \
  /opt/tradepulse/current/.env.example /etc/tradepulse/tradepulse.env
```

Update the file with production secrets (API keys, JWT secrets, database
credentials). The unit file consumes the environment without exposing it in the
process list.

## Deploy the unit files

```bash
sudo install -o root -g root -m 755 \
  /opt/tradepulse/current/deploy/systemd/start-tradepulse.sh /usr/local/bin/start-tradepulse.sh
sudo install -o root -g root -m 644 \
  /opt/tradepulse/current/deploy/systemd/tradepulse.service /etc/systemd/system/tradepulse.service
sudo systemctl daemon-reload
sudo systemctl enable --now tradepulse.service
```

The `Restart=always` policy combined with `RestartSec=5s` guarantees the service
returns automatically after a crash. Use the journal for troubleshooting:

```bash
sudo journalctl -u tradepulse.service -f
```

## Health and readiness checks

The FastAPI server exposes metrics and health probes on port `8001` when the
`METRICS_ENABLED` environment variable is set. Confirm the service responds
before attaching upstream load balancers:

```bash
curl -sf http://127.0.0.1:8001/readyz
curl -sf http://127.0.0.1:8001/metrics | head
```

Add a local `systemd` timer or external monitor that polls `/readyz` and
integrates with your alerting backend. For redundancy, deploy multiple
instances behind a load balancer and configure your reverse proxy (Nginx,
Traefik, HAProxy) with circuit breaking and passive health checks.

## Updating the deployment

```bash
sudo systemctl stop tradepulse.service
sudo -u tradepulse git -C /opt/tradepulse/current pull --ff-only
sudo -u tradepulse /opt/tradepulse/venv/bin/pip install -r /opt/tradepulse/current/requirements.txt
sudo systemctl start tradepulse.service
```

Consider a blue/green or canary rollout strategy for production clusters:
deploy a second unit file that points to `/opt/tradepulse/releases/<sha>` and
switch the symlink after the new version passes its readiness checks.
