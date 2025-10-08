# Installation Guide

This guide walks you through installing TradePulse for local development, backtesting, and staging deployments.

## 1. Prerequisites
- **Operating System:** Linux or macOS (WSL 2 supported on Windows)
- **Python:** 3.11 or later
- **Go:** 1.21 or later (for execution microservices)
- **Node.js:** 20 LTS (for the optional Next.js dashboard)
- **Git:** 2.40+
- **Make:** 3.81+

## 2. Clone the Repository
```bash
git clone https://github.com/neuron7x/TradePulse.git
cd TradePulse
```

## 3. Set Up Python Environment
1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install runtime and development dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. Verify installation:
   ```bash
   python -c "import yaml, hypothesis; print('Dependencies OK')"
   ```

## 4. Configure Environment Variables
Copy the example environment file and adjust values for your workspace:
```bash
cp configs/.env.example .env
```
Important variables include database URLs, API keys, and feature flags. Do not commit `.env` to version control.

## 5. Optional Services
### Go Execution Engine
```bash
make go-build
./bin/tradepulse-exec --help
```

### Web Dashboard
```bash
cd apps/web
npm install
npm run dev
```
Visit `http://localhost:3000` to access the dashboard (placeholder in current snapshot).

## 6. Smoke Test
Run the CLI backtest pipeline to validate the environment:
```bash
python -m interfaces.cli backtest configs/backtests/sample.yaml
```

## 7. Next Steps
- Review the [Quality Assurance Playbook](quality-assurance.md) for the integration workflow.
- Follow the [Deployment Guide](deployment.md) to promote builds to staging or production environments.
