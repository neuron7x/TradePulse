
# TradePulse 0.1.0

<p align="center">
  <img src="docs/assets/banner.png" alt="TradePulse Banner" />
</p>


## Table of Contents

## Badges
<p>
  <!-- Static truthful badges; dynamic GitHub/Codecov badges can be added after publishing the repo -->
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue.svg"/>
  <img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10-brightgreen.svg"/>
  <img alt="CI" src="https://img.shields.io/badge/ci-configured-informational.svg"/>
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-N%2FA-lightgrey.svg"/>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey.svg"/></a>
</p>

<!--
After pushing to GitHub, replace with dynamic badges (example):
<img src="https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg" alt="CI"/>
<img src="https://codecov.io/gh/<owner>/<repo>/branch/main/graph/badge.svg" alt="Coverage"/>
-->
- [About](#про-проєкт)
- [Badges](#badges)
- [Quick Start](#швидкий-старт)
- [Usage / Examples](#використання-cli-та-api)
- [Configurations](#конфігурації)
- [Results & Screenshots](#приклади-результатів)
- [Architecture](#архітектура-та-пайплайн)
- [Docker](#docker)
- [CI/CD & Quality](#тести-та-якість)
- [Documentation](#документація)
- [Roadmap](#дорожня-карта)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Security](#безпека)
- [License & Author](#ліцензія-та-автор)
**TradePulse** — це фрактально організована платформа досліджень і торгівлі, яка поєднує
аналіз мікроструктури (Kuramoto/Entropy/Hurst/Ricci), фазову детекцію, адаптивних агентів,
бектестинг і виконання ордерів. Архітектура побудована за **FPM-A (Fractal Project Method)**:
проєкт масштабований, модульний, із прозорою якістю коду та CI/CD.

## Ключові можливості
- **Core Indicators**: Kuramoto R, Shannon H та ΔH, Hurst, Ricci curvature (граф рівнів).
- **Advanced Metrics**: Direction Index, ISM, Volume Profile (CVD, imbalance, aggression).
- **Phase Detection**: фазові стани (proto/precognitive/emergent/post) та композит переходів.
- **Adaptive Agents**: стратегії, мутації, бандити (ε-greedy/UCB), пам'ять і ротація.
- **Data Infra**: CSV/WS ingestion (скелет Binance), препроцесинг, стрімінг буфери.
- **Backtest**: walk-forward симулятор (PnL, MaxDD, trades).
- **Execution**: ордери, розмір позиції, ризик.
- **Інтерфейси**: CLI (`tradepulse`) та Streamlit дашборд.
- **Якість/CI**: pre-commit, тест-матриці, CodeQL, SBOM/Trivy, buf guard для protobuf, автолейблинг, реліз-драфтер.

## Швидкий старт
```bash
# Встановлення у віртуальному середовищі
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Швидкий аналіз CSV
tradepulse analyze --csv data/sample.csv --price-col price --window 200

# Бектест
tradepulse backtest --csv data/sample.csv --price-col price --window 200 --fee 0.0005

# Streamlit панель
python -m streamlit run interfaces/dashboard_streamlit.py
```

## Структура репозиторію
```
/core
  indicators/   # Kuramoto, Entropy/ΔH, Hurst, Ricci
  metrics/      # Direction Index, ISM, Volume profile
  agent/        # Strategies, Bandits, Memory, PiAgent
  data/         # Ingestion (CSV/WS), preprocess, streaming
  phase/        # Phase rules and composite transition
/backtest       # Walk-forward engine
/execution      # Orders & risk
/interfaces     # CLI + Streamlit
/tests          # Unit & property tests
/docs           # MkDocs site (mkdocs.yml)
/configs        # YAML конфіги (default.yaml)
/domains        # FPM-A фрактальні юніти (markets/analytics/platform/ui)
/tools          # fpma runner, dep graph
/scripts        # lint, tests, fpma, data sanity
```

## Фазова логіка (скорочено)
- **Proto**: R < 0.4 і ΔH > 0 — хаос; діємо обережно.
- **Precognitive**: 0.4 < R < 0.7 і ΔH ≤ 0 — підготовка.
- **Emergent**: R > 0.75, ΔH < 0, κ̄ < 0 — активна дія.
- **Post-emergent**: R ↓, ΔH ↑ — вихід/repair.

## Рекомендовані пороги/параметри
- Kuramoto emergent threshold **R ∈ [0.75, 0.9]** (налаштовуйте бек-тестом).
- ΔH < 0 для згасання ентропії під прорив.
- κ̄ < 0 для напруги графа.
- Window: **200** (перевірте сіткою параметрів).

## Тести та якість
```bash
bash scripts/lint.sh
pytest -q
make fpma-graph
make fpma-check
```
CI складається з:
- `ci.yml` (матриці лінтів/тестів + кеші), `pre-commit.yml`,
- `codeql-analysis.yml`, `sbom-scan.yml`, `publish-image.yml` (cosign),
- `benchmarks.yml`, `integration.yml`, `data-sanity.yml`,
- `buf.yml`, `gen-drift.yml`, `release-drafter.yml`, `auto-merge.yml`, `pr-labeler.yml`, `todo.yml`.

## Конфігурація
- Загальні параметри → `configs/default.yaml`.
- Секрети/ключі для бірж **не зберігаються у репозиторії**: використовуйте GitHub Secrets або локальні `.env` (поза гітом).

## Ліцензія
MIT. Див. `LICENSE`.

## Дисклеймер
Це дослідницька платформа. Торгівля пов’язана з ризиком. Використовуйте на власну відповідальність.

## Contributing
Прочитайте [CONTRIBUTING.md](CONTRIBUTING.md) перед створенням PR/issue.


## Changelog
Див. [CHANGELOG.md](CHANGELOG.md) для історії змін.