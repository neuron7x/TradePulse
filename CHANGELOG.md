# CHANGELOG
Ведемо зміни за правилами [Keep a Changelog](https://keepachangelog.com/) та [SemVer](https://semver.org/).

## [Unreleased]
### Security
- **CRITICAL**: Upgraded pip dependency to version >=25.2 to mitigate vulnerability GHSA-4xh5-x5gv-qwph (arbitrary file overwrite via tarfile extraction). Updated requirements.txt, requirements-dev.txt, Dockerfile, and CI workflows (.github/workflows/tests.yml, .github/workflows/security.yml) to enforce minimum pip version 25.2. When pip 25.3+ becomes available, it will be automatically adopted due to the >= constraint.

## [2.1.3] - 2025-10-05
### Added
- Новий конвеєр **CI/CD**: `ci.yml` (матриці лінтів/тестів, concurrency, кеші), `pre-commit.yml`, `auto-merge.yml`, `sbom-scan.yml`, `publish-image.yml` (cosign), `data-sanity.yml`.
- Якість/управління змінами: `benchmarks.yml`, `integration.yml`, `commitlint.yml`, `pr-labeler.yml`, `todo.yml`.
- Контракти: `buf.yml` (lint+breaking), `gen-drift.yml` (+ Makefile `generate`).
### Changed
- Оновлено `.pre-commit-config.yaml` (black, ruff, prettier, buf hooks).
- Адаптовано JS gateway (`domains/platform/gateway`) під lint-джоб.

## [2.1.2] - 2025-10-05
### Added
- Конфіг для pre-commit (`.pre-commit-config.yaml`), pytest (`pytest.ini`).
- Шаблони GitHub: issue/PR, labeler, dependabot, release-drafter.
- Робочі процеси CI: release-drafter, codeql, lint, тест-матриця, docs-build, deploy, infra-check.

### Added
- Розширення скриптів автоматизації (`scripts/*`).
- Шаблони процесів: `.gitattributes`, `CODEOWNERS`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.

### Changed
- Уточнення документації щодо фрактальної декомпозиції (FPM-A).

## [2.1.1] - 2025-10-05
### Added
- Інтегровано професійні проектні артефакти: `.gitattributes`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `CODEOWNERS`.
- Додано `scripts/`: `gen-proto.sh`, `lint.sh`, `test.sh`, `fpma.sh`, `dev-up.sh`, `dev-down.sh`.

### Security
- Уніфікація ліній закінчення файлів через `.gitattributes`.

## [2.1.0] - 2025-10-05
### Added
- Інтеграція **FPM-A**: фрактальні юніти, граф залежностей, метрики цикломатичної складності, CI-гейти.

## [2.0.0] - 2025-10-05
### Added
- Початковий каркас TradePulse: protobuf-контракти, Python/Next.js скелети, інфраструктурні файли.

## [0.1.0] - 2025-10-05
### Added
- Повноцінний каркас проєкту з фрактальною архітектурою (FPM-A), індикаторами (Kuramoto/Entropy/Hurst/Ricci), агентною системою, пайплайнами даних, фазовою логікою, бектестом, CLI та Streamlit-панеллю.
- Професійна документація (README, MkDocs), CI/CD, безпека (CodeQL, SBOM), автолінти й тести.
### Fixed
- Заповнено шаблонні місця у CLI/скриптах; узгоджено версування і конфіги.
