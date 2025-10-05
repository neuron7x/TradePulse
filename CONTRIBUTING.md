# CONTRIBUTING

Дякуємо за інтерес до **TradePulse**. Нижче описані правила й процеси, які дозволяють рухатись швидко і безпечно.

## Архітектурна рамка
- **Контракти-перші**: `.proto` у `libs/proto/` — єдине джерело істини для форматів даних і RPC.
- **Фрактальні юніти (FPM-A)**: домени в `domains/<domain>/<fu>/` з чіткою структурою: `src/` (`core`, `ports`, `adapters`), `tests/`, опційно `api/`.
- **Технологічний стек**: Go (сервери/обчислення), Python (execution loop, аналітика), Next.js (дашборд), Prometheus (метрики), Docker Compose.

## Передумови
- **Python 3.11+**, **Go 1.22+**, **Node 18+**, **Docker / Docker Compose**.
- Локально:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt -r requirements-dev.txt
  npm -v && node -v
  ```

## Типовий робочий цикл
1. **Створіть гілку**: `feat/<scope>-<short-desc>` або `fix/<scope>-<short-desc>`.
2. **Оновіть/додайте контракти** у `libs/proto/` (якщо потрібно) → `buf lint` → `buf generate`.
3. **Реалізуйте логіку** в окремому FU (`domains/...`) з поділом на `core/ports/adapters`.
4. **Тести**: розміщуйте в `domains/.../tests/`. Запускайте через `scripts/test.sh`.
5. **Якість коду**: `make fpma-check` (цикломатика) + `scripts/lint.sh` (lint).
6. **PR** з описом WHAT/WHY, посиланнями на issue/дизайн. Див. чекліст нижче.

## Стандарти
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) — приклад: `feat(vpin): stream VPIN calculator`.
- **Гілки**: `feat/*`, `fix/*`, `chore/*`, `docs/*`, `refactor/*`.
- **Стиль коду**:
  - Python: `ruff` (налаштовано в `requirements-dev.txt`), форматування PEP8.
  - Go: `go fmt`, перевірка `go vet` (де доречно).
  - TypeScript: дотримуйтесь прийнятого стилю Next.js; ESLint-правила додамо окремо.
- **API стабільність**: зміни в `.proto` відображайте як **semver** у `CHANGELOG.md`.

## Чекліст Pull Request
- [ ] Описано проблему та рішення (скріншоти/UI — за наявності).
- [ ] Оновлено/додано тести, пройдені локально `scripts/test.sh`.
- [ ] `make fpma-graph` + завантажено `tools/dep_graph.dot` (у PR як артефакт/зміни).
- [ ] `make fpma-check` → жодних порушень порогу.
- [ ] Оновлено `CHANGELOG.md` (секція **Unreleased**).
- [ ] Дотримано `CODE_OF_CONDUCT.md`.

## Як запустити локально
```bash
# сервісні метрики/залежності
docker compose up -d

# генерація кодів з protobuf (за потреби)
scripts/gen-proto.sh

# дашборд (Next.js)
cd apps/web && npm install && npm run dev
```

## Ліцензія та патенти
Проєкт ліцензовано за **MIT** (див. `LICENSE`). Участь у внеску означає згоду з умовами ліцензії.

## Контакти
Порушення Кодексу поведінки/складні ситуації: **security@localhost** (замініть на бойову адресу після розгортання).
