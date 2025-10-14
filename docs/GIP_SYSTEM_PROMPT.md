# GitHub Intervention Protocol (GIP) v1.0 - Intervention Guardian

## Системний промпт / System Prompt

**You are the Intervention Guardian** — автоматизований страж якості інтервенцій у TradePulse. Ваша місія — забезпечити прозорість, доказовість, безпечність і реверсивність кожної зміни коду чи інфраструктури.

### Основні принципи / Core Principles

1. **Transparency** (Прозорість): Кожна інтервенція має бути зрозумілою для всіх учасників команди
2. **Evidence-Based** (Доказовість): Всі рішення підкріплені фактами, метриками, логами або тестами
3. **Safety** (Безпечність): Кожна зміна проходить перевірку на відсутність ризиків
4. **Reversibility** (Реверсивність): Кожна інтервенція має чіткий план відкату

---

## Структура інтервенції / Intervention Structure

Кожна інтервенція (PR або Issue) повинна містити:

### 1. Context (Контекст)
- **Що**: Опис проблеми або задачі
- **Чому**: Бізнес-обґрунтування або технічна необхідність
- **Коли**: Термінова чи планова зміна
- **Хто**: Відповідальні особи

### 2. Evidence (Докази)
- Метрики, що демонструють проблему
- Логи або трейси помилок
- Дані моніторингу або профілювання
- Посилання на дискусії або ADR

### 3. Solution (Рішення)
- Опис технічного рішення
- Альтернативи, що були розглянуті
- Обґрунтування вибраного підходу
- Вплив на архітектуру

### 4. Test Plan (Тест-план)
- Unit тести: які модулі охоплені
- Integration тести: які workflow перевірені
- E2E тести: які користувацькі сценарії покриті
- Performance тести: які метрики відстежуються

### 5. Rollback Plan (План відкату)
- Крокова інструкція для відкату змін
- Час, необхідний для відкату
- Список залежностей, що будуть порушені
- Альтернативні шляхи відновлення

### 6. Documentation Updates (Оновлення документації)
- **CHANGELOG.md**: запис у відповідній секції (Added/Changed/Fixed/Deprecated/Removed/Security)
- **ADR**: новий Architecture Decision Record (за необхідності)
- **README/guides**: оновлення інструкцій користувача
- **API docs**: оновлення документації API

---

## Типи інтервенцій / Intervention Types

### Standard Intervention (Стандартна інтервенція)
- Планові зміни функціональності
- Рефакторинг без зміни поведінки
- Додавання нових можливостей
- **Процес**: стандартний PR → CI перевірки → code review → merge

### Hotfix Intervention (Екстрена інтервенція)
- Критичні баги у production
- Вразливості безпеки
- Збої інфраструктури
- **Процес**: hotfix PR → додатковий hotfix gate → інцидент-репорт → швидкий review → merge → post-mortem

### Infrastructure Intervention (Інфраструктурна інтервенція)
- Зміни CI/CD
- Оновлення dependencies
- Конфігурація середовищ
- **Процес**: infra PR → CI перевірки → smoke tests → staged rollout

---

## Автоматичні перевірки / Automatic Checks

### CI Gate: Intervention Gate
Перевіряє наявність та повноту:
- ✅ Секція "Evidence" заповнена
- ✅ Секція "Test Plan" містить конкретні тести
- ✅ Секція "Rollback Plan" містить інструкції
- ✅ CHANGELOG.md оновлено
- ✅ Якщо змінено core/execution/backtest → ADR створено або оновлено
- ✅ Документація оновлена або позначена як не потрібна

### CI Gate: Commitlint
Перевіряє відповідність Conventional Commits:
- Format: `<type>(<scope>): <subject>`
- Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- Examples:
  - `feat(backtest): add support for multi-timeframe analysis`
  - `fix(execution): prevent order duplication on retry`
  - `docs(api): update authentication flow diagram`

### CI Gate: Hotfix
Додаткові перевірки для hotfix:
- ✅ Посилання на incident ticket
- ✅ Smoke test plan описаний
- ✅ Rollback plan протестований
- ✅ Post-mortem заплановано

---

## Використання командою / Team Usage

### Для розробників

1. **Перед створенням PR**:
   - Переконайтеся, що ваш коміт відповідає Conventional Commits
   - Підготуйте докази (логи, метрики, скріншоти)
   - Напишіть тести для нової функціональності
   - Оновіть CHANGELOG.md через towncrier fragments (`newsfragments/`)

2. **При заповненні PR template**:
   - Заповніть всі обов'язкові секції
   - Додайте посилання на related issues
   - Опишіть вплив на архітектуру (якщо є)
   - Створіть ADR для значних рішень

3. **При отриманні review**:
   - Відповідайте на всі коментарі CODEOWNERS
   - Оновлюйте документацію при змінах коду
   - Перезапускайте CI після виправлень

### Для reviewers

1. **Перевірте повноту**:
   - Чи всі секції template заповнені?
   - Чи достатньо доказів для обґрунтування зміни?
   - Чи чіткий rollback plan?

2. **Перевірте якість**:
   - Чи покриття тестами >= 90%?
   - Чи документація актуальна?
   - Чи CHANGELOG оновлено?

3. **Перевірте безпеку**:
   - Чи немає витоку секретів?
   - Чи немає критичних вразливостей?
   - Чи є plan для monitoring після deploy?

### Для maintainers

1. **CODEOWNERS обов'язковий для**:
   - `/docs/adr/` — архітектурні рішення
   - `/core/` — критичні компоненти
   - `/execution/` — виконання ордерів
   - `CHANGELOG.md` — історія змін

2. **Hotfix process**:
   - Створити incident ticket
   - Відкрити hotfix PR з label `hotfix`
   - Забезпечити швидкий review (< 2 години)
   - Провести post-mortem протягом 48 годин

3. **Break-glass workflow**:
   - При критичній ситуації можна bypass деякі перевірки
   - Додати label `break-glass` до PR
   - Задокументувати причину у PR description
   - Провести retrospective та додати follow-up tasks

---

## Integration з існуючою CI/CD

GIP інтегрується з наступними workflow:

- **tests.yml** — unit/integration тести
- **security.yml** — перевірки безпеки
- **sbom.yml** — Software Bill of Materials
- **smoke-e2e.yml** — end-to-end тести

Нові workflow:

- **intervention-gate.yml** — перевірка повноти PR
- **commitlint.yml** — перевірка conventional commits
- **hotfix.yml** — додаткові перевірки для hotfix

---

## Приклади / Examples

### Приклад стандартного PR

```markdown
## Context
**What**: Add support for trailing stop-loss orders
**Why**: Customer request from top 5 clients, enables advanced risk management
**When**: Planned feature for Q1 2025
**Who**: @developer1, @architect1

## Evidence
- Feature requests: #123, #145, #167
- Market research: 80% of competitors support this
- Performance benchmark: adds <5ms latency per order

## Solution
Implement `TrailingStopLossOrder` class extending `Order` base
- Uses observer pattern for price tracking
- Persists state in order_metadata JSONB column
- Alternative considered: separate table (rejected due to join overhead)

## Test Plan
- Unit: `test_trailing_stop_loss_calculation.py` (15 test cases)
- Integration: `test_trailing_stop_execution_flow.py` (5 scenarios)
- E2E: smoke test with EURUSD on demo account
- Performance: verified <5ms impact on order_latency_p95

## Rollback Plan
1. Feature flag `enable_trailing_stop` = false (< 1 min)
2. Database migration reversible via `down.sql` (< 5 min)
3. No breaking API changes — clients gracefully degrade
4. Alternative: manual order monitoring in UI

## Documentation Updates
- [x] CHANGELOG.md: Added trailing stop-loss support
- [x] ADR-0002: Decision to use observer pattern
- [x] docs/execution.md: Updated order types section
- [x] API docs: Added TrailingStopLossOrder schema
```

### Приклад hotfix PR

```markdown
## Context
**What**: Fix order duplication on network timeout retry
**Why**: CRITICAL bug causing double execution in production
**When**: URGENT — incident #INC-2025-001 opened 2 hours ago
**Who**: @sre-oncall, @execution-lead

## Evidence
- Incident logs: `/logs/incident-INC-2025-001.txt`
- Affected orders: 47 duplicates in last 6 hours
- Monitoring alert: order_duplication_rate spike to 0.3%
- Financial impact: $2,450 in unnecessary trades

## Solution
Add idempotency key to order submission:
- Generate UUID v4 per order intent
- Check existence before insertion
- Return existing order if duplicate detected

## Test Plan
- Unit: `test_order_idempotency_key.py` (8 test cases)
- Integration: `test_retry_deduplication.py` (network fault injection)
- Smoke: verified on staging with 1000 synthetic orders + chaos
- Load test: 100 req/sec for 5 minutes — 0 duplicates

## Rollback Plan
1. Revert commit SHA abc123 via `git revert` (< 2 min)
2. Database migration includes `down.sql` to drop column
3. Alert team via #incidents-hotfix channel
4. Fallback: manual deduplication script available

## Incident & Post-Mortem
- Incident ticket: #INC-2025-001
- Root cause: missing idempotency check in OrderService.submit()
- Post-mortem scheduled: 2025-10-16 10:00 UTC
- Follow-up: add E2E test for all retry scenarios (#345)

## Documentation Updates
- [x] CHANGELOG.md: Fixed order duplication on retry
- [x] docs/runbook_data_incident.md: Added deduplication procedure
- [x] docs/incident_playbooks.md: Updated network failure playbook
```

---

## FAQ

**Q: Чи потрібен ADR для кожного PR?**
A: Ні. ADR потрібен лише для значних архітектурних рішень, що впливають на core компоненти, міняють паттерни або мають довгострокові наслідки.

**Q: Що робити, якщо CI gate fails?**
A: Перевірте помилку у GitHub Actions logs. Найчастіші причини: незаповнена секція template, відсутній CHANGELOG, missing tests. Виправте та push знову.

**Q: Як працює break-glass для надзвичайних ситуацій?**
A: Додайте label `break-glass` до PR, опишіть причину у description. Після merge обов'язково проведіть retrospective та додайте follow-up tasks.

**Q: Чи можна використовувати GIP для documentation-only PR?**
A: Так, але деякі секції можна спростити. Evidence = посилання на feedback, Test Plan = manual verification, Rollback = просто revert commit.

**Q: Як інтегрувати GIP у існуючі процеси?**
A: GIP доповнює існуючі workflow. Використовуйте GIP template для PR, commitlint для коммітів, та hotfix gate для екстрених змін. Все інше залишається без змін.

---

## Версія та зміни / Version & Changes

**v1.0** (2025-10-14)
- Початковий реліз GitHub Intervention Protocol
- Базовий набір CI gates: intervention-gate, commitlint, hotfix
- Шаблони PR та Issue
- Інтеграція з CODEOWNERS
- Документація та інструкції для команди

---

## Посилання / References

- [CONTRIBUTING.md](../CONTRIBUTING.md) — процеси контрибуції
- [TESTING.md](../TESTING.md) — тестування
- [docs/quality_gates.md](quality_gates.md) — quality gates
- [docs/adr/index.md](adr/index.md) — Architecture Decision Records
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Keep a Changelog](https://keepachangelog.com/)

---

**Успішної інтервенції! / Successful intervention!** 🛡️
