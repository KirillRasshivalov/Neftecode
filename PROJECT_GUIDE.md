# Neftecode — полное руководство по проекту

Документ описывает: что требует ТЗ, как устроен каркас, что делает каждый пакет и файл, что уже готово, чего не хватает и куда двигаться дальше.

---

## 1. О чём проект (из ТЗ)

Хакатонное задание: **мультиагентная система (МАС) поддержки решений оператора** для цепочки производства дизеля:

**АВТ → гидроочистка → блендинг**

Изменение режима на одном участке влияет на качество, оборудование и экономику дальше по цепочке. Нужен не «один ML-скрипт», а цикл:

1. Снять снимок процесса (КИП + ЛИМС/ПАК + возраст данных).
2. Проверить полноту и согласованность данных.
3. Оценить качество сейчас / прогноз и риск выхода за спецификацию.
4. Оценить тяжесть режима / риск для оборудования.
5. Сгенерировать несколько допустимых управляющих воздействий.
6. Отбросить нарушения жёстких ограничений.
7. Сравнить варианты и выбрать лучший.
8. Выдать оператору **объяснимую рекомендацию** — или честный **отказ**.

### Жёсткие принципы из ТЗ

| Принцип | Как отражено в коде |
|---|---|
| Качество и tech-лимиты важнее экономики | `HardConstraints` + веса в `ranking_weights` |
| Сера товарного дизеля ≤ 10 мг/кг | `configs/constraints.yaml` → `sulfur_mg_kg_max: 10` |
| Синхронизация только по времени (`date`), не по строке | `data/sync.py` → `merge_asof_on_date` |
| Train/test только temporal split, без shuffle | `data/time_split.py` |
| Уметь сказать «рекомендации нет» | `DataQualityGate` + `refuse` в `OperatorRecommendation` |
| Приоритет факта качества: ЛИМС → ПАК → виртуальный | `freshness.apply_quality_priority` + `constraints.yaml` |
| Телеметрия ~10-минутный шаг | зафиксировано в `configs/assumptions.md` |
| Ответ оператору с объяснением | `OperatorRecommendation` + `explain.build_explanation` |

### Роли в команде

| Роль | Зона ответственности |
|---|---|
| **Бэк (ты)** | data pipeline, `ProcessState`, оркестратор, safety, оптимизатор-каркас, формат рекомендации, CLI/API, stubs, демо |
| **ML-1** | агент качества: прогноз показателей, риск спеки, confidence |
| **ML-2** | агент надёжности: прокси тяжести режима, soft-constraints |

Оркестратор не должен знать, stub внутри или модель: оба реализуют один `Protocol.assess(...)`.

---

## 2. Текущий статус

Сейчас это **рабочий каркас на stubs**. Цикл крутится end-to-end **без реальных CSV**:

- оркестратор, gate, hard constraints, optimizer — работают;
- Quality / Reliability — эвристические stubs;
- `ProcessStateBuilder` — scaffold (midpoints + демо-сера);
- данные хакатона в `data/` ещё не подключены;
- `QualityAgentML` / `ReliabilityAgentML` — скелеты с `NotImplementedError`.

**Архитектура и проводка — да. Бизнес-логика на реальных данных — нет.**

---

## 3. Дерево репозитория

```text
neftecode/
├── README.md                 # краткий старт
├── PROJECT_GUIDE.md          # этот документ
├── pyproject.toml            # пакет, зависимости, entrypoint, pytest
├── requirements.txt          # плоский список зависимостей
├── .gitignore
├── configs/                  # yaml + допущения (как application.yml)
│   ├── tags_whitelist.yaml
│   ├── constraints.yaml
│   └── assumptions.md
├── data/                     # сырые выгрузки (пока пусто, только .gitkeep)
├── artifacts/                # JSON рекомендаций после run_cycle
├── models/                   # артефакты ML (.joblib и т.п.)
├── notebooks/                # исследования ML (пусто)
├── tests/                    # pytest
└── src/neftecode/            # основной код
    ├── __init__.py
    ├── __main__.py
    ├── main.py               # CLI
    ├── domain/               # DTO / контракты
    ├── data/                 # loaders, sync, state, features
    ├── agents/               # Quality / Reliability / Optimizer
    ├── safety/               # hard constraints + data gate
    ├── orchestration/        # use-case цикла
    ├── demo/                 # 4 сценария для жюри
    └── api/                  # optional FastAPI
```

Аналогия Java → Python:

| Пакет | Аналог в Java | Назначение |
|---|---|---|
| `domain/` | DTO / records | общие модели |
| `data/` | Repository | чтение, sync, freshness, state |
| `agents/` | Services | Quality / Reliability / Optimizer |
| `safety/` | Validation | hard constraints, gate |
| `orchestration/` | Use-case | `Orchestrator.run_cycle` |
| `demo/` | Fixtures | сценарии для демо |
| `api/` | Controller | тонкий HTTP |
| `configs/` | application.yml | whitelist, constraints, assumptions |

---

## 4. Поток одного цикла решения

```text
CLI / API
    │
    ▼
Orchestrator.run_cycle(timestamp, scenario?)
    │
    ├─ ProcessStateBuilder.build(...)     → ProcessState
    ├─ DataQualityGate.check(state)       → ok / refuse
    ├─ QualityAgent.assess(state)         → baseline quality
    ├─ ReliabilityAgent.assess(state)     → baseline reliability
    │
    ├─ [если gate fail] → OperatorRecommendation(refuse=True) → artifacts/*.json
    │
    └─ OptimizerAgent.propose(state)
           │
           ├─ сетка кандидатов из whitelist.deltas
           ├─ для каждого: Quality + Reliability + HardConstraints
           ├─ rank (ниже score лучше)
           └─ best + 1–2 alternatives
                │
                ▼
         OperatorRecommendation + explanation → artifacts/*.json
```

---

## 5. Описание каждого пакета и файла

### 5.1 Корень проекта

| Файл | Что делает |
|---|---|
| `README.md` | Краткая инструкция: установка, CLI, структура, handoff для ML |
| `PROJECT_GUIDE.md` | Полное руководство (этот файл) |
| `pyproject.toml` | Имя пакета `neftecode`, Python ≥3.11, deps, extras `[api,ml,dev]`, script `neftecode`, pytest config, `src`-layout |
| `requirements.txt` | Тот же стек плоским списком (удобно для `pip install -r`) |
| `.gitignore` | venv, кэши, IDE; `data/*`, `artifacts/*`, `models/*` (кроме `.gitkeep` / `models/README.md`) |

### 5.2 `configs/` — конфигурация и допущения

| Файл | Что делает |
|---|---|
| `tags_whitelist.yaml` | Список **управляемых** тегов (сейчас `PLACEHOLDER_*`), диапазоны, `deltas` для сетки кандидатов, пустые `state_tags` под телеметрию. Диапазоны с `assumption: true` — модельные, не заводские лимиты |
| `constraints.yaml` | Hard: сера ≤10, сумма долей бленда =1; пороги stale ЛИМС/ПАК; приоритет источников; веса ранжирования; тексты отказов |
| `assumptions.md` | Явные допущения для жюри: whitelist, качество, надёжность, sync, train/test |

### 5.3 `data/` (папка репозитория) — сырые данные

Сейчас только `.gitkeep`. Сюда кладутся выгрузки хакатона:

- `avt_tags.csv`
- `242000_tags.csv`
- `lims.xlsx` / `pak.xlsx`
- `tags_dictionary.xlsx`

Имена путей заданы в `src/neftecode/data/loaders.py` → `DataPaths`.

### 5.4 `artifacts/`

JSON-ответы после каждого `run_cycle`. Имя: `recommendation_YYYYMMDDTHHMMSS.json`.  
В репозитории лежат примеры прогонов `normal` и `stale_data`.

### 5.5 `models/`

Место для обученных артефактов ML (`quality_sulfur.joblib`, `reliability_proxy.joblib` и т.п.). Оркестратор не трогать — только адаптеры агентов.

### 5.6 `notebooks/`

Для исследований ML. В прод-цикл не входят. Сейчас пусто.

### 5.7 `tests/`

| Файл | Что проверяет |
|---|---|
| `test_orchestrator_stub.py` | `normal` → есть рекомендация; `stale_data` → `refuse=True` |
| `test_constraints.py` | сера >10 отклоняется; тег вне диапазона отклоняется |
| `test_sync.py` | `merge_asof` идёт по времени; `time_based_split` без shuffle |

### 5.8 `src/neftecode/` — точка входа пакета

| Файл | Что делает |
|---|---|
| `__init__.py` | Версия пакета `0.1.0`, краткое описание |
| `__main__.py` | Позволяет `python -m neftecode ...` вызвать `main()` |
| `main.py` | CLI на `argparse`: команды `scenarios` и `run` (`--scenario` / `--timestamp` / `--pretty`). Собирает `Orchestrator`, печатает JSON |

---

### 5.9 Пакет `domain/` — контракты (DTO)

Общие pydantic-модели. Их менять осторожно: это API между бэком и ML.

| Файл | Что делает |
|---|---|
| `__init__.py` | Реэкспорт публичных DTO |
| `state.py` | `TagValue`, `QualityReading` (метрика + source + age), `ProcessState` — снимок завода на момент решения: `kip`, `quality`, `controllable`, `data_flags`, `notes` |
| `actions.py` | `ControlAction` — словарь `tag → новое значение`, `label`, метод `is_noop()` |
| `agent_results.py` | `QualityAssessment`, `ReliabilityAssessment`, `ScoredScenario` (action + оценки + feasible/score) |
| `recommendation.py` | `OperatorRecommendation` — финальный ответ оператору (ТЗ §5): refuse, действие, эффект, constraints, confidence, explanation, alternatives, audit |

---

### 5.10 Пакет `data/` — данные и снимок состояния

| Файл | Что делает | Статус |
|---|---|---|
| `__init__.py` | Реэкспорт API слоя | готово |
| `config.py` | `project_root()`, загрузка `tags_whitelist.yaml` и `constraints.yaml` | готово |
| `loaders.py` | `DataPaths` (пути к CSV/XLSX), `TelemetryStore` — lazy-чтение AVT/гидро CSV; если файла нет — пустой DataFrame со столбцом `date` | каркас; ЛИМС/ПАК ещё не читаются |
| `sync.py` | `merge_asof_on_date` — выравнивание по `date` (`pd.merge_asof`, direction=backward) | готово |
| `freshness.py` | `compute_age_minutes`, `apply_quality_priority` (ЛИМС→ПАК→virtual), хелпер datetime | готово, почти не используется state_builder’ом |
| `tag_dictionary.py` | `TagDictionary` — чтение справочника тегов из Excel (эвристика колонок) | каркас; файла нет |
| `feature_builder.py` | `FeatureBuilder.build_frame` — aligned таблица для ML (+ asof targets); TODO: лаги, age | задел под ML |
| `time_split.py` | `time_based_split(train_end, val_end)` — только по времени | готово |
| `state_builder.py` | Собирает `ProcessState` на timestamp. **Сейчас scaffold**: midpoints из whitelist, демо-сера по scenario | **нужна реальная реализация** |

Поведение `ProcessStateBuilder` по сценариям:

| scenario | sulfur | age_minutes | эффект |
|---|---|---|---|
| `normal` / `full_cycle` / default | 8.0 | 30 | gate OK, рекомендация возможна |
| `risk_sulfur` | 11.5 | 30 | высокий quality risk / давление на серу |
| `stale_data` | 8.0 | 10000 | gate отказывает (stale LIMS) |

---

### 5.11 Пакет `agents/` — агенты

| Файл | Что делает | Статус |
|---|---|---|
| `__init__.py` | Экспорт stub-агентов и Optimizer | готово |
| `base.py` | `Protocol`: `QualityAgent.assess`, `ReliabilityAgent.assess` | контракт для ML |
| `quality.py` | `QualityAgentStub` — эвристика по сере (+ чувствительность к `PLACEHOLDER_HDT_TEMP`); `QualityAgentML` — скелет | stub работает; ML пустой |
| `reliability.py` | `ReliabilityAgentStub` — грубый risk по «краям» TEMP; `ReliabilityAgentML` — скелет | stub работает; ML пустой |
| `optimizer.py` | Сетка кандидатов из `deltas` whitelist → оценка Q/R → hard filter → weighted score (ниже лучше) → список feasible | каркас готов; throughput/energy пока 0.0 |

Ранжирование (из `constraints.yaml`):

- `quality_risk` × 100
- `equipment_risk` × 20
- `throughput` × 10
- `energy_or_cost_proxy` × 5

Hard constraints **никогда** не торгуются за score.

---

### 5.12 Пакет `safety/` — безопасность

| Файл | Что делает |
|---|---|
| `__init__.py` | Экспорт gate и constraints |
| `constraints.py` | `HardConstraints.check_action`: сера ≤ max, whitelist ranges, сумма BLEND_RATIO ≈ 1, `reliability.is_mode_allowed` |
| `data_quality_gate.py` | `DataQualityGate.check`: incomplete flag, наличие серы, age ЛИМС/ПАК vs пороги |

Если gate не проходит — оркестратор **не** предлагает действие, а возвращает `refuse=True`.

---

### 5.13 Пакет `orchestration/` — главный use-case

| Файл | Что делает |
|---|---|
| `__init__.py` | Экспорт `Orchestrator`, `build_explanation` |
| `orchestrator.py` | `run_cycle`: state → gate → baseline → optimize / refuse → explanation → persist JSON в `artifacts/` |
| `explain.py` | Собирает человекочитаемый текст на русском для оператора |

`Orchestrator.__init__` по умолчанию собирает stubs. Подмена ML:

```python
Orchestrator(
    quality_agent=QualityAgentML("models/quality_sulfur.joblib"),
    reliability_agent=ReliabilityAgentML("models/reliability_proxy.joblib"),
)
```

---

### 5.14 Пакет `demo/` — сценарии для жюри

| Файл | Что делает |
|---|---|
| `__init__.py` | Экспорт сценариев |
| `scenarios.py` | 4 кейса: `normal`, `risk_sulfur`, `stale_data`, `full_cycle` (имя, timestamp, описание) |

| Имя | Timestamp | Ожидание для демо |
|---|---|---|
| `normal` | 2024-06-01 12:00 | устойчивый режим, не дёргать зря |
| `risk_sulfur` | 2024-08-15 08:00 | риск качества / смена режима |
| `stale_data` | 2025-01-10 18:00 | отказ из-за устаревших данных |
| `full_cycle` | 2024-11-20 14:30 | полный проход агентов до рекомендации |

---

### 5.15 Пакет `api/` — optional HTTP

| Файл | Что делает |
|---|---|
| `__init__.py` | Маркер пакета |
| `app.py` | FastAPI: `GET /health`, `POST /run_cycle` → тот же `Orchestrator` |

Запуск: `uvicorn neftecode.api.app:app --reload`

---

## 6. Что уже сделано vs что было в ТЗ

| Требование ТЗ / плана | Статус |
|---|---|
| Контракты ProcessState / assessments / Recommendation | ✅ |
| Оркестратор одного цикла | ✅ |
| Safety: hard constraints + отказ | ✅ (на stubs/scaffold) |
| Optimizer: кандидаты + ranking | ✅ (throughput/energy = 0) |
| 4 демо-сценария + CLI | ✅ |
| Объяснение оператору + audit JSON | ✅ |
| merge_asof по `date` | ✅ |
| time-based split | ✅ |
| Whitelist + assumptions | ⚠️ placeholders |
| Реальный ingest CSV/XLSX | ❌ папка `data/` пустая |
| Реальный ProcessState (as-of + age) | ❌ scaffold |
| Загрузка ЛИМС/ПАК в loaders | ❌ |
| Tag dictionary на реальном файле | ❌ |
| FeatureBuilder с лагами / age | ⚠️ скелет |
| ML-1 QualityAgent | ❌ NotImplemented |
| ML-2 ReliabilityAgent | ❌ NotImplemented |
| Доли блендинга в whitelist | ❌ (намеренно убраны до реальных тегов) |
| Метрики throughput / energy в score | ❌ нули |

---

## 7. Что тебе ещё сделать (бэк) — порядок работ

### Этап A — данные (блокер для всего остального)

1. Положить выгрузки хакатона в `data/`.
2. В `loaders.py` дописать чтение ЛИМС/ПАК (Excel), нормализовать колонку времени в `date`.
3. Прогнать `TagDictionary` на `tags_dictionary.xlsx`, поправить угадывание колонок под реальные имена листов.
4. Заменить `PLACEHOLDER_*` в `tags_whitelist.yaml` на реальные управляемые теги из справочника КИП; заполнить `state_tags`.
5. Зафиксировать каждое допущение в `configs/assumptions.md`.

### Этап B — реальный `ProcessStateBuilder`

1. As-of lookup телеметрии на `timestamp` (`merge_asof` / last known).
2. Подтянуть качество: ЛИМС → ПАК → virtual через `apply_quality_priority`.
3. Считать `age_minutes` от реального `measured_at`.
4. Ставить `data_flags.complete=False`, если критичных тегов нет.
5. Убрать зависимость демо только от hardcoded sulfur — сценарии должны выбирать **реальные timestamp’ы** из данных (имена сценариев можно оставить).

### Этап C — constraints под спецификацию

1. Уточнить hard limits (не только сера).
2. Добавить реальные blend shares (все компоненты, сумма = 1).
3. Проверить пороги stale с технологами / материалами хакатона.
4. Добавить unit-тесты на новые правила.

### Этап D — handoff ML

1. Собрать aligned dataset через `FeatureBuilder` (+ лаги процесса→качество).
2. Отдать ML `time_based_split` с конкретными `train_end` / `val_end`.
3. Зафиксировать 2–3 timestamp’а, на которых JSON стабилен для демо.
4. Договориться об именах файлов в `models/`.
5. Когда модели готовы — реализовать `QualityAgentML` / `ReliabilityAgentML` (или принять PR от ML) и подставить в `Orchestrator(...)` **без смены** `run_cycle`.

### Этап E — демо для жюри

1. `python -m neftecode run --scenario ... --pretty` на 4 кейсах.
2. Показать отказ на stale.
3. Показать, что hard constraints не нарушаются.
4. Показать explanation + alternatives + artifacts.
5. (Опционально) FastAPI для UI, если нужен экран.

---

## 8. Куда двигаться стратегически

Рекомендуемый вектор (не распыляться):

```text
1) Реальные данные → 2) Честный ProcessState → 3) Жёсткий safety
        ↓
4) Handoff таблица ML → 5) Подмена stubs → 6) Полировка демо/объяснений
```

**Не делать сейчас:** тяжёлый Spring-like DI, микросервисы, сложный UI, shuffle в ML, оптимизацию «экономики» в ущерб сере, ручной join по номеру строки.

**Критерии готовности к сдаче (бэк-часть):**

- [ ] Цикл работает на реальных CSV/XLSX, не только на scaffold
- [ ] Возраст ЛИМС/ПАК считается правильно; stale → refuse
- [ ] Whitelist без `PLACEHOLDER_*`
- [ ] Hard constraints покрыты тестами
- [ ] ML подключены через тот же `assess()` (или честно stubs + план, если ML не успели)
- [ ] 4 сценария воспроизводимы одной командой
- [ ] `assumptions.md` заполнен для жюри

---

## 9. Как запускать

```bash
cd /Users/kirillrasshivalov/PycharmProjects/neftecode
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,ml]"

python -m neftecode scenarios
python -m neftecode run --scenario normal --pretty
python -m neftecode run --scenario stale_data --pretty
pytest

# optional API
uvicorn neftecode.api.app:app --reload
```

PyCharm: Sources Root = `src`; Run module `neftecode` / `neftecode.main`, params `run --scenario normal --pretty`.

---

## 10. Контракты для ML (кратко)

### ML-1 — Quality

```text
assess(state, action=None) → QualityAssessment
  metrics: {sulfur_mg_kg, ...}
  risk_of_spec_breach: 0..1
  confidence: 0..1
  horizon_minutes
  features_used, assumptions
```

Скелет: `agents/quality.py` → `QualityAgentML`.

### ML-2 — Reliability

```text
assess(state, action=None) → ReliabilityAssessment
  risk_index, risk_class
  risk_factors
  is_mode_allowed
  soft_constraints, assumptions
```

Скелет: `agents/reliability.py` → `ReliabilityAgentML`.

### Инфраструктура от бэка

- фичи: `data/feature_builder.py`
- split: `data/time_split.py`
- артефакты: `models/`
- оркестратор **не менять** при подключении моделей

---

## 11. Шпаргалка: «где править, если…»

| Задача | Куда идти |
|---|---|
| Поменять лимит серы / stale | `configs/constraints.yaml` |
| Добавить управляемый тег / deltas | `configs/tags_whitelist.yaml` |
| Подключить CSV | `data/loaders.py` + файлы в `data/` |
| Реальный снимок процесса | `data/state_builder.py` |
| Текст отказа / объяснения | `constraints.yaml` `refuse_messages` + `orchestration/explain.py` |
| Логика кандидатов / score | `agents/optimizer.py` |
| Новое hard-правило | `safety/constraints.py` + тест |
| Подставить модель | `agents/quality.py` / `reliability.py` + `Orchestrator(...)` |
| Новый демо-кейс | `demo/scenarios.py` |
| HTTP-обёртка | `api/app.py` |

---

## 12. Итог одной фразой

Каркас МАС уже закрывает **архитектуру ТЗ** (оркестратор, safety, stubs, демо, контракты под ML). Дальше бэк должен **накормить систему реальными данными и честным `ProcessState`**, а ML — **заполнить два `assess()`**; после этого останется отполировать демо и assumptions для жюри.
