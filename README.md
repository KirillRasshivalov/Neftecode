# Neftecode MAS (каркас)

Мультиагентная система поддержки решений для цепочки **АВТ → гидроочистка → блендинг**.

Сейчас это **рабочий каркас на stubs**: оркестратор, safety-gate, оптимизатор и демо-сценарии уже крутятся без реальных CSV. Данные и ML-модели подключаются без смены контрактов.

## Открыть в PyCharm

1. **File → Open** → `/Users/kirillrasshivalov/PycharmProjects/neftecode`
2. PyCharm предложит создать venv — согласись (Python 3.11+).
3. В Terminal:

```bash
pip install -e ".[dev,api,ml]"
# или
pip install -r requirements.txt && pip install -e .
```

4. Mark Directory as → **Sources Root** для `src` (обычно подхватывается из `pyproject.toml`).
5. Run configuration: Module name `neftecode.main`, parameters `run --scenario normal --pretty`.

## Быстрый запуск

```bash
python -m neftecode scenarios
python -m neftecode run --scenario normal --pretty
python -m neftecode run --scenario stale_data --pretty
pytest
```

Опционально API:

```bash
uvicorn neftecode.api.app:app --reload
```

## Структура (Java → Python)

| Пакет | Аналог в Java | Назначение |
|---|---|---|
| `domain/` | DTO / records | `ProcessState`, `ControlAction`, assessments, recommendation |
| `data/` | Repository | loaders, `merge_asof`, freshness, feature builder, time split |
| `agents/` | Services | Quality / Reliability / Optimizer |
| `safety/` | Validation | hard constraints, data quality gate |
| `orchestration/` | Use-case | `Orchestrator.run_cycle` |
| `demo/` | Fixtures | 4 сценария для жюри |
| `api/` | Controller | optional FastAPI |
| `configs/` | application.yml | whitelist тегов, constraints, assumptions |

## Контракты для ML

- **ML-1** реализует `QualityAgent.assess(state, action=None) → QualityAssessment`  
  Скелет: `agents/quality.py` → `QualityAgentML`
- **ML-2** реализует `ReliabilityAgent.assess(...) → ReliabilityAssessment`  
  Скелет: `agents/reliability.py` → `ReliabilityAgentML`
- Фичи / split: `data/feature_builder.py`, `data/time_split.py`
- Артефакты моделей: положить в `models/`, оркестратор не трогать

Пока подключены stubs: `QualityAgentStub`, `ReliabilityAgentStub`.

## Данные

Положи выгрузки хакатона в `data/`:

- `avt_tags.csv`
- `242000_tags.csv`
- `lims.xlsx` / `pak.xlsx` (имена можно поправить в `data/loaders.py`)
- справочник тегов → `tags_dictionary.xlsx`

Синхронизация **только по `date`**, не по номеру строки.

## Что делать дальше (бэк)

1. Заменить `PLACEHOLDER_*` в `configs/tags_whitelist.yaml` на реальные управляемые теги.
2. Дописать реальный `ProcessStateBuilder` (as-of lookup из телеметрии + ЛИМС/ПАК + age).
3. Уточнить hard constraints под спецификацию сценария.
4. Отдать ML aligned dataset + 2–3 timestamp’а, на которых JSON стабилен.
5. Подставить ML-агенты в `Orchestrator(...)`.

## Допущения

См. `configs/assumptions.md`.
