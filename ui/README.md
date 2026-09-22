# UI · консоль оператора

```bash
# из корня репозитория
streamlit run ui/app.py
```

Карточки — сохранённые JSON-трассы решений на данных организаторов: `stable`, `risk`,
`degraded`, `shutdown`. Пересобрать их можно так (кэш и метки — см. README в корне):

```bash
python -m scripts.run_real --at "2026-01-27 16:00"   # stable
python -m scripts.run_real --at "2026-07-18 16:00"   # risk
python -m scripts.run_real --at "2026-07-02 10:00"   # degraded
python -m scripts.run_real --at "2026-06-21 10:00"   # shutdown
```

Каждый запуск пишет `artifacts/real/recommendation_<метка времени>.json` — скопируйте его
в `ui/showcase/<имя>.json`.
