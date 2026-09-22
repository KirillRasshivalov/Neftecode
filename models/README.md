# models/

Коэффициенты моделей в JSON: обучение офлайн, в боевом пути только numpy. Каждый файл
пересобирается своим скриптом (см. README в корне репозитория).

| Файл | Чем создаётся | Что внутри |
|---|---|---|
| `quality_baseline.json` | `python -m scripts.analysis.quality_baseline_fit` | параметры сглаживания лабораторной серы и квантили интервала |
| `breach_classifier.json` | `python -m scripts.analysis.breach_classifier` | коэффициенты классификатора нарушения спецификации, отчёт о приёмке и порог |
| `reliability_reference.json` | `python -m scripts.analysis.reliability_reference` | рабочий диапазон рычагов, медиана `T5` по полосам расхода, уровень замены катализатора |
| `economics_reference.json` | `python -m scripts.analysis.economics_reference` | опорные величины прокси выпуска и энергии |
| `blend_components.json` | `python -m scripts.analysis.blend_components` | свойства компонентов смеси из лабораторных данных |
| `predictability.json` | `python -m scripts.analysis.predictability` | результат проверки предсказуемости серы по телеметрии |

`labels.parquet` в git не хранится: это выборка из лабораторных данных организаторов.
Она строится командой `python -m scripts.analysis.labels` и нужна до первого запуска.
