# Favorita Store Sales Forecasting

HSE Final Year Project — Time Series Forecasting (2025–2026)

## Задача

16-дневное прогнозирование продаж для сети магазинов Corporacion Favorita (Kaggle competition).
~125M строк обучающих данных, 54 магазина, 4000+ товаров.

## Подход

1. **Статистические бейзлайны**: Naive, SeasonalNaive(7), AutoETS(7), AutoTheta(7)
2. **CatBoost** с winner-inspired features: multi-window статистики, gap=16 для предотвращения утечки данных, threshold tuning
3. **Neural Network**: sequence-модель с Conv1D promo trick (8th place approach)
4. **Ensemble**: линейная комбинация CatBoost + NN с подбором alpha

## Структура репозитория

```
├── run_experiment.py           # Точка входа — полный pipeline
├── src/
│   ├── config.py               # Все константы и гиперпараметры
│   ├── metrics.py              # NWRMSLE, WMAPE, BIAS
│   ├── baselines.py            # Статистические бейзлайны
│   ├── data/
│   │   ├── extract.py          # Распаковка 7z (Colab)
│   │   ├── load.py             # Загрузка CSV
│   │   └── prepare.py          # Подготовка данных
│   ├── features/
│   │   ├── window_stats.py     # Оконные статистики
│   │   ├── pipeline.py         # Feature engineering pipeline
│   │   ├── dense_grid.py       # Dense grid (structural zeros)
│   │   └── lags.py             # Lag features
│   └── models/
│       ├── catboost_model.py   # CatBoost: обучение, предсказание, submission
│       └── nn_model.py         # Neural Network + Ensemble
├── results/
│   └── analysis_results.ipynb  # EDA + анализ результатов
├── requirements.txt
│
├── run_colab.ipynb             # Запуск
│
└── Отчёт
```

## Запуск

1. Для запуска рекомендуется смотреть файл `run_colab.ipynb`.

2. Перед этим скачать [данные](https://drive.google.com/drive/folders/14oi8Z5y74okkTlFBDj7IO_hnoJTV8jqw?usp=sharing) на свой Google Drive.

## Протокол валидации

- **Train**: 2016-08-01 — 2017-07-30 (dense grid: последние 50 дней)
- **Validation**: 2017-07-31 — 2017-08-15 (16 дней)
- **Метрика**: NWRMSLE (1.25× вес для perishable товаров)

## Ключевые результаты

| Модель | NWRMSLE | WMAPE | BIAS |
|--------|---------|-------|------|
| Ensemble CatBoost+NN (α=0.6) | 0.5545 | 0.5828 | -1.1688 |
| CatBoost (winner features + gap=16) | 0.5649 | 0.6102 | -1.1906 |
| NN (sequence + Conv1D promo) | 0.5698 | 0.5908 | -1.1317 |
| AutoETS(7) | 0.7380 | 0.7704 | 0.7916 |
| AutoTheta(7) | 0.7422 | 0.7678 | 0.7665 |
| SeasonalNaive(7) | 0.7553 | 0.7570 | 0.4108 |
| Naive | 0.7636 | 0.8906 | 1.3778 |
