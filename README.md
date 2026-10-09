# Heart Disease Recognition — Thesis Practice Part

Магистерская диссертация: **«Разработка методов и алгоритмов искусственного интеллекта для распознавания болезни сердца»**
*(Development of Artificial Intelligence Methods and Algorithms for Heart Disease Recognition)*

Магистрант: Сабит Жұмабек
Программа: Applied Artificial Intelligence, 2 курс, Astana IT University
Научный руководитель: Серек Азамат Галымжанович

Репозиторий содержит код исследовательской практики / НИРМ3, посвящённый переходу от классических ML-моделей на клинических данных (год 1) к глубоким нейросетевым моделям на ЭКГ-сигналах (год 2).

## Контекст исследования

**Год 1 (завершён):** Cleveland Heart Disease Dataset (UCI) + XGBoost, SHAP-анализ, гендерно-стратифицированное моделирование, оптимизация порога классификации, внешняя валидация на Hungarian Heart Disease Dataset. Результаты опубликованы в статье на IEEE SIST 2026:
> *"Gender-Specific Feature Asymmetry in Heart Disease Prediction: An Explainable AI Approach for High-Sensitivity Clinical Screening"*, IEEE 6th SIST 2026, Astana, Kazakhstan.

**Год 2 (текущий этап):** переход к ЭКГ-данным (PTB-XL) и глубоким архитектурам — 1D-CNN, LSTM, Transformer.

## Датасеты

| Датасет | Описание | Доступ |
|---|---|---|
| [UCI Cleveland Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 наблюдения, 13 клинических признаков | public |
| [UCI Hungarian Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | 294 наблюдения, внешняя валидация | public |
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 21 799 записей, 12-канальная ЭКГ, 100/500 Гц | public |

## Структура репозитория

```
.
├── notebooks/
│   ├── sist_research.ipynb          # Год 1: Cleveland + XGBoost + SHAP (статья IEEE SIST 2026)
│   ├── nirm3_week1_ptbxl (1).ipynb  # Год 2, НИРМ3 неделя 1: подключение PTB-XL, EDA, маппинг диагнозов
│   ├── nirm3_week2_ptbxl.ipynb      # неделя 2: демография, качество сигналов, records100 vs records500
│   ├── nirm3_week3_ptbxl.ipynb      # неделя 3: таблица архитектур, QC всего датасета, предобработка
│   ├── nirm3_week4_ptbxl.ipynb      # неделя 4: метки, разбиение, протокол оценки, baseline XGBoost
│   └── nirm3_week5_ptbxl.ipynb      # неделя 5: baseline 1D-CNN (ResNet1d), абляция нормировки
├── docs/
│   ├── paper/
│   │   └── 250.docx                       # черновик статьи IEEE SIST 2026
│   ├── practice/                          # административные документы по практике/НИРМ
│   │   ├── ИПРМ_Сабит_Жұмабек.pdf          # индивидуальный план работы магистранта
│   │   ├── Календарный_план_практики.pdf
│   │   ├── NIRM1_Sabit.pdf                # отчёт НИРМ, семестр 1
│   │   ├── NIRM2_Sabit.pdf                # отчёт НИРМ, семестр 2
│   │   └── науч_стаж.pdf                  # научная стажировка
│   ├── reports/                           # еженедельные отчёты о прогрессе НИРМ3
│   │   ├── week01.pdf ... week05.pdf      # по одному отчёту на неделю
│   │   └── _src/                          # генератор PDF-отчётов (ReportLab)
│   └── planning/
│       ├── weekly-report-plan.md          # план еженедельных отчётов (заполняется)
│       └── project-memory.md              # контекст/память проекта (заполняется)
├── README.md
```

## Описание ноутбуков

### `notebooks/sist_research.ipynb` — Год 1: Cleveland + XGBoost + SHAP

Пайплайн классического ML на клинических данных, легший в основу статьи IEEE SIST 2026:

1. Загрузка Cleveland Heart Disease Dataset (`wget` напрямую в Colab)
2. Очистка данных: замена `?` на `NaN`, удаление пропусков
3. One-Hot encoding категориальных признаков (`cp`, `restecg`, `slope`, `thal`)
4. Стандартизация числовых признаков (`StandardScaler`)
5. Проверка баланса классов (`target`: healthy / disease)
6. Обучение базовой модели `XGBClassifier` с `scale_pos_weight` под дисбаланс классов
7. Подбор гиперпараметров через `GridSearchCV` (max_depth, learning_rate, n_estimators, subsample)
8. Оценка качества: accuracy, ROC-AUC, classification report, confusion matrix
9. Интерпретация модели через SHAP (summary plot, waterfall для отдельного пациента, dependence plot по возрасту)
10. Гендерно-стратифицированный SHAP-анализ (отдельно для мужчин и женщин)
11. Оптимизация порога классификации (0.5 → 0.3) для повышения recall в клиническом сценарии скрининга
12. Сравнение confusion matrix и метрик (accuracy/precision/recall/F1) при разных порогах

### `notebooks/nirm3_week1_ptbxl (1).ipynb` — Год 2, НИРМ3 неделя 1: PTB-XL

Первый шаг перехода к ЭКГ-данным, продолжает `sist_research.ipynb`:

1. Установка зависимости `wfdb`
2. Скачивание и распаковка архива PTB-XL (~1.7 GB) напрямую с physionet.org
3. Загрузка метаданных (`ptbxl_database.csv`, `scp_statements.csv`)
4. Маппинг диагностических SCP-кодов на 5 суперклассов: NORM, MI, STTC, CD, HYP
5. Подсчёт и визуализация распределения классов (bar chart)
6. Чтение сигнала одной записи через `wfdb.rdsamp` и визуализация всех 12 отведений ЭКГ
7. Итоговая сводка недели: объём данных (21 837 записей), дисбаланс классов относительно NORM, необходимость балансировки (SMOTE / class weights) и следующие шаги (EDA по возрасту/полу, предобработка сигнала, обучение 1D-CNN)

### `notebooks/nirm3_week2_ptbxl.ipynb` — неделя 2: EDA и качество сигналов

Демография по возрасту и полу (в том числе в разрезе суперклассов), метрики качества сигналов
(дрейф изолинии, ВЧ-шум, плоские и мёртвые отведения) со сверкой по врачебной разметке PTB-XL,
сравнение `records100` и `records500`. Решение: основной формат — `records100`, окно 0.5–9.5 с.

### `notebooks/nirm3_week3_ptbxl.ipynb` — неделя 3: архитектуры и предобработка

Сравнительная таблица архитектур (1D-CNN, LSTM, Transformer, гибриды) по опубликованным результатам
на PTB-XL; метрики качества по всем 21 799 записям; пайплайн предобработки (Баттерворт 0.5–40 Гц
с нулевой фазой, окно 0.5–9.5 с, нормировка по статистике фолдов 1–8). Создаёт кэш `notebooks/cache/`.

### `notebooks/nirm3_week4_ptbxl.ipynb` — неделя 4: разбиение, протокол оценки, XGBoost

Многометочная матрица меток, разбиение `strat_fold` 1–8 / 9 / 10, единый протокол оценки
(macro-AUC с бутстреп-интервалами, анализ по полу, порог скрининга), признаки сигнала + XGBoost
(test macro-AUC 0.897) и TreeSHAP по полу.

### `notebooks/nirm3_week5_ptbxl.ipynb` — неделя 5: baseline 1D-CNN

ResNet1d (вариант `resnet1d_wang`) на PyTorch: test macro-AUC 0.930 [0.923–0.936], на уровне
опубликованного бенчмарка; +0.033 к XGBoost; абляция нормировки (z-score каждой записи снижает AUC
HYP с 0.905 до 0.837); разрыв качества и чувствительности скрининга между мужчинами и женщинами.

## Документы

- **`docs/paper/250.docx`** — черновик статьи *"Gender-Specific Feature Asymmetry in Heart Disease Prediction"*, IEEE SIST 2026
- **`docs/practice/`** — административные документы магистратуры: индивидуальный план работы (ИПРМ), календарный план практики, отчёты НИРМ за 1-й и 2-й семестры, документ по научной стажировке
- **`docs/reports/`** — еженедельные отчёты о прогрессе по НИРМ3 (10-недельный план), по одному PDF на неделю (`week01.pdf`, `week02.pdf`, ...)
- **`docs/planning/`** — рабочие заметки по планированию: план еженедельных отчётов и контекст/память проекта (перенесены как заготовки с сайта claude.ai, пока пустые — заполняются по мере работы)

## Текущий прогресс (неделя 5 из 10)

- ✅ Неделя 1: PTB-XL подключён, маппинг диагнозов на 5 суперклассов, визуализация сигналов
- ✅ Неделя 2: демография, метрики качества сигналов, выбор `records100`
- ✅ Неделя 3: таблица архитектур, контроль качества всего датасета, пайплайн предобработки
- ✅ Неделя 4: разбиение, протокол оценки, baseline XGBoost (macro-AUC 0.897)
- ✅ Неделя 5: baseline 1D-CNN ResNet1d (macro-AUC 0.930), абляция нормировки, анализ по полу
- ⏳ Далее: пороги скрининга с учётом пола, устойчивость по seed, гибридные модели CNN-LSTM / CNN-Transformer

## Технологии

- Python 3.10+ (локально проверено на 3.14), Google Colab
- `pandas`, `numpy`, `scipy`, `wfdb`, `matplotlib`, `scikit-learn`, `xgboost`
- PyTorch (1D-CNN; далее LSTM, Transformer)

## Запуск

Оба ноутбука рассчитаны на Google Colab:

**`notebooks/sist_research.ipynb`** (Год 1, Cleveland + XGBoost):
1. Открыть ноутбук в Colab и выполнить ячейки последовательно — датасет скачивается напрямую (`wget`) с GitHub

**`notebooks/nirm3_week1_ptbxl (1).ipynb`** (Год 2, PTB-XL):
1. Открыть ноутбук в Colab
2. Выполнить ячейку установки зависимостей: `!pip install wfdb --quiet`
3. Выполнить ячейку загрузки датасета PTB-XL (архив ~1.7 GB, скачивается напрямую с physionet.org)
4. Запустить оставшиеся ячейки последовательно

**Ноутбуки недель 3–5 локально.** Выполняются по порядку: неделя 3 создаёт кэш `notebooks/cache/`
(предобработанные сигналы, ~0.9 ГБ), недели 4–5 его используют. Данные ищутся по `**/ptbxl_database.csv`
относительно `notebooks/`; достаточно `ptbxl_database.csv`, `scp_statements.csv` и `records100/`.
Быстрее всего их скачать с зеркала PhysioNet на AWS: `https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/`.
`notebooks/ptbxl/`, `notebooks/cache/` и `.venv/` в git не попадают. Обучение ResNet1d на CPU — около
40 минут на модель.

## Лицензия данных

PTB-XL и датасеты UCI распространяются на условиях открытого доступа их правообладателями (PhysioNet Credentialed Health Data License / UCI ML Repository). Код в этом репозитории используется исключительно в образовательных и научно-исследовательских целях в рамках магистерской диссертации.
