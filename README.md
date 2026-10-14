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
│   ├── nirm3_week5_ptbxl.ipynb      # неделя 5: baseline 1D-CNN (ResNet1d), абляция нормировки
│   ├── nirm3_week6_ptbxl.ipynb      # неделя 6: EDA по полу/возрасту/аппарату, пороги скрининга по полу
│   ├── nirm3_week7-8_colab_training.ipynb  # недели 7–8: обучение всех глубоких моделей на GPU (Colab)
│   ├── nirm3_week7_ptbxl.ipynb      # неделя 7: базовая 1D-CNN, гиперпараметры, устойчивость к seed
│   ├── nirm3_week8_ptbxl.ipynb      # неделя 8: гибриды CNN-LSTM / CNN-Transformer, сравнение архитектур
│   ├── nirm3_week9_ptbxl.ipynb      # неделя 9: сравнение всех моделей, Grad-CAM, объяснения по полу
│   ├── nirm4_sexaware_colab.ipynb   # НИРМ 4: обучение CNN-LSTM с возрастом/полом и весами по полу (Colab)
│   ├── nirm4_sexaware_ptbxl.ipynb   # НИРМ 4: анализ первого эксперимента по разрыву между полами
│   ├── nirm4_method_colab.ipynb     # НИРМ 4: учёт пола на уровне отведений, FiLM, нормировка по полу, adversarial (Colab)
│   ├── nirm4_method_ptbxl.ipynb     # НИРМ 4: сравнение 8 стратегий снижения разрыва
│   └── ecg_models.py                # архитектуры для загрузки весов, обученных на Colab
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
│   │   ├── week01.pdf ... week09.pdf      # по одному отчёту на неделю
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

### `notebooks/nirm3_week6_ptbxl.ipynb` — неделя 6: исследовательский анализ данных

Без обучения, на предсказаниях ResNet1d недели 5: баланс классов по полу и возрасту, демография
пациентов, связь флагов качества с типом аппарата, влияние помех на AUC, пороги скрининга, подобранные
отдельно по полу (сокращают разрыв чувствительности по MI с −0.084 до −0.025), чувствительность к MI
по подтипам инфаркта.

### `notebooks/nirm3_week7-8_colab_training.ipynb` — недели 7–8: обучение на GPU

Единый ноутбук для Google Colab (T4): ResNet1d, простая 1D-CNN, CNN-LSTM и CNN-Transformer с
attention-pooling. Чекпойнт после каждой эпохи — после обрыва сессии обучение продолжается с той же
эпохи. Сохраняет предсказания, веса и историю каждого запуска в `results/`.

### `notebooks/nirm3_week7_ptbxl.ipynb` — неделя 7: 1D-CNN, гиперпараметры, seed

Воспроизведение ResNet1d на GPU (0.9295 против 0.9297 на CPU), простая 1D-CNN (0.932), сетка
гиперпараметров ResNet1d (на тесте прироста нет), 3 seed — разрыв по полу устойчив
(macro −0.023 [−0.037; −0.009]).

### `notebooks/nirm3_week8_ptbxl.ipynb` — неделя 8: гибридные модели

CNN-LSTM 0.932, ResNet1d 0.930, CNN-Transformer 0.929 (по 3 seed, парный бутстреп); разрыв по полу у
всех архитектур; CNN-LSTM выбрана основной моделью — лучшее качество при обучении за 1.2 минуты.

### `notebooks/nirm3_week9_ptbxl.ipynb` — неделя 9: сравнение и интерпретация

XGBoost и шесть нейросетей по метрикам ИПРМ (ROC-AUC, PR-AUC, precision, recall, F1, accuracy):
нейросети лучше на 0.035 ROC-AUC. Grad-CAM, веса attention-pooling и вклад отведений (|градиент × вход|)
по весам из `cache/results/`; тест удаления подтверждает Grad-CAM, но не веса внимания. Для MI у женщин
модель меньше опирается на V4–V5 и на QRS — в том числе внутри подтипов инфаркта.

### НИРМ 4: можно ли устранить разрыв качества между мужчинами и женщинами

`nirm4_sexaware_colab.ipynb` и `nirm4_method_colab.ipynb` обучают на Colab (T4) восемь вариантов CNN-LSTM по
5 seed: возраст и пол на входе, веса потерь с равным вкладом полов, учёт пола на уровне отведений (lead
gating), FiLM, нормировку по статистике своего пола и adversarial debiasing. Анализ — в
`nirm4_sexaware_ptbxl.ipynb` и `nirm4_method_ptbxl.ipynb`: разрыв по MI не сокращает ни одна стратегия,
macro-разрыв сужает только adversarial debiasing и только за счёт качества у мужчин; амплитуда грудных
отведений у женщин на 12–21 % ниже, но её выравнивание разрыв не меняет. Пороги с равной чувствительностью
выравнивают пропуски MI ценой 5–8 п. п. специфичности у женщин. Для запуска нужен `cache/colab_meta_v2.npz`
(метаданные с возрастом) рядом с `X100_bp05-40_crop.npy` на Drive; результаты — в `cache/results_nirm4/`.

## Документы

- **`docs/paper/250.docx`** — черновик статьи *"Gender-Specific Feature Asymmetry in Heart Disease Prediction"*, IEEE SIST 2026
- **`docs/practice/`** — административные документы магистратуры: индивидуальный план работы (ИПРМ), календарный план практики, отчёты НИРМ за 1-й и 2-й семестры, документ по научной стажировке
- **`docs/reports/`** — еженедельные отчёты о прогрессе по НИРМ3 (10-недельный план), по одному PDF на неделю (`week01.pdf`, `week02.pdf`, ...)
- **`docs/planning/`** — рабочие заметки по планированию: план еженедельных отчётов и контекст/память проекта (перенесены как заготовки с сайта claude.ai, пока пустые — заполняются по мере работы)

## Текущий прогресс (НИРМ3 завершена, НИРМ 4 начата)

- ✅ Неделя 1: PTB-XL подключён, маппинг диагнозов на 5 суперклассов, визуализация сигналов
- ✅ Неделя 2: демография, метрики качества сигналов, выбор `records100`
- ✅ Неделя 3: таблица архитектур, контроль качества всего датасета, пайплайн предобработки
- ✅ Неделя 4: разбиение, протокол оценки, baseline XGBoost (macro-AUC 0.897)
- ✅ Неделя 5: baseline 1D-CNN ResNet1d (macro-AUC 0.930), абляция нормировки, анализ по полу
- ✅ Неделя 6: EDA по полу, возрасту и аппарату; пороги скрининга по полу; анализ пропусков MI
- ✅ Неделя 7: обучение на GPU, базовая 1D-CNN, подбор гиперпараметров, устойчивость к seed
- ✅ Неделя 8: CNN-LSTM (0.932) и CNN-Transformer (0.929), разрыв по полу у всех архитектур
- ✅ Неделя 9: сравнение всех моделей, рекомендации, Grad-CAM и вклад отведений, объяснения по полу
- ✅ Неделя 10: итоговые выводы, отчёт по практике и отчёт НИРМ-3
- ✅ НИРМ 4, начало: 8 стратегий снижения разрыва по полу (2 эксперимента × 5 seed)
- ⏳ Далее: согласование новизны с руководителем, внешняя валидация, вторая статья

## Технологии

- Python 3.10+ (локально проверено на 3.14), Google Colab
- `pandas`, `numpy`, `scipy`, `wfdb`, `matplotlib`, `scikit-learn`, `xgboost`
- PyTorch (1D-CNN, CNN-LSTM, CNN-Transformer), Google Colab (Tesla T4)

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

**Обучение недель 7–8 на Colab.** Загрузить на Google Drive в `MyDrive/thesis_ptbxl/` (или
`MyDrive/Colab Notebooks/thesis_ptbxl/`) два файла из `notebooks/cache/`: `X100_bp05-40_crop.npy` и
`colab_meta.npz`; открыть `nirm3_week7-8_colab_training.ipynb`, выбрать T4 GPU и `Run all` (19 запусков,
около 1 ч 40 мин). Папку `results/` скачать в `notebooks/cache/results/` — её читают ноутбуки недель 7–8.

## Лицензия данных

PTB-XL и датасеты UCI распространяются на условиях открытого доступа их правообладателями (PhysioNet Credentialed Health Data License / UCI ML Repository). Код в этом репозитории используется исключительно в образовательных и научно-исследовательских целях в рамках магистерской диссертации.
