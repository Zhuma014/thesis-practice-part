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
│   └── nirm3_week1_ptbxl.ipynb   # подключение PTB-XL, EDA, маппинг диагнозов
├── README.md
```

## Текущий прогресс (Неделя 1)

- ✅ Датасет PTB-XL скачан и распакован (21 799 записей)
- ✅ Реализован маппинг диагностических кодов на 5 суперклассов: NORM (43.6%), MI (25.1%), STTC (24.0%), CD (22.5%), HYP (12.2%); 1.9% записей без метки
- ✅ Реализовано чтение и визуализация сигналов через `wfdb`
- ⏳ Следующий шаг: полноценный EDA (возраст/пол, качество сигналов), предобработка (фильтрация, нормализация), обучение 1D-CNN

## Технологии

- Python 3.10+, Google Colab
- `pandas`, `numpy`, `wfdb`, `matplotlib`
- Планируется: PyTorch (1D-CNN, LSTM, Transformer)

## Запуск

Ноутбук рассчитан на Google Colab:

1. Открыть `notebooks/nirm3_week1_ptbxl.ipynb` в Colab
2. Выполнить ячейку установки зависимостей: `!pip install wfdb --quiet`
3. Выполнить ячейку загрузки датасета PTB-XL (архив ~1.7 GB, скачивается напрямую с physionet.org)
4. Запустить оставшиеся ячейки последовательно

## Лицензия данных

PTB-XL и датасеты UCI распространяются на условиях открытого доступа их правообладателями (PhysioNet Credentialed Health Data License / UCI ML Repository). Код в этом репозитории используется исключительно в образовательных и научно-исследовательских целях в рамках магистерской диссертации.
