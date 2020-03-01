## Решение конкурса https://retailhero.ai/c/recommender_system/ (8 место)

check: 0.1316
public: 0.1294
private: 0.143422
Текущая версия репозитория не совсем совпадает с лучшим на прайвате решением, архив с решением можно найти 
[здесь](https://x5-retailhero.s3-eu-central-1.amazonaws.com/submissions/350e82f7debbfc241864a6d44c1144aa/08167-432954-solution_300k_1472_1316_1397.zip)

## Технические детали
avg latency: 20ms
95%% latency: 28ms

Я по максимуму постарался избавиться от использования `pandas` в рантайме, все построено на родных питонячьих структурах - спискас и словарях.
В контексте одного запроса мы оперируем не очень большим объемом данных, и оверхед на создание `pd.DataFrame` и операции с ним досаточно высок.
Например, наивный `pd.merge` фичей продуктов к кандидатам занимал на `15ms` больше, чем самописная функция `lib.utils.inplace_hash_join`

Также, код, который формирует фичи для обучения и который используется на этапе инференса - один и тот же, это позволяет избежать ошибок.

## Описание решения

Основная идея решения в том - что все прошлые покупки клиентов содержат в среднем 42% продуктов, которые он купит в следующей покупке.
Сформируем кандидатов для ранжирвания и запихаем в GBDT (в моем случае, CatBoost).
Также, я брал дополнительных кандидатов из `implicit.nn.CosineRecommender(K=10)`


### Фичи для ранжирования (`lib/preprocessing.py`):
 - различные статистики по продукту + пользователю (доля транзакций, в которой был продукт, когда была последняя транзакция с продуктом и т.д.)
 - скоры из `implicit`: это `dot product` среднего вектора клиента и продукта и скор из `CosineRecommender`
 - фичи продуктов из `products.csv` + различные статистики (популярность, цена)
 [shap_values.png]
 
### из того, что не взлетело:
 - фичи, связанные со `store_id`
 - фичи, связанные со `store_id` + `product_id` (`lib/product_store_features.py`)



### Пайплайн обучения (`pipepline.py`):

Перед запуском непосредственно обучения, надо подготовить данные при помощи `reformat_data.py`, получится tsv файл, 
где в первом столбце json с историей клиента в train период, а во втором - в test.

Все пути к файликам и часть параметров моделей хранятся в конфиге (`config.json`), а сам конфиг представляет собой класс (`lib/config.py`), 
что очень удобно, т.к. есть автокомплит в PyCharm.

В pipeline.py можно передать аргументы, позволяющие начать с какого-то конкретного шага (например, сразу запустить процесс обучения, а фичи загрузить с диска)
Шаги:
- обучение модели implicit NN
- обучения модели implicit ALS
- подсчет фичей products/store
- подсчет фичей product
- фичи для train
- фичи для test
- обучение модели

Запаковать решение можно скриптом `zip_solution.py`, ему нужно передать путь к конфигу и название.

