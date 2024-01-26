# ReviewNet

## Описание проекта

На основе [открытого датасета с отзывами из Яндекс.Карт](https://github.com/yandex/geo-reviews-dataset-2023) обучена модель для автоматического выставления рейтинга(от 0 до 5) на основе текста отзыва. Логика работы разделена на 2 этапа:
- Получение эмбеддиногового представления текста отзыва
- Мультиклассовая классификация на 6 классов

## 1. Получение эмбеддингового представления
Для получения эмбеддингов использовалась small версия модели ruGPT3. [HuggingFace](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2)
Значения на выходе из модели агрегировались с помощью mean pooling. Реализация модели находится по пути 
```
models/GPTFeatureExtraction.py
```
Процесс получения эмбеддингов для датасета реализован в следующем файле
```
feature_extraction.py
```
Результат работы доступен на [Kaggle](https://www.kaggle.com/datasets/lockiultra/yandex-geo-reviews-embeddings)

## 2. Мультиклассовая классификация
Для классификации использовалась полносвязная нейронная сеть, реализация которой доступна по пути
```
models/ReviewNet.py
```
Модель обучалась с помощью кросс-энтропии в течение 3 эпох, оптимизатором был выбран Adam с learning_rate 3e-4. Код обучения модели находится в файле 
```
model_train.py
``` 
В качестве метрики качества итоговой модели была выбрана f1_score на отложенной выборке, результаты представлены на графике:
<a href="https://imgbb.com/"><img src="https://i.ibb.co/yf25MwT/image.png" alt="image" border="0"></a>