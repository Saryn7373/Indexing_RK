# Индексирование текстов — экзамен, вариант 10

Выполненные задания: **02, 03, 04, 05, 06, 07, 08, 12**.

Каждое задание лежит в отдельной папке `taskXX_*`. Внутри:
- `main.py` запускает задание целиком;
- в `results/` сохраняется вывод: таблицы в `run_log.txt` и json с метриками;
- в `temp/` пишутся временные файлы, эта папка не хранится в git.

Общий код вынесен в `utils/`, общие данные — в `data/`. Модели Word2Vec лежат в `task08_word2vec/models/`.

## Установка

Python 3.11. Все команды выполняются из корня проекта.

```bash
pip install -r requirements.txt
```

## Запуск

| № | Задание | Команды |
|---|---|---|
| 02 | Морфологическое индексирование | `python -m task02_morphology.main`<br>`python -m task02_morphology.search "нигилист Базаров" --mode lemma` |
| 03 | Позиционный индекс и язык запросов | `python -m task03_positional_index.main`<br>`python -m unittest task03_positional_index.tests`<br>`python -m task03_positional_index.main --query '"мёртвые души" AND NOT гоголь' --tree` |
| 04 | Сжатие и ускорение индекса | `python -m task04_compression.main`<br>`python -m unittest task04_compression.tests` |
| 05 | Статистическое ранжирование | `python -m task05_ranking.main`<br>`python -m task05_ranking.main --query "три карты и старая графиня" --method bm25 --explain D026` |
| 06 | Тезаурусный поиск | `python -m task06_thesaurus.main`<br>`python -m task06_thesaurus.search "судьба собаки" --mode synonyms --w-syn 0.7` |
| 07 | Поиск дубликатов | `python -m task07_duplicates.generator`<br>`python -m task07_duplicates.main`<br>`python -m task07_duplicates.tests`<br>`python -m task07_duplicates.main compare "a b c d" "a b c e" --k 2` |
| 08 | Word2Vec | `python -m task08_word2vec.main`<br>`python -m task08_word2vec.main --word Мышкин`<br>`python -m task08_word2vec.main --query "произведения о жизни в деревне"` |
| 12 | Ключевые фразы и кластеризация | `python -m task12_keyphrases_clustering.main`<br>`python -m task12_keyphrases_clustering.main keyphrases --doc D022` |

## Синтаксис запросов (задание 03)

```
query     := <пусто> | or_expr
or_expr   := and_expr ( "OR" and_expr )*
and_expr  := not_expr ( "AND" not_expr )*
not_expr  := "NOT" not_expr | near_expr
near_expr := primary ( "NEAR/k" primary )*
primary   := СЛОВО | "ФРАЗА" | "(" or_expr ")"
```

- Приоритет операторов: `NEAR/k` > `NOT` > `AND` > `OR`. Операторы пишутся латиницей заглавными буквами, русские «и», «не», «или» — обычные слова.
- `"фраза"` — точная фраза: соседние слова стоят на позициях с разницей 1.
- `A NEAR/k B` — упорядоченная близость: `0 < позиция(B) − позиция(A) ≤ k`.
- `NOT X` вычисляется относительно всех актуальных документов.
- Между операндами нужен оператор, неявного AND нет. Пустой запрос даёт `[]`.
- Фраза и NEAR могут переходить через границу предложения, но не между документами.

## Данные

Корпус собирается командой:

```bash
python -m data.prepare_data
```

Скрипт разбирает скачанные файлы и записывает `data/corpus.jsonl`: 160 документов, по 40 прозы, поэзии, публицистики и критики.

**1. Kaggle, «Russian literature»** — проза, поэзия и публицистика, документы D001–D120. Лицензия PDDL.
https://www.kaggle.com/datasets/d0rj3228/russian-literature
Нажать Download и сохранить архив как `data/raw/archive.zip`.

**2. Русская Википедия** — статьи о произведениях, документы D121–D160. Лицензия CC BY-SA 4.0.
Ответ по каждой ссылке нужно сохранить как `data/raw/wiki/NN.json`.

| NN | Статья | Ссылка |
|---|---|---|
| 01 | Анна Каренина | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%90%D0%BD%D0%BD%D0%B0%20%D0%9A%D0%B0%D1%80%D0%B5%D0%BD%D0%B8%D0%BD%D0%B0) |
| 02 | Война и мир | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%92%D0%BE%D0%B9%D0%BD%D0%B0%20%D0%B8%20%D0%BC%D0%B8%D1%80) |
| 03 | Воскресение (роман) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%92%D0%BE%D1%81%D0%BA%D1%80%D0%B5%D1%81%D0%B5%D0%BD%D0%B8%D0%B5%20%28%D1%80%D0%BE%D0%BC%D0%B0%D0%BD%29) |
| 04 | Смерть Ивана Ильича | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%A1%D0%BC%D0%B5%D1%80%D1%82%D1%8C%20%D0%98%D0%B2%D0%B0%D0%BD%D0%B0%20%D0%98%D0%BB%D1%8C%D0%B8%D1%87%D0%B0) |
| 05 | Крейцерова соната (повесть) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9A%D1%80%D0%B5%D0%B9%D1%86%D0%B5%D1%80%D0%BE%D0%B2%D0%B0%20%D1%81%D0%BE%D0%BD%D0%B0%D1%82%D0%B0%20%28%D0%BF%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D1%8C%29) |
| 06 | Хаджи-Мурат (повесть) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%A5%D0%B0%D0%B4%D0%B6%D0%B8-%D0%9C%D1%83%D1%80%D0%B0%D1%82%20%28%D0%BF%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D1%8C%29) |
| 07 | Идиот (роман) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%98%D0%B4%D0%B8%D0%BE%D1%82%20%28%D1%80%D0%BE%D0%BC%D0%B0%D0%BD%29) |
| 08 | Бесы (роман) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%91%D0%B5%D1%81%D1%8B%20%28%D1%80%D0%BE%D0%BC%D0%B0%D0%BD%29) |
| 09 | Братья Карамазовы | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%91%D1%80%D0%B0%D1%82%D1%8C%D1%8F%20%D0%9A%D0%B0%D1%80%D0%B0%D0%BC%D0%B0%D0%B7%D0%BE%D0%B2%D1%8B) |
| 10 | Белые ночи (повесть) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%91%D0%B5%D0%BB%D1%8B%D0%B5%20%D0%BD%D0%BE%D1%87%D0%B8%20%28%D0%BF%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D1%8C%29) |
| 11 | Бедные люди | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%91%D0%B5%D0%B4%D0%BD%D1%8B%D0%B5%20%D0%BB%D1%8E%D0%B4%D0%B8) |
| 12 | Мёртвые души | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9C%D1%91%D1%80%D1%82%D0%B2%D1%8B%D0%B5%20%D0%B4%D1%83%D1%88%D0%B8) |
| 13 | Шинель (повесть) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%A8%D0%B8%D0%BD%D0%B5%D0%BB%D1%8C%20%28%D0%BF%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D1%8C%29) |
| 14 | Повести Белкина | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9F%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D0%B8%20%D0%91%D0%B5%D0%BB%D0%BA%D0%B8%D0%BD%D0%B0) |
| 15 | Тарас Бульба | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%A2%D0%B0%D1%80%D0%B0%D1%81%20%D0%91%D1%83%D0%BB%D1%8C%D0%B1%D0%B0) |
| 16 | Вий (повесть) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%92%D0%B8%D0%B9%20%28%D0%BF%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D1%8C%29) |
| 17 | Отцы и дети (роман) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9E%D1%82%D1%86%D1%8B%20%D0%B8%20%D0%B4%D0%B5%D1%82%D0%B8%20%28%D1%80%D0%BE%D0%BC%D0%B0%D0%BD%29) |
| 18 | Муму | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9C%D1%83%D0%BC%D1%83) |
| 19 | Записки охотника | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%97%D0%B0%D0%BF%D0%B8%D1%81%D0%BA%D0%B8%20%D0%BE%D1%85%D0%BE%D1%82%D0%BD%D0%B8%D0%BA%D0%B0) |
| 20 | Ася (повесть) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%90%D1%81%D1%8F%20%28%D0%BF%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D1%8C%29) |
| 21 | Капитанская дочка | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9A%D0%B0%D0%BF%D0%B8%D1%82%D0%B0%D0%BD%D1%81%D0%BA%D0%B0%D1%8F%20%D0%B4%D0%BE%D1%87%D0%BA%D0%B0) |
| 22 | Пиковая дама (повесть) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9F%D0%B8%D0%BA%D0%BE%D0%B2%D0%B0%D1%8F%20%D0%B4%D0%B0%D0%BC%D0%B0%20%28%D0%BF%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D1%8C%29) |
| 23 | Дубровский (роман) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%94%D1%83%D0%B1%D1%80%D0%BE%D0%B2%D1%81%D0%BA%D0%B8%D0%B9%20%28%D1%80%D0%BE%D0%BC%D0%B0%D0%BD%29) |
| 24 | Евгений Онегин | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%95%D0%B2%D0%B3%D0%B5%D0%BD%D0%B8%D0%B9%20%D0%9E%D0%BD%D0%B5%D0%B3%D0%B8%D0%BD) |
| 25 | Медный всадник (поэма) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9C%D0%B5%D0%B4%D0%BD%D1%8B%D0%B9%20%D0%B2%D1%81%D0%B0%D0%B4%D0%BD%D0%B8%D0%BA%20%28%D0%BF%D0%BE%D1%8D%D0%BC%D0%B0%29) |
| 26 | Руслан и Людмила (поэма) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%A0%D1%83%D1%81%D0%BB%D0%B0%D0%BD%20%D0%B8%20%D0%9B%D1%8E%D0%B4%D0%BC%D0%B8%D0%BB%D0%B0%20%28%D0%BF%D0%BE%D1%8D%D0%BC%D0%B0%29) |
| 27 | Герой нашего времени | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%93%D0%B5%D1%80%D0%BE%D0%B9%20%D0%BD%D0%B0%D1%88%D0%B5%D0%B3%D0%BE%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%B8) |
| 28 | Бородино (стихотворение) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%91%D0%BE%D1%80%D0%BE%D0%B4%D0%B8%D0%BD%D0%BE%20%28%D1%81%D1%82%D0%B8%D1%85%D0%BE%D1%82%D0%B2%D0%BE%D1%80%D0%B5%D0%BD%D0%B8%D0%B5%29) |
| 29 | Кому на Руси жить хорошо | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9A%D0%BE%D0%BC%D1%83%20%D0%BD%D0%B0%20%D0%A0%D1%83%D1%81%D0%B8%20%D0%B6%D0%B8%D1%82%D1%8C%20%D1%85%D0%BE%D1%80%D0%BE%D1%88%D0%BE) |
| 30 | Двенадцать (поэма) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%94%D0%B2%D0%B5%D0%BD%D0%B0%D0%B4%D1%86%D0%B0%D1%82%D1%8C%20%28%D0%BF%D0%BE%D1%8D%D0%BC%D0%B0%29) |
| 31 | Дама с собачкой | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%94%D0%B0%D0%BC%D0%B0%20%D1%81%20%D1%81%D0%BE%D0%B1%D0%B0%D1%87%D0%BA%D0%BE%D0%B9) |
| 32 | Палата № 6 | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9F%D0%B0%D0%BB%D0%B0%D1%82%D0%B0%20%E2%84%96%206) |
| 33 | Человек в футляре | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%A7%D0%B5%D0%BB%D0%BE%D0%B2%D0%B5%D0%BA%20%D0%B2%20%D1%84%D1%83%D1%82%D0%BB%D1%8F%D1%80%D0%B5) |
| 34 | Мать (роман) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9C%D0%B0%D1%82%D1%8C%20%28%D1%80%D0%BE%D0%BC%D0%B0%D0%BD%29) |
| 35 | Старуха Изергиль | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%A1%D1%82%D0%B0%D1%80%D1%83%D1%85%D0%B0%20%D0%98%D0%B7%D0%B5%D1%80%D0%B3%D0%B8%D0%BB%D1%8C) |
| 36 | Исповедь (Толстой) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%98%D1%81%D0%BF%D0%BE%D0%B2%D0%B5%D0%B4%D1%8C%20%28%D0%A2%D0%BE%D0%BB%D1%81%D1%82%D0%BE%D0%B9%29) |
| 37 | Дневник писателя | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%94%D0%BD%D0%B5%D0%B2%D0%BD%D0%B8%D0%BA%20%D0%BF%D0%B8%D1%81%D0%B0%D1%82%D0%B5%D0%BB%D1%8F) |
| 38 | Несвоевременные мысли | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9D%D0%B5%D1%81%D0%B2%D0%BE%D0%B5%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5%20%D0%BC%D1%8B%D1%81%D0%BB%D0%B8) |
| 39 | Кто виноват? (роман) | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9A%D1%82%D0%BE%20%D0%B2%D0%B8%D0%BD%D0%BE%D0%B2%D0%B0%D1%82%3F%20%28%D1%80%D0%BE%D0%BC%D0%B0%D0%BD%29) |
| 40 | Каштанка | [JSON](https://ru.wikipedia.org/w/api.php?action=query&prop=extracts%7Cinfo&explaintext=1&redirects=1&inprop=url&format=json&titles=%D0%9A%D0%B0%D1%88%D1%82%D0%B0%D0%BD%D0%BA%D0%B0) |

**3. Стоп-слова** — список `russian` из NLTK (https://github.com/nltk/nltk_data). Сохранён в `data/stopwords_ru.txt`.

### Разметка и словари

В `data/` лежат:

| Файл | Что это | Задание |
|---|---|---|
| `queries.json` | 30 запросов: Q01–Q10 для настройки, Q11–Q30 для проверки. Релевантность по шкале 0/1/2. У Q29 нет релевантных документов | 02, 05, 06, 08 |
| `small_corpus.json` | малый корпус с ручными позициями и проверками запросов | 03 |
| `thesaurus.json` | тезаурус | 06 |
| `duplicate_pairs.json` | размеченные пары; создаётся генератором | 07 |
| `keyphrases_gold.json` | ручной эталон ключевых фраз | 12 |

Запросы, разметка релевантности, тезаурус, эталон ключевых фраз и малый корпус составлены с помощью LLM **Claude (Anthropic)**.
