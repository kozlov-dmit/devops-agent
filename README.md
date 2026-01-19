# DevOps / Performance AI Agent (MVP)

AI-агент для анализа **инцидентов**, **логов**, **метрик** и **исходного кода** с целью поиска **узких мест и первопричин (RCA)** в backend-приложениях.

Проект использует:

* RAG по исходному коду (embeddings + vector search)
* эвристический rerank по сигналам инцидента
* LLM (GigaChat) для формирования структурированного отчёта

Подходит для:

* performance / latency инцидентов
* ошибок конфигурации
* проблем с пулом соединений, таймаутами, Envoy/Istio, TLS
* первичного RCA по логам и трейсам

---

## Архитектура (MVP)

```
Incident (JSON)
   │
   ▼
Signal extraction (exceptions, endpoints, stacktrace)
   │
   ▼
Vector search (FastEmbed + Qdrant local)
   │
   ▼
Heuristic rerank (path + signals)
   │
   ▼
Context pack (code + config)
   │
   ▼
GigaChat
   │
   ▼
Structured RCA report (JSON)
```

---

## Технологии

| Компонент       | Выбор                                  |
| --------------- | -------------------------------------- |
| Python          | **3.13**                               |
| Embeddings      | **FastEmbed (ONNX Runtime, CPU-only)** |
| Vector DB       | **Qdrant local mode** (без Docker)     |
| Payload storage | SQLite                                 |
| LLM             | **GigaChat (официальный SDK)**         |
| OS              | Windows / macOS                        |

---

## Ограничения и допущения

* ❌ GPU не используется
* ❌ Docker не используется
* ✅ Поддержка больших репозиториев (300k+ чанков)
* ✅ Работает в корпоративных средах с TLS

---

## Установка

### 1. Требования

* Python **3.13**
* `pip >= 24`
* Доступ к GigaChat API (Client ID + Client Secret)

---

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```txt
numpy==2.0.2
fastembed
qdrant-client
gigachat
```

---

## Настройка GigaChat (ВАЖНО)

### ❗️ `GIGACHAT_CREDENTIALS` — это **base64(ClientID:ClientSecret)**

Это **НЕ client secret** и **НЕ access token**.

### Как получить base64

#### Windows PowerShell

```powershell
$plain = "<CLIENT_ID>:<CLIENT_SECRET>"
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($plain))
```

#### macOS / Linux

```bash
printf "%s" "<CLIENT_ID>:<CLIENT_SECRET>" | base64
```

---

### Переменные окружения

#### Windows PowerShell

```powershell
$env:PYTHONPATH="src"
$env:GIGACHAT_CREDENTIALS="<base64>"
$env:GIGACHAT_SCOPE="GIGACHAT_API_PERS"
```

#### macOS

```bash
export PYTHONPATH=src
export GIGACHAT_CREDENTIALS="<base64>"
export GIGACHAT_SCOPE=GIGACHAT_API_PERS
```

---

## Структура проекта

```
src/
 └─ agent/
    ├─ cli.py                # CLI (index / run / analyze)
    ├─ indexer.py            # Сканирование репозитория
    ├─ chunking.py           # Разбиение кода на чанки
    ├─ embeddings_fastembed.py
    ├─ vectordb_qdrant.py    # Qdrant local mode
    ├─ retriever.py          # Vector search + rerank
    ├─ signals.py            # Извлечение сигналов инцидента
    ├─ analyzer.py           # LLM-анализ
    ├─ llm_client.py         # GigaChat wrapper
    ├─ prompts.py            # Русский системный промпт
    ├─ report_schema.py      # Валидация JSON-отчёта
    ├─ store_sqlite.py       # Payload store
    └─ config.py
```

---

## Индексация репозитория

```bash
python -m agent.cli index \
  --repo /path/to/repository \
  --out ./data/index/service1
```

### Что происходит:

* исключаются `src/test/**`, `target/`, `node_modules/` и т.д.
* код режется на чанки (по строкам)
* чанки сохраняются:

  * текст → SQLite
  * embeddings → Qdrant local

### Рекомендуемые env для больших репозиториев

```bash
export CHUNK_MAX_LINES=80
export CHUNK_OVERLAP=15
export EMBED_BATCH_SIZE=16
export INDEX_BATCH_SIZE=64
export EMBED_MODEL=jinaai/jina-embeddings-v2-small-code
```

---

## Проверка retrieval (без LLM)

```bash
python -m agent.cli run \
  --index ./data/index/service1 \
  --incident ./data/incidents/incident.sample.json \
  --topk 12
```

Вы увидите:

```
[score=2.51 base=0.21 rr=+2.30] src/.../EnvoyInspectorService.java:196-275
```

* `base` — косинусная близость embedding
* `rr` — вклад эвристик (stacktrace, keywords, path)

---

## Полный анализ инцидента (RAG + GigaChat)

```bash
python -m agent.cli analyze \
  --index ./data/index/service1 \
  --incident ./data/incidents/incident.sample.json \
  --out-report ./data/reports/report.json
```

Результат: **валидный JSON-отчёт**, например:

```json
{
  "summary": "...",
  "classification": {"type": "performance", "confidence": 0.82},
  "hypotheses": [...],
  "hotspots": [...],
  "checks": [...],
  "missing_data": [...]
}
```

---

## Формат incident.json (пример)

```json
{
  "service": "istio-route-explorer",
  "symptoms": {
    "latency_p99_ms": 4200,
    "error_rate": "3%"
  },
  "logs": [
    "TimeoutException: request timed out",
    "HikariPool-1 - Connection is not available"
  ],
  "traces": {
    "top_spans": [
      "EnvoyInspectorService.inspectRoutes",
      "RouteExplorer.findRoutes"
    ]
  }
}
```

---

## Принципы, заложенные в агент

* ❌ Никаких галлюцинаций
* ✅ Только код и конфигурация из индекса
* ✅ Каждая гипотеза = доказательства
* ✅ Если данных мало — агент прямо это говорит

---

## Roadmap

* [ ] Авто-классификация инцидентов (latency / availability / config / resource)
* [ ] Envoy/Istio чек-листы (503 UF, upstream reset, TLS)
* [ ] Diff-анализ (два инцидента / два коммита)
* [ ] Интеграция с Jira / Confluence
* [ ] Streaming-анализ логов

---

## Статус

**MVP стабилен** и пригоден для реального RCA в backend / platform командах.
