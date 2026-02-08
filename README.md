
# IMDB Sentiment Analysis

Sentiment analysis on IMDB movie reviews comparing two pre-trained transformer models: DistilBERT (fast) and RoBERTa (accurate). Includes a benchmarking pipeline, FastAPI endpoint and Gradio demo.

## Results

Benchmarked on 100 IMDB reviews (42 positive, 58 negative):

| Model | Accuracy | F1 Score | Avg Time | Throughput |
|-------|----------|----------|----------|------------|
| RoBERTa | 95.0% | 0.949 | 363 ms | 2.8 pred/s |
| DistilBERT | 88.0% | 0.878 | 196 ms | 5.1 pred/s |

Both models are fine-tuned on SST-2, so the comparison isolates model architecture and size rather than training data differences. RoBERTa is more accurate but ~1.9x slower.

## Setup

Requires Python 3.11+ and [UV](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/M-hAkIr/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Usage

### Benchmarking

```bash
python src/main.py
```

Outputs prediction CSVs in `data/`, benchmark reports (JSON + Markdown) and confusion matrices in `outputs/`.

### API

I used FastAPI because it auto-generates OpenAPI docs and works well with Pydantic for request validation -- the `/docs` endpoint is handy for testing during development.

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Docs at http://localhost:8000/docs

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was fantastic!", "model": "distilbert"}'
```

```json
{
  "text": "This movie was fantastic!",
  "sentiment": "positive",
  "confidence": 0.9987,
  "model_used": "distilbert",
  "inference_time_ms": 45.32
}
```

### Demo

Gradio fits here since it's part of the Hugging Face ecosystem and takes just a few lines to set up a model demo.

```bash
python demo/app.py
```

Opens at http://localhost:7860 where you can type reviews and pick a model.

### Docker

```bash
docker build -t imdb-sentiment .
docker run -p 8000:8000 imdb-sentiment
```

## Models

- **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`) -- 66M params, fine-tuned on SST-2, smaller and faster
- **RoBERTa** (`textattack/roberta-base-SST-2`) -- 125M params, fine-tuned on SST-2, larger and more accurate

## Dataset

- 100 IMDB movie reviews (semicolon-delimited CSV, ISO-8859-1 encoded)
- Ground truth labels: "positive" / "negative"
- Preprocessing: HTML tag removal and whitespace normalization
