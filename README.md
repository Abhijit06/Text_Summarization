# Python Summarization Service

FastAPI microservice that exposes Hugging Face summarization models as a REST API.

## Models supported
| Key    | Model                        | Best for            |
|--------|------------------------------|---------------------|
| `bart` | facebook/bart-large-cnn      | News, articles, PDFs |
| `t5`   | t5-small                     | Lightweight / faster |

## Setup & run locally

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run server
uvicorn main:app --reload --port 8000
```

## API

### POST /summarize
```json
{
  "text": "Your long text here...",
  "length": "medium",   // short | medium | long
  "model": "bart"       // bart | t5
}
```

**Response:**
```json
{
  "summary": "Summarized text...",
  "original_length": 1500,
  "summary_length": 180,
  "model_used": "facebook/bart-large-cnn",
  "compression_ratio": 0.12
}
```

### GET /health
Returns service status.

## Run with Docker

```bash
docker build -t summarizer-python .
docker run -p 8000:8000 summarizer-python
```

## Run tests

```bash
pytest tests/ -v
```

## Interactive docs
Visit `http://localhost:8000/docs` for Swagger UI after starting the server.
