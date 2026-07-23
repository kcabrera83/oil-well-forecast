# Deployment Guide - Oil Well Production Forecast

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/train.py

EXPOSE 5002

CMD ["python", "app.py"]
```

### Build and Run
```bash
docker build -t oil-well-forecast .
docker run -p 5002:5002 oil-well-forecast
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5002:5002"
    environment:
      - FLASK_DEBUG=0
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_DEBUG | Enable debug mode | 1 |
| PORT | Server port | 5002 |
| HOST | Server host | 0.0.0.0 |

## Manual Deployment

### Prerequisites
- Python 3.8+
- pip

### Steps
```bash
git clone https://github.com/kcabrera83/oil-well-forecast.git
cd oil-well-forecast
pip install -r requirements.txt
python scripts/train.py
python scripts/evaluate.py  # optional
python app.py
```

## Production Considerations

### Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5002 app:app
```

### Security
- Set `DEBUG=False` in production
- Use HTTPS with a reverse proxy
- Add authentication for API endpoints if exposed publicly
- Validate input ranges before model inference

### Monitoring
- Monitor `/api/health` for uptime checks
- Log all prediction and forecast requests
- Track anomaly detection rates for system health

### Performance
- Models are lazy-loaded on first request
- Pre-load models at startup for faster first response
- Use connection pooling for any external integrations

## CI/CD
GitHub Actions workflow runs on every push:
1. Model training
2. Predictions
3. Full evaluation
4. API tests (5 endpoints)

## API Self-Documentation
Access OpenAPI docs at: `http://localhost:5002/api/docs`
