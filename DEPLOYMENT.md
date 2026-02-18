# Deployment Blueprint for LLM Evaluation Framework

## Overview

This guide provides step-by-step instructions for deploying the LLM Evaluation Framework in an enterprise environment.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- Access to LLM APIs (OpenAI, Anthropic, etc.) or local models

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mahesh-digi/llm_evaluation.git
cd llm_evaluation
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# For development (optional)
pip install -e ".[dev]"
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (if using external judge models)
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/evaluation.log

# Cost Control
MAX_COST_PER_EVALUATION=10.0

# Database (optional, for storing results)
DATABASE_URL=postgresql://user:pass@localhost/llm_eval
```

### 2. Judge Configuration

Create a configuration file `config/judges.yaml`:

```yaml
judges:
  - model_name: "gpt-4"
    model_type: "openai"
    temperature: 0.0
    api_key_env: "OPENAI_API_KEY"
    
  - model_name: "claude-3-opus"
    model_type: "anthropic"
    temperature: 0.0
    api_key_env: "ANTHROPIC_API_KEY"
    
  - model_name: "gemini-ultra"
    model_type: "custom"
    temperature: 0.0
    endpoint: "https://your-endpoint.com/v1"
    api_key_env: "GOOGLE_API_KEY"

evaluation:
  min_judges: 3
  evaluation_method: "pairwise"
  scoring_method: "bradley_terry"
  
bias_mitigation:
  randomize_order: true
  use_blind_evaluation: true
  position_bias_correction: true
  length_normalization: true
  num_permutations: 2

statistical_validation:
  compute_confidence_intervals: true
  confidence_level: 0.95
  num_bootstrap_samples: 1000
  compute_inter_judge_agreement: true
  min_judge_agreement: 0.5
```

## Deployment Scenarios

### Scenario 1: Local Development/Testing

```python
# test_local.py
from llm_eval import EvaluationFramework, EvaluationConfig
from llm_eval.core.evaluation_config import JudgeConfig

# Use mock judges for testing
judges = [
    JudgeConfig(model_name=f"test_judge_{i}", model_type="local")
    for i in range(3)
]

config = EvaluationConfig(
    judges=judges,
    evaluation_method="pairwise",
    task_type="qa",
)

framework = EvaluationFramework(config)
# Run evaluations...
```

### Scenario 2: Single Server Deployment

```bash
# 1. Set up systemd service
sudo nano /etc/systemd/system/llm-eval.service

# Content:
[Unit]
Description=LLM Evaluation Service
After=network.target

[Service]
Type=simple
User=llmeval
WorkingDirectory=/opt/llm_evaluation
Environment="PATH=/opt/llm_evaluation/venv/bin"
ExecStart=/opt/llm_evaluation/venv/bin/python -m llm_eval.server
Restart=always

[Install]
WantedBy=multi-user.target

# 2. Enable and start
sudo systemctl enable llm-eval
sudo systemctl start llm-eval
```

### Scenario 3: Container Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Set environment
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run
CMD ["python", "-m", "llm_eval.server"]
```

Build and run:

```bash
docker build -t llm-eval:latest .
docker run -d \
  --name llm-eval \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  -v $(pwd)/logs:/app/logs \
  -p 8000:8000 \
  llm-eval:latest
```

### Scenario 4: Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-eval
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-eval
  template:
    metadata:
      labels:
        app: llm-eval
    spec:
      containers:
      - name: llm-eval
        image: your-registry/llm-eval:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-eval-secrets
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-eval-secrets
              key: anthropic-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: llm-eval
spec:
  selector:
    app: llm-eval
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

Deploy:

```bash
kubectl apply -f k8s/deployment.yaml
```

## Integration Patterns

### 1. CI/CD Integration

```yaml
# .github/workflows/evaluate.yml
name: Model Evaluation

on:
  pull_request:
    paths:
      - 'models/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      
      - name: Run evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python scripts/evaluate_models.py
      
      - name: Post results
        run: |
          python scripts/post_results.py
```

### 2. API Service

```python
# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_eval import EvaluationFramework, EvaluationConfig

app = FastAPI()

@app.post("/evaluate")
async def evaluate(request: EvaluationRequest):
    try:
        # Parse request
        framework = EvaluationFramework(request.config)
        result = framework.evaluate_single(request.prompt, request.responses)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn server:app --host 0.0.0.0 --port 8000
```

### 3. Batch Processing

```python
# batch_processor.py
from llm_eval import EvaluationFramework
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    def __init__(self, config):
        self.framework = EvaluationFramework(config)
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def process_batch(self, prompts, responses_by_prompt):
        return self.framework.evaluate_batch(prompts, responses_by_prompt)
    
    def process_parallel(self, tasks):
        futures = [
            self.executor.submit(self.process_batch, task.prompts, task.responses)
            for task in tasks
        ]
        return [f.result() for f in futures]
```

## Monitoring and Observability

### 1. Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Metrics Collection

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
evaluations_total = Counter('evaluations_total', 'Total evaluations')
evaluation_duration = Histogram('evaluation_duration_seconds', 'Evaluation duration')
evaluation_cost = Histogram('evaluation_cost_dollars', 'Evaluation cost')

# Use in code
@evaluation_duration.time()
def run_evaluation():
    result = framework.evaluate_single(prompt, responses)
    evaluations_total.inc()
    evaluation_cost.observe(result.total_cost)
    return result

# Start metrics server
start_http_server(9090)
```

### 3. Alerts

```yaml
# prometheus/alerts.yml
groups:
  - name: llm_eval_alerts
    interval: 30s
    rules:
      - alert: HighEvaluationCost
        expr: rate(evaluation_cost_dollars_sum[5m]) > 10
        annotations:
          summary: "High evaluation costs detected"
      
      - alert: LowJudgeAgreement
        expr: judge_agreement < 0.5
        annotations:
          summary: "Low inter-judge agreement"
```

## Security Best Practices

1. **API Key Management**
   - Use environment variables or secret managers
   - Rotate keys regularly
   - Never commit keys to version control

2. **Data Privacy**
   - Sanitize PII from evaluation data
   - Use encryption for sensitive data
   - Implement access controls

3. **Rate Limiting**
   ```python
   from ratelimit import limits, sleep_and_retry
   
   @sleep_and_retry
   @limits(calls=100, period=60)
   def call_judge_api():
       # API call
       pass
   ```

4. **Input Validation**
   ```python
   from pydantic import BaseModel, validator
   
   class EvaluationRequest(BaseModel):
       prompt: str
       responses: List[str]
       
       @validator('prompt')
       def validate_prompt(cls, v):
           if len(v) > 10000:
               raise ValueError('Prompt too long')
           return v
   ```

## Performance Optimization

### 1. Caching

```python
from functools import lru_cache
import hashlib

def cache_key(prompt, response):
    content = f"{prompt.text}:{response.response_text}"
    return hashlib.md5(content.encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_judgment(cache_key):
    return judge.evaluate(prompt, response)
```

### 2. Parallelization

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    judgments = list(executor.map(
        lambda judge: judge.judge_pairwise(prompt, resp_a, resp_b),
        judges
    ))
```

### 3. Batch API Calls

```python
def batch_evaluate(prompts, responses, batch_size=10):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_results = framework.evaluate_batch(batch, responses)
        results.extend(batch_results)
    return results
```

## Troubleshooting

### Common Issues

1. **Module Not Found**
   ```bash
   pip install -e .
   ```

2. **API Rate Limits**
   - Implement exponential backoff
   - Use rate limiting
   - Consider caching

3. **Low Judge Agreement**
   - Review prompts for clarity
   - Check judge configurations
   - Add more judges
   - Review task type

4. **High Costs**
   - Enable caching
   - Use cheaper judges for screening
   - Implement sampling
   - Set cost limits

## Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=llm_eval tests/

# Run specific test
pytest tests/test_evaluation_framework.py::test_pairwise_evaluation
```

## Maintenance

### Regular Tasks

1. **Update dependencies**
   ```bash
   pip list --outdated
   pip install --upgrade package_name
   ```

2. **Clean up logs**
   ```bash
   find logs/ -name "*.log" -mtime +30 -delete
   ```

3. **Monitor costs**
   ```bash
   python scripts/cost_report.py --last-30-days
   ```

4. **Backup configurations**
   ```bash
   tar -czf config_backup_$(date +%Y%m%d).tar.gz config/
   ```

## Support

For issues or questions:
- GitHub Issues: https://github.com/Mahesh-digi/llm_evaluation/issues
- Documentation: See DOCUMENTATION.md
- Examples: See llm_eval/examples/

## License

MIT License - See LICENSE file for details.
