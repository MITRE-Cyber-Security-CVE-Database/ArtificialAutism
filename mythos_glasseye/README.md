# Mythos Glasseye - Agent-Driven AI Pipeline 🤖

> **Modular, agent-driven architecture for the OpenMythos Recurrent-Depth Transformer**

Mythos Glasseye provides a complete workflow automation system for building, training, and deploying OpenMythos models through composable agents and YAML-defined pipelines.

## ✨ Features

- 🔧 **Modular Agent System** - Composable agents for each pipeline stage
- 🤖 **Gemini Integration** - Collaborative AI building with Google Gemini
- 🚀 **FastAPI + Docker Deployment** - Production-ready inference endpoints
- 📊 **TensorBoard Monitoring** - Real-time training visualization
- 🎯 **User-Friendly CLI** - Simple commands for complex workflows
- 📝 **YAML Pipelines** - Declarative workflow definitions

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MITRE-Cyber-Security-CVE-Database/ArtificialAutism
cd ArtificialAutism

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or with Poetry
poetry install
```

### CLI Commands

```bash
# Build a model checkpoint
glasseye build --size tiny --device cpu

# Prepare dataset
glasseye prepare --source HuggingFaceFW/fineweb-edu

# Train model
glasseye train --steps 1000 --batch-size 4

# Deploy inference endpoint
glasseye serve --model artifacts/model.pt --port 5000

# Start TensorBoard monitoring
glasseye monitor --log-dir artifacts/runs

# List available agents
glasseye list-agents
```

## 📦 Available Agents

| Agent | Purpose | Key Methods |
|-------|---------|-------------|
| **ModelBuilderAgent** | Build model checkpoints (tiny/3B) | `build(model_name, config, device)` |
| **DataPreparerAgent** | Prepare FineWeb-Edu dataset | `prepare(source, subset, split)` |
| **TrainerAgent** | Train/fine-tune with LoRA | `train(max_steps, batch_size)` |
| **DeployerAgent** | Deploy FastAPI + Docker | `deploy(model_path, port, mode)` |
| **MonitorAgent** | TensorBoard monitoring | `start(log_dir, port)` |

## 🔧 Programmatic Usage

### Using Individual Agents

```python
from mythos_glasseye.registry import AgentRegistry

# Initialize registry
registry = AgentRegistry()

# Build a model
builder = registry.get("model_builder")
result = builder.run(
    model_name="glasseye-tiny",
    config={"size": "tiny"},
    device="cpu"
)

if result.success:
    print(f"Model built: {result.output_path}")
```

### Running YAML Pipelines

```python
from mythos_glasseye.pipeline_runner import run_pipeline_file

# Execute full training pipeline
summary = run_pipeline_file("mythos_glasseye/pipelines/training.yaml")

print(f"Success rate: {summary['success_rate']:.1%}")
```

### Gemini Collaboration

```python
from mythos_glasseye.gemini_integration import GeminiCollaborator

# Initialize Gemini
collaborator = GeminiCollaborator(api_key="your-key")

# Generate code
code = collaborator.generate_code(
    prompt="Create a data augmentation agent",
    language="python"
)

# Review code
review = collaborator.review_code(code, focus="security")
print(f"Review score: {review['score']}/10")
```

## 📋 Pipeline Definitions

### Training Pipeline (`pipelines/training.yaml`)

```yaml
steps:
  - name: prepare_data
    agent: data_preparer
    args:
      source: "HuggingFaceFW/fineweb-edu"
      subset: "sample-10BT"
  
  - name: train
    agent: trainer
    args:
      max_steps: 1000
      batch_size: 4
```

### Deployment Pipeline (`pipelines/deployment.yaml`)

```yaml
steps:
  - name: deploy_fastapi
    agent: deployer
    args:
      model_path: "artifacts/model.pt"
      port: 5000
      mode: "fastapi"
  
  - name: start_monitoring
    agent: monitor
    args:
      log_dir: "artifacts/runs"
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_agents.py -v

# Run with coverage
pytest tests/ --cov=mythos_glasseye
```

## 🤝 Gemini Integration

Set up Google Gemini for collaborative building:

```bash
# Set API key
export GEMINI_API_KEY="your-api-key-here"

# Test integration
python mythos_glasseye/gemini_integration.py
```

## 🚀 Deployment Options

### FastAPI (Quick Testing)

```bash
# Deploy with FastAPI
glasseye serve --model artifacts/model.pt --mode fastapi

# Start server
uvicorn artifacts.inference_server:app --port 5000
```

### Docker (Production)

```bash
# Deploy with Docker
glasseye serve --model artifacts/model.pt --mode docker

# Container runs automatically at http://localhost:5000
```

### API Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [1, 2, 3, 4, 5],
    "max_new_tokens": 20,
    "n_loops": 4
  }'
```

## 📊 Monitoring

```bash
# Start TensorBoard
glasseye monitor

# Open browser to http://localhost:6006
```

## 🏗️ Architecture

```
mythos_glasseye/
├─ agents/           # All sub-agents
│   ├─ base.py       # BaseAgent class
│   ├─ model_builder.py
│   ├─ data_preparer.py
│   ├─ trainer.py
│   ├─ deployer.py
│   └─ monitor.py
├─ pipelines/        # YAML workflow definitions
│   ├─ build_checkpoint.yaml
│   ├─ training.yaml
│   └─ deployment.yaml
├─ registry.py       # Agent discovery & registration
├─ pipeline_runner.py # YAML pipeline executor
├─ gemini_integration.py # Gemini API wrapper
└─ cli.py           # User-friendly CLI
```

## 🔧 Configuration

### Environment Variables

```bash
# Gemini API
export GEMINI_API_KEY="your-key"

# OpenMythos
export OPENMYTHOS_DEVICE="cpu"  # or "cuda"
export OPENMYTHOS_THREADS="4"
```

### Azure Deployment

```bash
# Install Azure ML extension
az extension add --name ml

# Deploy to Azure AI Foundry
# (See deployment docs for full instructions)
```

## 📚 Documentation

- [Implementation Plan](files/implementation_plan.md)
- [OpenMythos Architecture](docs/open_mythos.md)
- [Local CPU Run Guide](docs/local_cpu_run.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- OpenMythos architecture based on Claude Mythos research
- Built with PyTorch, Hugging Face Transformers, FastAPI
- Gemini integration by Google AI

---

**Made with ❤️ by the Mythos Glasseye Team**
