"""
Glasseye CLI - User-friendly command-line interface

Usage:
    glasseye build --size tiny --device cpu
    glasseye train --dataset fineweb-edu --steps 1000
    glasseye serve --model artifacts/model.pt --port 5000
    glasseye monitor --log-dir artifacts/runs
"""

import click
import logging
from pathlib import Path

from mythos_glasseye.registry import AgentRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="glasseye")
def cli():
    """Mythos Glasseye - Agent-driven AI pipeline"""
    pass


@cli.command()
@click.option("--size", type=click.Choice(["tiny", "1b", "3b"]), default="tiny", help="Model size")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", help="Device to build for")
@click.option("--output", "-o", type=click.Path(), help="Output path for checkpoint")
def build(size, device, output):
    """Build a Mythos Glasseye model checkpoint"""
    click.echo(f"🔨 Building {size} model for {device}...")
    
    registry = AgentRegistry()
    builder = registry.get("model_builder")
    
    result = builder.run(
        model_name=f"glasseye-{size}",
        config={"size": size},
        device=device
    )
    
    if result.success:
        click.echo(f"✅ Model built successfully!")
        click.echo(f"📦 Checkpoint: {result.output_path}")
        if result.metadata:
            click.echo(f"📊 Metadata: {result.metadata}")
    else:
        click.echo(f"❌ Build failed: {result.metadata.get('error')}")
        raise click.Abort()


@cli.command()
@click.option("--dataset", default="HuggingFaceFW/fineweb-edu", help="Dataset source")
@click.option("--steps", type=int, default=1000, help="Training steps")
@click.option("--batch-size", type=int, default=4, help="Batch size")
@click.option("--gpus", type=int, default=1, help="Number of GPUs")
@click.option("--ddp", is_flag=True, help="Use DDP for multi-GPU")
def train(dataset, steps, batch_size, gpus, ddp):
    """Train/fine-tune a Mythos Glasseye model"""
    click.echo(f"🚂 Starting training...")
    click.echo(f"  Dataset: {dataset}")
    click.echo(f"  Steps: {steps}")
    click.echo(f"  Batch size: {batch_size}")
    
    registry = AgentRegistry()
    trainer = registry.get("trainer")
    
    result = trainer.run(
        max_steps=steps,
        batch_size=batch_size,
        num_gpus=gpus,
        use_ddp=ddp
    )
    
    if result.success:
        click.echo(f"✅ Training started!")
        click.echo(f"📊 Status: {result.metadata.get('status', 'running')}")
    else:
        click.echo(f"❌ Training failed: {result.metadata.get('error')}")
        raise click.Abort()


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Model checkpoint path")
@click.option("--port", "-p", type=int, default=5000, help="Port to serve on")
@click.option("--mode", type=click.Choice(["docker", "fastapi"]), default="fastapi", help="Deployment mode")
def serve(model, port, mode):
    """Deploy inference endpoint"""
    click.echo(f"🚀 Deploying model...")
    click.echo(f"  Model: {model}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Mode: {mode}")
    
    registry = AgentRegistry()
    deployer = registry.get("deployer")
    
    result = deployer.run(
        model_path=model,
        port=port,
        mode=mode
    )
    
    if result.success:
        click.echo(f"✅ Deployment successful!")
        click.echo(f"🌐 Endpoint: {result.metadata.get('endpoint')}")
        if mode == "fastapi":
            click.echo(f"\n📝 To start server, run:")
            click.echo(f"   {result.metadata.get('command')}")
    else:
        click.echo(f"❌ Deployment failed: {result.metadata.get('error')}")
        raise click.Abort()


@cli.command()
@click.option("--log-dir", default="artifacts/runs", help="TensorBoard log directory")
@click.option("--port", type=int, default=6006, help="Port for TensorBoard")
def monitor(log_dir, port):
    """Start TensorBoard monitoring"""
    click.echo(f"📊 Starting TensorBoard...")
    
    registry = AgentRegistry()
    monitor_agent = registry.get("monitor")
    
    result = monitor_agent.run(
        log_dir=log_dir,
        port=port
    )
    
    if result.success:
        click.echo(f"✅ TensorBoard running!")
        click.echo(f"🌐 URL: {result.metadata.get('url')}")
        click.echo(f"📂 Logs: {result.metadata.get('log_dir')}")
        click.echo(f"\n⚠️  Keep this terminal open to keep TensorBoard running")
    else:
        click.echo(f"❌ Monitor failed: {result.metadata.get('error')}")
        raise click.Abort()


@cli.command()
@click.option("--source", default="HuggingFaceFW/fineweb-edu", help="Dataset source")
@click.option("--subset", default="sample-10BT", help="Dataset subset")
def prepare(source, subset):
    """Prepare dataset for training"""
    click.echo(f"📦 Preparing dataset...")
    click.echo(f"  Source: {source}")
    click.echo(f"  Subset: {subset}")
    
    registry = AgentRegistry()
    preparer = registry.get("data_preparer")
    
    result = preparer.run(
        source=source,
        subset=subset
    )
    
    if result.success:
        click.echo(f"✅ Dataset ready!")
        click.echo(f"📊 Features: {result.metadata.get('features', 'N/A')}")
    else:
        click.echo(f"❌ Preparation failed: {result.metadata.get('error')}")
        raise click.Abort()


@cli.command()
def list_agents():
    """List all available agents"""
    click.echo("📋 Available agents:\n")
    
    registry = AgentRegistry()
    agents = registry.list_agents()
    
    for name, agent_cls in agents.items():
        doc = agent_cls.__doc__ or "No description"
        doc_first_line = doc.split("\n")[0].strip()
        click.echo(f"  • {name}: {doc_first_line}")


if __name__ == "__main__":
    cli()
