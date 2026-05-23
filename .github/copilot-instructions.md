# Copilot Instructions for OpenMythos

## Project Overview

OpenMythos is an open-source theoretical implementation of the Claude Mythos Recurrent-Depth Transformer (RDT) architecture. The project implements a three-stage transformer with Prelude, Recurrent Block, and Coda stages, supporting both Multi-Latent Attention (MLA) and Grouped Query Attention (GQA).

## Build & Test Commands

**Install dependencies:**
```bash
pip install -e .
# Or with flash attention support:
pip install -e .[flash]
```

**Run tests:**
```bash
pytest
```

**Code formatting & linting:**
```bash
# Format code
black .

# Lint code
ruff check .
```

**Train models:**
```bash
# 3B model on FineWeb-Edu
python training/3b_fine_web_edu.py
```

## Architecture

- **Three-stage RDT**: Prelude (transformer blocks) → Recurrent Block (looped up to `max_loop_iters`) → Coda (final transformer blocks)
- **Attention mechanisms**: Switchable between MLA and GQA
- **Feed-forward**: Sparse MoE with routed and shared experts
- **Model scales**: Pre-configured variants from 1B to 1T parameters
- **Key modules**:
  - `open_mythos/main.py` - Core OpenMythos model and configuration
  - `mythos_glasseye/` - Extended functionality and CLI
  - `training/` - Training scripts and configurations

## Critical: File Operations & Data Preservation

⚠️ **PRESERVATION-FIRST POLICY** ⚠️

When working with files, directories, or disk operations:

1. **NEVER delete, remove, or overwrite files unless explicitly requested multiple times with clear confirmation**
2. **Default to preservation** - move, copy, or organize instead of deleting
3. **When organizing storage:**
   - Move files to organized locations (never delete)
   - Create backups before any risky operations
   - Preserve directory structure and relationships
   - Ask for confirmation before any destructive operation
4. **If disk space is tight:**
   - Suggest moving to external drives (e.g., `/mnt/backup-hdd/`)
   - Recommend compression or archival
   - Identify safe-to-remove cache directories
   - NEVER auto-delete without explicit permission
5. **Storage layout awareness:**
   - External drives mounted at `/mnt/backup` and `/mnt/backup-hdd`
   - Organized structure: `projects/`, `tools/`, `security-research/`, `development/`
   - Main system drive frequently runs at high capacity

**Examples of acceptable actions:**
- ✅ Moving old projects to `/mnt/backup-hdd/organized/projects/`
- ✅ Creating symlinks to relocated cache directories
- ✅ Suggesting "safe to delete" cache files with explanation
- ✅ Compressing large directories into archives

**Examples of unacceptable actions without explicit confirmation:**
- ❌ Deleting "duplicate" files
- ❌ Removing old versions or backups
- ❌ Cleaning up "unnecessary" directories
- ❌ Auto-removing cache directories

When in doubt, preserve first and ask questions.

## Key Conventions

- **Python 3.10+** required
- **PyTorch 2.11.0** for core deep learning operations
- **Poetry** for dependency management
- **Line length**: 88 characters (Black/Ruff standard)
- **Type hints**: Use Pydantic models for configuration
- **Logging**: Use loguru for structured logging
- **Testing**: pytest for all test cases
- **Model configs**: Defined in `MythosConfig` dataclass with validation

## CLI Tools

- `openmythos` - Main CLI for model operations
- `glasseye` - Extended functionality CLI for mythos_glasseye module

## Storage & Disk Layout

**System Architecture:**
- **Main SSD** (`/`): 221GB total, frequently at 75-100% capacity
  - Active development work stays here for fast access
  - Keep only current/active projects on main drive
- **Backup HDD** (`/mnt/backup-hdd`): 364GB, ~89% used
  - Primary organized storage location
  - Structure: `organized/projects/`, `organized/tools/`, `organized/security-research/`, `organized/development/`
  - Master file index at `/mnt/backup-hdd/MASTER_INDEX/`
- **Backup Drive** (`/mnt/backup`): 932GB, ~99% used (nearly full)
  - Large development environments (copilottesting, spark)
  - Structure: `organized/development/`, `organized/jetbrains-backup/`, `organized/archives/`

**Organization Philosophy:**
- Categorize by purpose: projects, tools, security-research, development, archives
- Maintain discoverability through master index and search tools
- Preserve all files - move to appropriate backup location instead of deleting
- Use symlinks for relocated caches and large directories

**Quick Access:**
- Search files: `/mnt/backup-hdd/MASTER_INDEX/search_files.sh <keyword>`
- Find by type: `/mnt/backup-hdd/MASTER_INDEX/find_by_type.sh <extension>`
- Browse categories: `/mnt/backup-hdd/MASTER_INDEX/by_category.sh <category>`

## Communication Style

**User Preferences:**
- Casual, energetic, informal communication style
- Louisiana/NOLA slang and metaphors are common:
  - "sha" (Louisiana term of endearment)
  - "couyon" (Cajun French for "fool")
  - "Who Dat" (New Orleans Saints reference)
  - Project metaphors: "gremlins" (fine-tuning), "goblins" (live builds), "trash pandas/racoons" (optimization loops)
- Match the user's energy level and informal tone
- Technical accuracy matters, but overly formal responses feel disconnected
- Understand slang terms as project concepts, not confusion

## Security Research Workflow

**User Profile:**
- Active bug bounty hunter and security researcher
- Testing frameworks located in `/home/x/jetbrains/jettests/`
- Focus areas: IDOR, SQL injection, authentication bypass, API security
- Methodical approach: reconnaissance → automated testing → manual verification → reporting

**Key Locations:**
- Testing scripts: `/home/x/jetbrains/jettests/portswigger-net-pentest/`, `/home/x/jetbrains/jettests/superhuman/`
- Security tools: `/mnt/backup-hdd/organized/tools/` (SecLists, glasseye versions)
- Reports & results: `/mnt/backup-hdd/organized/security-research/`

**Workflow Support:**
- Help analyze bug bounty program scopes
- Assist with test automation and script creation
- Support vulnerability research and payload development
- Understand in-scope vs out-of-scope boundaries
- Never suggest illegal activities (only authorized testing on scoped targets)

## Project Ecosystem

**Main Repository:** ArtificialAutism (OpenMythos)
- Active ML/AI research project focused on Recurrent-Depth Transformers
- 7GB, most actively developed
- Located at `/home/x/ArtificialAutism/`

**Related Projects:**
- **hancock-glasseye** (11GB, archived) - AI factory system combining Hancock orchestration with Glasseye vision capabilities
- **glasseye** (2.7GB, archived) - Computer vision and ML observation system
- **Hancock** (2.6GB, archived) - AI orchestration and deployment system
- **peachtree** - Related ML infrastructure project
- **openmythos** - Multiple versions across directories (main work in ArtificialAutism)

**Project Relationships:**
- OpenMythos = Core transformer architecture research
- Glasseye = Vision/observation capabilities
- Hancock = Orchestration/deployment layer
- Integration point: mythos_glasseye module combines OpenMythos with vision capabilities

**Development Stack:**
- PyTorch 2.11 for deep learning
- Transformers library for model interfaces
- FastAPI for serving models
- CUDA/GPU acceleration for training
- Google Colab for free GPU access when local GPU unavailable
