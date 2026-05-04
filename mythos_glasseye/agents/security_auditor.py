import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from .base import BaseAgent, Result

logger = logging.getLogger(__name__)

class SecurityAuditorAgent(BaseAgent):
    """Defensive cybersecurity agent.
    
    Performs Static Application Security Testing (SAST) and 
    dependency vulnerability checks.
    """

    class Config(BaseAgent.Config):
        target_dir: str = "."
        check_dependencies: bool = True
        run_sast: bool = True

    def run(self, **kwargs) -> Result:
        cfg = self.validate(**kwargs)
        target_path = Path(cfg.target_dir)
        
        if not target_path.exists():
            return Result(success=False, metadata={"error": f"Target directory not found: {target_path}"})
            
        logger.info(f"Starting security audit for {target_path}...")
        metadata: Dict[str, Any] = {"status": "audit_completed"}
        all_passed = True
        
        # 1. Dependency Check (Simulated for MVP, would use pip-audit or similar)
        if cfg.check_dependencies:
            req_file = target_path / "requirements.txt"
            if req_file.exists():
                logger.info(f"Checking dependencies in {req_file}...")
                # In a real agent, we would run `pip-audit -r requirements.txt`
                # Here we do a basic heuristic check for demonstration.
                vulnerable_packages = []
                with open(req_file, "r") as f:
                    content = f.read().lower()
                    # Example: alert if using a known old/vulnerable package version
                    if "requests==2.0.0" in content: 
                        vulnerable_packages.append("requests==2.0.0 (CVE-YYYY-XXXX)")
                
                if vulnerable_packages:
                    metadata["dependency_vulnerabilities"] = vulnerable_packages
                    all_passed = False
                    logger.warning("Vulnerable dependencies found!")
                else:
                    metadata["dependency_vulnerabilities"] = "None detected."
                    logger.info("Dependencies look secure.")
            else:
                logger.info("No requirements.txt found. Skipping dependency check.")

        # 2. Static Application Security Testing (SAST)
        if cfg.run_sast:
            logger.info(f"Running SAST on {target_path}...")
            # In a real agent, we'd run `bandit -r .` or `semgrep`
            # For the MVP, we run a simulated check for hardcoded secrets or 'eval'
            findings = []
            
            # Simple python file scan
            for py_file in target_path.rglob("*.py"):
                if "venv" in py_file.parts:
                    continue
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if "eval(" in line:
                                findings.append(f"{py_file.name}:{i+1} - Use of dangerous 'eval()' detected.")
                            if "API_KEY" in line and "=" in line and ("'" in line or '"' in line):
                                findings.append(f"{py_file.name}:{i+1} - Potential hardcoded API key detected.")
                except Exception:
                    pass
            
            if findings:
                metadata["sast_findings"] = findings
                all_passed = False
                logger.warning(f"SAST found {len(findings)} potential issues.")
            else:
                metadata["sast_findings"] = "Clean"
                logger.info("SAST found no obvious issues.")
                
        if all_passed:
            logger.info("Security audit passed with no critical findings.")
        else:
            logger.warning("Security audit finished with warnings.")

        return Result(success=all_passed, metadata=metadata)
