"""
Load nnU-Net environment variables from the companion shell script into the current process.

Typical usage at the top of any HPO or training script:

    from scripts.nnunet_env import load_env
    load_env()
"""
import os
import subprocess
from pathlib import Path


def load_env() -> None:
    """Source nnunet_env.sh and inject all exported variables into os.environ.

    Reads ``scripts/nnunet_env.sh`` (located next to this module), sources it
    in a subprocess, captures the resulting environment, and updates
    ``os.environ`` with every key=value pair found.

    Raises:
        FileNotFoundError: If ``nnunet_env.sh`` does not exist alongside this
            module.
    """
    script_path = Path(__file__).parent / "nnunet_env.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"Environment script not found: {script_path}")

    command = f"bash -c 'source {script_path} && env'"
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    output = proc.communicate()[0].decode()

    for line in output.split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value
