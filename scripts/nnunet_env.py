import os
import subprocess
from pathlib import Path

def load_env():
    """
    Loads nnU-Net environment variables from the nnunet_env.sh script into Python.
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

    # Parse environment variables
    for line in output.split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value
