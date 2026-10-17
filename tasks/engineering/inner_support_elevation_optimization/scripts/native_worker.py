"""Enter a delegated resource group before executing one trusted native process."""

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    group, cpu, *command = sys.argv[1:]
    (Path(group) / "cgroup.procs").write_text(str(os.getpid()))
    os.sched_setaffinity(0, {int(cpu)})
    os.execv(command[0], command)
