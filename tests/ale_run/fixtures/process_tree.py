import os
from pathlib import Path
import signal
import subprocess
import sys
import time


if "--child" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(os.environ["AUDIT_CHILD_PID"]).write_text(str(os.getpid()))
else:
    subprocess.Popen([sys.executable, __file__, "--child"])

while True:
    time.sleep(0.05)
