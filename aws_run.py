"""Windows-friendly entry point; implementation lives in src/."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from kontext_remote.client import main

if __name__ == '__main__':
    sys.exit(main())
