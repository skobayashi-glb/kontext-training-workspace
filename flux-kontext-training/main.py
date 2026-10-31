import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from flux_kontext_training.api import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
