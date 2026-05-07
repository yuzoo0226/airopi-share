"""Compatibility wrapper for legacy openpi_utils launchers.

The training implementation now lives in scripts/train.py.
"""

import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import train


def main() -> None:
    train.main(train.load_train_config_from_cli())


if __name__ == "__main__":
    main()
