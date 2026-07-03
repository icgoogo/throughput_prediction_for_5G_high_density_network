"""Kaggle script kernel entry point.

Upload this project as a Kaggle dataset or copy the `src/` folder beside this file,
then run the script kernel.
"""

import sys
from pathlib import Path

# Adjust this if you upload code as a Kaggle dataset.
CODE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(CODE_ROOT))

from hdd_pipeline.orchestrate import main

if __name__ == "__main__":
    main()
