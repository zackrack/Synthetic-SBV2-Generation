#!/usr/bin/env python
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.synthetic_ped_gradio import build_app

if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False)
