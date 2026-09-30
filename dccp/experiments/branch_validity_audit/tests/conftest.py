import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = ROOT.parent / "state_selection_audit"
for path in (ROOT, STATE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
