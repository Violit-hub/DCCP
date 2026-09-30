"""让测试只导入当前独立实验包。"""

import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIT_ROOT))
