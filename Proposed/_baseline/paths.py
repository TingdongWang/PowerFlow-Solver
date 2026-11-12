"""Repository-local imports shared by the baseline implementations."""
import sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROPOSED=ROOT/'Proposed'
DATA_ROOT=Path(os.environ.get('PF_DATA_ROOT',str(ROOT.parent/'Data'))).expanduser().resolve()
for p in (ROOT,PROPOSED):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
