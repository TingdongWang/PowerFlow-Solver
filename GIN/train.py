"""Train GIN from the supplied JSON configuration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Proposed._baseline import paths
from Proposed._baseline.training import main
if __name__=='__main__': main()
