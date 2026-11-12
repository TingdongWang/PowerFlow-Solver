"""Evaluate PINN4PF on the complete CSV test set."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Proposed._baseline import paths
from Proposed._baseline.evaluate import run
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--system',choices=['IEEE_14','IEEE_118','IEEE_300','IEEE_1354'],required=True)
    a=p.parse_args()
    path=Path(__file__).resolve().parent/'Results'/a.system/(a.system+'.json')
    run('PINN4PF',a.system,device='cuda',config=json.loads(path.read_text()))
