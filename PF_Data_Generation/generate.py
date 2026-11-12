"""Build and run the retained generator without overwriting a dataset."""
import argparse, os, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--system',type=int,choices=[14,118,300,1354],required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args()
    here=Path(__file__).resolve().parent
    for name in ('bus.txt','line.txt','gen.txt'):
        if not (here/f'IEEE_{a.system}'/name).is_file():
            p.error(f'Missing original network input: IEEE_{a.system}/{name}')
    include=os.environ.get('NICSLU_INCLUDE');lib=os.environ.get('NICSLU_LIB')
    if not include or not lib:p.error('Set NICSLU_INCLUDE and NICSLU_LIB first')
    root=a.output_root.expanduser().resolve();out=root/f'IEEE_{a.system}'
    if out.exists() and any(out.iterdir()):p.error('Output directory must be absent or empty')
    exe=here/f'Generate_IEEE_{a.system}'
    definition='-DGEN_OUTPUT_ROOT="'+root.as_posix().rstrip('/')+'/"'
    subprocess.run(['g++','-O3','-std=c++17','-pthread',f'-DBUS_TOTAL={a.system}',definition,
        '-I'+include,str(here/'Slover_Gen_Data.cpp'),'-L'+lib,'-Wl,-rpath,'+lib,
        '-lnicslu','-ldl','-lm','-o',str(exe)],check=True,cwd=here)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'Gen.log').open('x') as log:
        subprocess.run([str(exe)],cwd=here,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
