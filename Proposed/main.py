"""Run with python main.py; all options live in config.json."""
import runtime  # Activate isolated dependencies before numerical imports.
import fcntl
import traceback
import torch
from runtime import ROOT, Logger, read_config, setup, json_write, code_hashes, model_meta
from data import load_data
from model import RJMNC
from Function import PowerFlow
from train import train_full, preflight
from evaluate import full_test


def run(config, output_root=ROOT):
    out = output_root / 'Results' / config['system']
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out, config['mode'])
    with (out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another run is already using {out}')
        try:
            device = setup(config)
            log('start',config=config,code_sha256=code_hashes())
            meta, groups, report = load_data(config, log)
            json_write(out / (config['mode'] + '_config.json'),config)
            json_write(out / 'data_report.json',report)
            if config['mode'] == 'prepare':
                return report
            if config['mode'] == 'check':
                model = RJMNC(model_meta(meta,config)).to(device)
                batch = preflight(model,PowerFlow(meta).to(device),groups['train'],config,log)
                log('check_complete',effective_batch=batch,trained_checkpoint_written=False)
                return report
            if config['mode'] == 'train_test':
                json_write(out / 'config_used.json',config)
                json_write(out / 'code_manifest.json',code_hashes())
                model, meta = train_full(meta,groups,config,report,out,log)
            else:
                checkpoint = torch.load(out / 'model.pt',map_location='cpu',weights_only=False)
                if checkpoint['system'] != config['system'] or checkpoint['data_signature'] != report['signature']:
                    raise ValueError('Checkpoint system or dataset signature mismatch')
                meta = checkpoint['meta']
                model = RJMNC(meta).to(device)
                model.load_state_dict(checkpoint['state_dict'],strict=True)
                model.eval()
            test=groups['test']
            summary = full_test(model,meta,test,config,out,log)
            summary.update(data_source='csv',data_cache_read=False,data_cache_written=False,start='dataset_initial_from_csv')
            json_write(out/'summary.json',summary)
            if config['save_heatmap']:
                from plot import save_heatmap
                save_heatmap(out)
            log('complete',output=str(out),accuracy_target_met=summary['accuracy_target_met'])
            return summary
        except Exception:
            log('failed',traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Plain MLP PF solver: all settings in JSON')
    parser.add_argument('--config',type=Path,default=ROOT/'Results/IEEE_300/IEEE_300.json')
    args=parser.parse_args()
    run(read_config(args.config))
