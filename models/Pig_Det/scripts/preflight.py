"""Read-only checks for a PigDetect YOLO dataset and the training contract."""
import argparse
import json
import platform
import sys
from pathlib import Path

from scripts.dataset_layout import configure
from scripts.data_tools import audit

REQUIRED_VERSION = '8.4.48'


def run(dataset=None, out='reports/preflight', check_runtime=True):
    root = configure(explicit=dataset)
    report = audit('config/data.yaml', out)
    errors = list(report['errors'])
    warnings = list(report['warnings'])
    runtime = {'python': platform.python_version(), 'platform': platform.platform()}
    if sys.version_info[:2] != (3, 11):
        warnings.append('Recommended Python 3.11; current interpreter differs')
    if check_runtime:
        try:
            import ultralytics
            runtime['ultralytics'] = ultralytics.__version__
            if ultralytics.__version__ != REQUIRED_VERSION:
                errors.append(f'Ultralytics must be {REQUIRED_VERSION}; found {ultralytics.__version__}')
        except ImportError:
            errors.append('Ultralytics not installed: pip install -r requirements-train.txt')
        try:
            import torch
            runtime['torch'] = torch.__version__
            runtime['mps_available'] = bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
            runtime['cuda_available'] = torch.cuda.is_available()
            if not (runtime['mps_available'] or runtime['cuda_available']):
                warnings.append('No MPS/CUDA accelerator detected; training may be very slow on CPU')
        except ImportError:
            errors.append('PyTorch not installed')
    summary = {'dataset': str(root), 'runtime': runtime, 'stats': report['stats'],
               'errors': errors, 'warnings': warnings,
               'box_policy': 'Expand prediction width and height 12% (6% per side); keep labels unchanged',
               'model_filename': 'pig_set.pt'}
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset')
    p.add_argument('--out', default='reports/preflight')
    p.add_argument('--no-runtime', action='store_true')
    a = p.parse_args()
    result = run(a.dataset, a.out, not a.no_runtime)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
