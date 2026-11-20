"""Locate the user's existing YOLO dataset without copying or modifying images/labels."""
from pathlib import Path
import yaml

SPLITS = ('train', 'val', 'test')

def locate(project=None, explicit=None):
    project = Path(project or Path(__file__).resolve().parents[1]).resolve()
    candidates = ([Path(explicit).expanduser()] if explicit else [
        project.parent / 'pig_det', project / 'pig_det',
        project / 'data' / 'pig_det', project / 'data' / 'PigDetectYOLO',
    ])
    for candidate in candidates:
        root = candidate.resolve()
        if all((root/'images'/s).is_dir() and (root/'labels'/s).is_dir() for s in SPLITS):
            return root
    raise FileNotFoundError('Cannot locate pig_det dataset with images/{train,val,test} and labels/{train,val,test}. Checked: ' + ', '.join(map(str,candidates)))

def configure(project=None, explicit=None):
    project = Path(project or Path(__file__).resolve().parents[1]).resolve()
    root = locate(project, explicit)
    cfg = {'path':str(root), 'train':'images/train', 'val':'images/val', 'test':'images/test', 'names':{0:'pig'}}
    target = project/'config'/'data.yaml'
    target.write_text('# Auto-configured dataset location. Original data are never modified.\n'+yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return root

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',help='Optional explicit path to pig_det directory');a=p.parse_args()
    print('Dataset:',configure(explicit=a.dataset))
