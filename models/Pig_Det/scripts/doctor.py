import json,platform,sys,importlib.util

def main():
    result={'python':sys.version,'platform':platform.platform(),'recommended_python':'3.11','dependencies':{m:bool(importlib.util.find_spec(m)) for m in ['PIL','yaml','matplotlib','torch','ultralytics']}}
    if result['dependencies']['torch']:
        try:
            import torch
            result.update(torch=torch.__version__,cuda=torch.cuda.is_available(),mps=bool(hasattr(torch.backends,'mps') and torch.backends.mps.is_available()))
        except Exception as e: result['torch_error']=str(e)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
