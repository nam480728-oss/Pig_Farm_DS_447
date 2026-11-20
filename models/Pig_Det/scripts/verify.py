import subprocess,sys,ast
from pathlib import Path

def main():
    for p in Path('.').rglob('*.py'):
        if '.venv' not in p.parts: ast.parse(p.read_text(encoding='utf-8'))
    subprocess.run([sys.executable,'-W','error::ResourceWarning','-m','unittest','discover','-s','tests','-v'],check=True)
if __name__=='__main__':main()
