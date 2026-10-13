import glob, re

for path in glob.glob('PaperWorks/cvpr-2026-author-kit-template/sec/*.tex'):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # \igcup -> \bigcup
    content = content.replace(r'\igcup', r'\bigcup')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed \\igcup in {path}")
