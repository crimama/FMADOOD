import glob, re

for path in glob.glob('PaperWorks/cvpr-2026-author-kit-template/sec/*.tex'):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # fix broken line-break + ight] -> \right]
    content = re.sub(r'\n\s*ight\]', r'\\right]', content)
    content = re.sub(r'\bight\]', r'\\right]', content)
    content = re.sub(r'\boindent\b', r'\\noindent', content)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed formatting in {path}")
