"""Static checks on built notebooks: every code cell (and every %%writefile body) must compile."""
import ast, json, pathlib, re, sys
bad = 0
for nbp in sorted(pathlib.Path(__file__).parent.glob('build/*/*/*.ipynb')):
    nb = json.loads(nbp.read_text())
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code': continue
        s = c['source']
        if s.startswith('%%writefile'):
            s = s.split('\n', 1)[1]
            if '-a ' in c['source'].split('\n', 1)[0]: continue  # appended fragments are checked as a whole file below
        s = '\n'.join((l[:len(l) - len(l.lstrip())] + 'pass  #' + l.lstrip()) if l.lstrip().startswith(('!', '%')) else l for l in s.split('\n'))
        try: ast.parse(s)
        except SyntaxError as e: bad += 1; print('SYNTAX', nbp, 'cell', i, e)
    # reassemble appended %%writefile files
    files = {}
    for c in nb['cells']:
        if c['cell_type'] == 'code' and c['source'].startswith('%%writefile'):
            head, body = c['source'].split('\n', 1)
            parts = head.split(); name = parts[-1]
            files[name] = (files.get(name, '') + body) if '-a' in parts else body
    for name, body in files.items():
        if name.endswith('.py'):
            try: ast.parse(body)
            except SyntaxError as e: bad += 1; print('SYNTAX', nbp, name, e)
    print('ok' if not bad else 'FAIL', nbp, len(nb['cells']), 'cells', len(files), 'written files')
sys.exit(1 if bad else 0)
