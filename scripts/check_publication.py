#!/usr/bin/env python3
"""Check the public tree for accidental private paths, engines and credentials.

This complements a dedicated secret scanner; it is not a security guarantee.
Runtime output and private credentials must remain outside the published tree.
"""
from pathlib import Path
import ast
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '__pycache__', '.pytest_cache', 'runs', 'tmp'}
FORBIDDEN_FILES = {'odvl_adapter.py', 'basira_odvl.py', 'agent_identity.py'}
PATTERNS = {
    'credential': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|bsr_(?:live|test)_key-[A-Za-z0-9_-]{40,}|AKIA[A-Z0-9]{16})',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'private_home_path': rb'/Users/[^/\s\"\x27]+/(?:Documents|Downloads|Desktop|\.sawm|\.codex)',
    'private_tmp_path': rb'/private/tmp/(?:claude|codex|sawm-publication)',
}


def paths():
    # Use Git's publication boundary when present, otherwise inspect clean files.
    process = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True)
    if process.returncode == 0 and process.stdout:
        return [ROOT / name.decode() for name in process.stdout.split(b'\0') if name]
    return [p for p in ROOT.rglob('*') if p.is_file() and not EXCLUDED.intersection(p.relative_to(ROOT).parts)]


def main():
    findings = []
    checked = paths()
    for path in checked:
        rel = str(path.relative_to(ROOT))
        if path.is_symlink():
            findings.append((rel, 'symlink'))
            continue
        if path.name in FORBIDDEN_FILES or (path.name.startswith('.env') and path.name != '.env.example'):
            findings.append((rel, 'private_file'))
        data = path.read_bytes()
        for label, pattern in PATTERNS.items():
            if re.search(pattern, data):
                findings.append((rel, label))
        if path.suffix == '.py':
            for node in ast.walk(ast.parse(data)):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                         else [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    if name.split('.')[0] in {'backend', 'services', 'basira_odvl', 'odvl_adapter', 'agent_identity'}:
                        findings.append((rel, 'private_engine_import'))
    print(json.dumps({'checked_files': len(checked), 'findings': findings}, indent=2))
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
