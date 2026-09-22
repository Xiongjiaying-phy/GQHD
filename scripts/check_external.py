"""Verify externally supplied files without downloading or modifying anything."""
import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
failed = False
for item in json.loads((root/'data/external/Required.json').read_text()):
    path = root/item['destination']
    if path.name == 'bps.dat':
        path = Path(os.environ.get('GQHD_BPS', str(path)))
    if not path.is_file():
        print('MISSING:', item['destination'])
        failed = True
    elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        print('HASH MISMATCH:', item['destination'])
        failed = True
    else:
        print('OK:', item['destination'])
raise SystemExit(1 if failed else 0)
