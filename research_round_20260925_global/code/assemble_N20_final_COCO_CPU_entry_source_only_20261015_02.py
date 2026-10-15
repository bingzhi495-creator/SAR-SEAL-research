"""UNBOUND_NOT_EXECUTED: ordinary text/AST assembly for 32 CPU final only.

No scientific module is imported. All paths are exact ordinary TMP basenames.
Only the inherited CURRENT ninth counter/dispatch changes; historical code stays.
"""
from pathlib import Path
import ast
import hashlib
import json

root = Path('E:/SAR/seal/tmp').resolve(strict=True)


def ordinary(name, *, output=False):
    assert Path(name).name == name and name.endswith('.py')
    path = root / name
    assert path.parent == root and not path.is_symlink()
    if output:
        assert not path.exists(), 'Never overwrite an old or bound source'
    else:
        assert path.is_file() and path.resolve(strict=True).parent == root
    return path


prefix_path = ordinary('N20_SAVED_TRAINING_ARTIFACT_RECOVERY_CPU_ENTRY_UNBOUND_DRAFT_20261015_02.py')
legacy_path = ordinary('N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_04.py')
failure25_path = ordinary('N20_PRESERVED_DEV_CAPTURE_STRIDE_FAILURE25_UNBOUND_20261015_01.py')
failure26_path = ordinary('N20_PRESERVED_CPU_NUMERIC_RESOURCE_FAILURE26_UNBOUND_20261015_01.py')
front_path = ordinary('N20_FIXEDDEV_FULL187_COCO_CPU_FRONT_UNBOUND_DRAFT_20261015_02.py')
tail_path = ordinary('N20_FULL187_CPU_COCO_FINAL_RUN_TAIL_UNBOUND_DRAFT_20261015_02.py')
target = ordinary('N20_FIXEDDEV_FULL187_COCO_CPU_ENTRY_UNBOUND_DRAFT_20261015_02.py', output=True)

original = prefix_path.read_text(encoding='utf-8-sig')
prefix = original.split('\nN20_TRAIN_CPU_SCHEMA =', 1)[0]
old = '(N20_TRAIN_CPU_SCHEMA if cpu else N20_GPU_SCHEMA)'
assert prefix.count(old) == 1
prefix = prefix.replace(old, '(N20_FINAL_CPU_SCHEMA if cpu else N20_DEV_SCHEMA)')
assert prefix.count("spec['total_failed_RAW_count'] == 24") == 1
assert prefix.count("spec['total_terminal_identity_pin_count'] == expected + 24") == 1
prefix = prefix.replace("spec['total_failed_RAW_count'] == 24", "spec['total_failed_RAW_count'] == 26").replace(
    "spec['total_terminal_identity_pin_count'] == expected + 24", "spec['total_terminal_identity_pin_count'] == expected + 26").replace(
    'typed21/22/23/24 stay separate; normal+24', 'typed21/22/23/24/25/26 stay separate; normal+26')

legacy = legacy_path.read_text(encoding='utf-8-sig'); lines = legacy.splitlines(keepends=True)
wanted = {'OLD_CHECKS','LEGACY_SHARD_RANGES','COCO_METRIC_NAMES','validate_legacy_COCO','legacy_gate'}
selected = []; seen = set()
for node in ast.parse(legacy).body:
    names = {node.name} if isinstance(node, ast.FunctionDef) else (
        {item.id for item in node.targets if isinstance(item, ast.Name)} if isinstance(node, ast.Assign) else set())
    if names & wanted:
        selected.append(''.join(lines[node.lineno-1:node.end_lineno])); seen.update(names & wanted)
assert seen == wanted
cli_at = original.rfind('if __name__'); assert cli_at >= 0
cli = original[cli_at:]
assert '--spec' in cli and '--owner-execute' in cli and 'run(args.spec)' in cli
parts = [prefix, failure25_path.read_text(encoding='utf-8-sig'), failure26_path.read_text(encoding='utf-8-sig'),
         '\n\n'.join(selected), front_path.read_text(encoding='utf-8-sig'), tail_path.read_text(encoding='utf-8-sig'), cli]
text = '# UNBOUND_NOT_EXECUTED: 32 CPU final, ordinary text assembly only.\n' + '\n\n'.join(parts)
tree = ast.parse(text); functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
assert functions.count('source_gate') == functions.count('run') == 1
for name in ('n20_failed26_raw_only','n20_failed26_closed','n20_validate_final_header','n20_final_completed_CPU','inherited_source_gate'):
    assert name in functions, name
assert text.endswith(cli)
with target.open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(text)
ast.parse(target.read_text(encoding='utf-8'))
print(json.dumps(dict(status='UNBOUND_NOT_EXECUTED_AST_ONLY',path=str(target),bytes=target.stat().st_size,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_gate_count=1,run_count=1,
    original_CLI_preserved=True,scientific_import_or_execution=False)))
