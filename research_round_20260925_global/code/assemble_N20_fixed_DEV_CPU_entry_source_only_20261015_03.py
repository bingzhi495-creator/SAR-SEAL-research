"""UNBOUND_NOT_EXECUTED: assembler02 plus current26/fixed32 scopes only.
Only explicit ordinary tmp source text/stdlib AST is read; inputs are not imported.
"""
from pathlib import Path

root = Path('E:/SAR/seal/tmp').resolve(strict=True)
assert root.drive.upper() == 'E:' and root == Path('E:/SAR/seal/tmp')


def ordinary(name, *, output=False):
    assert Path(name).name == name and 'UNBOUND' in name and name.endswith('.py')
    path = root/name
    assert path.parent == root and not path.is_symlink()
    if output:
        assert not path.exists(), 'Never overwrite any prior assembled/bound entry'
    else:
        assert path.is_file() and path.resolve(strict=True).parent == root
    return path


prefix_path = ordinary('N20_SAVED_TRAINING_ARTIFACT_RECOVERY_CPU_ENTRY_UNBOUND_DRAFT_20261015_02.py')
legacy_path = ordinary('N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_04.py')
helper25_path = ordinary('N20_PRESERVED_DEV_CAPTURE_STRIDE_FAILURE25_UNBOUND_20261015_01.py')
helper26_path = ordinary('N20_PRESERVED_CPU_NUMERIC_RESOURCE_FAILURE26_UNBOUND_20261015_01.py')
front_path = ordinary('N20_FIXEDDEV_FEATURE_ERROR_CPU_FRONT_UNBOUND_DRAFT_20261015_04.py')
prior_path = ordinary('N20_PRIOR_DEV_CARRIER_METADATA_GATE_UNBOUND_DRAFT_20261015_01.py')
tail_path = ordinary('N20_FIXEDDEV_FEATURE_ERROR_CPU_NUMERIC_RUN_TAIL_UNBOUND_DRAFT_20261015_02.py')
target = ordinary('N20_FIXEDDEV_FEATURE_ERROR_CPU_ENTRY_UNBOUND_DRAFT_20261015_03.py',output=True)

# All input/output paths checked as ordinary/unbound before AST operations.
import ast
import hashlib
import json

original = prefix_path.read_text(encoding='utf-8-sig')
prefix = original.split('\nN20_TRAIN_CPU_SCHEMA =',1)[0]
old = '(N20_TRAIN_CPU_SCHEMA if cpu else N20_GPU_SCHEMA)'
assert prefix.count(old) == 1
prefix = prefix.replace(old,'(N20_DEV_CPU_SCHEMA if cpu else N20_DEV_SCHEMA)')
assert prefix.count("spec['total_failed_RAW_count'] == 24") == 1
assert prefix.count("spec['total_terminal_identity_pin_count'] == expected + 24") == 1
prefix = prefix.replace("spec['total_failed_RAW_count'] == 24","spec['total_failed_RAW_count'] == 26").replace(
    "spec['total_terminal_identity_pin_count'] == expected + 24","spec['total_terminal_identity_pin_count'] == expected + 26").replace(
    'typed21/22/23/24 stay separate; normal+24','typed21/22/23/24/25/26 stay separate; normal+26')

legacy = legacy_path.read_text(encoding='utf-8-sig'); lines = legacy.splitlines(keepends=True)
wanted = {'OLD_CHECKS','LEGACY_SHARD_RANGES','COCO_METRIC_NAMES','validate_legacy_COCO','legacy_gate'}
selected = []; seen = set()
for node in ast.parse(legacy).body:
    names = {node.name} if isinstance(node,ast.FunctionDef) else (
        {item.id for item in node.targets if isinstance(item,ast.Name)} if isinstance(node,ast.Assign) else set())
    if names & wanted:
        selected.append(''.join(lines[node.lineno-1:node.end_lineno])); seen.update(names & wanted)
assert seen == wanted

helper25 = helper25_path.read_text(encoding='utf-8-sig')
helper26 = helper26_path.read_text(encoding='utf-8-sig')
front = front_path.read_text(encoding='utf-8-sig')
prior = prior_path.read_text(encoding='utf-8-sig')
tail = tail_path.read_text(encoding='utf-8-sig')
cli_at = original.rfind('if __name__'); assert cli_at >= 0
cli = original[cli_at:]
assert '--spec' in cli and '--owner-execute' in cli and 'run(args.spec)' in cli
text = '# UNBOUND_NOT_EXECUTED: fixed32 CPU scopes; no scientific leaf executed.\n'+prefix+'\n\n'
text += helper25+'\n\n'+helper26+'\n\n'+'\n\n'.join(selected)+'\n\n'
text += front+'\n\n'+prior+'\n\n'+tail+'\n\n'+cli
tree = ast.parse(text)
functions = [node.name for node in tree.body if isinstance(node,ast.FunctionDef)]
assert functions.count('source_gate') == functions.count('run') == 1
for name in ('n20_validate_cpu_header','n20_cpu_scope','n20_prior_DEV_carrier_gate','legacy_gate',
             'n20_failed25_raw_only','n20_failed25_closed','n20_failed26_raw_only','n20_failed26_closed'):
    assert functions.count(name) == 1, name
assert front.index('failed26 = n20_failed26_raw_only(spec)') < front.index('gate = inherited_source_gate(path, spec, cpu=True)')
assert front.index('gate = inherited_source_gate(path, spec, cpu=True)') < front.index('n20_failed26_closed(spec, gate, failed26)')
assert text.endswith(cli) and all(part in text for part in (helper25,helper26,front,prior,tail))
with target.open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(text)
assert target.resolve(strict=True).parent == root and not target.is_symlink()
ast.parse(target.read_text(encoding='utf-8'))
print(json.dumps(dict(status='UNBOUND_NOT_EXECUTED_AST_ONLY',path=str(target),bytes=target.stat().st_size,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_gate_count=1,run_count=1,
    original_CLI_preserved=True,typed25_unchanged=True,typed26_raw_before_SHA=True,
    scientific_import_or_execution=False)))
