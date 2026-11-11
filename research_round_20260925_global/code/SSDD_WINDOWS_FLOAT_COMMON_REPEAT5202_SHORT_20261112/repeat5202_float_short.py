"""Run only the prospectively selected SSDD float/common downstream-seed 5202 short pair."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from unittest.mock import patch


SELF = Path(__file__).resolve()
ROOT = SELF.parent.parent
SHORT = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_SHORT_PUBLISHED_STAGE_PRIVATE_20261112_01"
SHORT_SOURCE = SHORT / "windows_float_common_short.py"
SHORT_SHA = "7b0b7b3f12d802f888240b1b979366d9df0304213b5eaceee5faa9d9711b3c27"
SHORT_REQUEST = SHORT / "REQUEST_PRIVATE.json"
SHORT_REQUEST_SHA = "b4f4617b110267f4b56cf5c2c6c917c7821058e7213a83b07c1fa3477b8bc84c"
FLOAT_MANIFEST = "56fdd4be8d19b4e2e42ff736e89775f32b1e6ce2a9c91d2a90afad509640bc14"
SEED_SOURCE = ROOT / "github_release_20261011/research_round_20260925_global/code/SSDD_DEV187_ALIAS_AND_COMMON_FIT_20261111/ssdd_repeat5202_short.py"
SEED_SHA = "2c008f48cf8c0b09327279c78d977ad02a0795e12054fb848c19bc12deb7ac10"
FIRST_SCORE = ROOT / "SSDD_WINDOWS_FLOAT_PRIMARY_DEV187_ACTUAL_PRIVATE_20261112_01/INDEPENDENT_FLOAT_PRIMARY_SCORE_SAFE.json"
FIRST_SCORE_SHA = "07bbf74198bec0e55f131a2aeeca2f7efa4c6008654dd3816e7b3ea57d8178f4"
FIRST_TRAIN_PAIR_SHA = "029a0b69a5cc0903d5f410fa1d79ced0eafcd7e910c70d1d6ff93bb23e498dc1"
OUTPUT = ROOT / "SSDD_WINDOWS_Q100_FLOAT_COMMON_REPEAT5202_SHORT_ACTUAL_PRIVATE_20261112_01"
AUTHORITY = "SSDD_WINDOWS_Q100_FLOAT_COMMON_REPEAT5202_SHORT"
REQUEST = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_SHORT_REQUEST_V1"
WORKER_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_SHORT_ARM_CLOSED_PENDING_ACCEPTANCE_V1"
PAIR_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_SHORT_PAIR_INDEPENDENTLY_ACCEPTED_V1"
WAIT_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_SHORT_ORIGINAL_OWNER_WAIT_V1"
PAIR_NAME = "FLOAT_COMMON_REPEAT5202_SHORT_PAIR_ACCEPTED_SAFE.json"
PARSER = ROOT / "SSDD_WINDOWS_FLOAT_CLOSED_VISIT_ARRAY_READBACK_PRIVATE_20261112_01/visit_array.py"
PARSER_SHA = "2e4a0060dc1bdbc542ac7a92d7290d2c3165cedebb484590bc7e9942129e4899"
ANCHOR = 70.4732456
SCOPE = dict(FIT=True,DEV=False,TEST=False,AP=False)
MODES = ("native","center")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return sha256(raw).hexdigest()


def ref(path):
    raw=path.read_bytes()
    return dict(path=path.as_posix(),bytes=len(raw),sha256=digest(raw))


def pinned(path, expected):
    raw=path.read_bytes()
    require(path.is_absolute() and path.resolve(strict=True)==path and digest(raw)==expected,
            "Expected exact canonical closed or source file")
    return raw


def load_module(path, expected, name):
    obj=ModuleType(name)
    obj.__file__=str(path)
    exec(compile(pinned(path,expected),str(path),"exec"),obj.__dict__)
    return obj


def check_first_seed_gate(score):
    gate=score["first_seed_gate"]
    require(score["seed"]==5201 and score["FIT"] is score["TEST"] is False and score["DEV"] is score["AP"] is True
            and gate.get("status")=="AT_MOST_ONE_PRELOCKED_REPLICATION"
            and gate.get("paired_center_minus_native_AP_points")==3.233436705382531
            and gate.get("controlling_margin_AP_points")==gate.get("center_minus_integer_trained_native_AP_points")
            ==0.22034536085387968 and gate.get("threshold_points")==1.0,
            "Exactly one prospectively selected 5202 repeat from the closed first-seed score")


def bind_repeat_runtime(admitted, programs, adapter):
    torch,init,worker,deps=admitted
    require(worker.__name__=="run_fit_arm", "Expected exact native trainer before downstream binding")
    repeat=deps._replace(initialize_q100=adapter.repeat_initializer(deps.initialize_q100))
    return torch,init,programs.worker.run_fit_arm,repeat


def request_value(commit):
    require(type(commit) is str and re.fullmatch("[0-9a-f]{40}",commit) is not None,
            "Exact prospectively verified operation commit")
    return dict(status=REQUEST,operation_commit=commit,worker=ref(SELF),
                original_float_source_sha256=SHORT_SHA,original_float_request_sha256=SHORT_REQUEST_SHA,
                original_float_manifest_sha256=FLOAT_MANIFEST,downstream_seed_adapter_sha256=SEED_SHA,
                first_float_score_sha256=FIRST_SCORE_SHA,initialization_seed=5201,training_seed=5202,
                visits_per_arm=32,optimizer_updates_per_arm=8,ema_updates_per_arm=8,
                anchor_AP=ANCHOR,output_root=OUTPUT.as_posix(),**SCOPE)


def context(config=None, expected=None):
    original=load_module(SHORT_SOURCE,SHORT_SHA,"_exact_original_float_common_short")
    ctx=original.context(SHORT_REQUEST,SHORT_REQUEST_SHA)
    base_cfg=ctx.cfg
    require(ctx.worker.manifest(ctx.entry,dict(base_cfg,stage="short"))==FLOAT_MANIFEST,
            "Exact original 5201 float source manifest")
    score=json.loads(pinned(FIRST_SCORE,FIRST_SCORE_SHA))
    check_first_seed_gate(score)
    require(score["training_pair_sha256"]==FIRST_TRAIN_PAIR_SHA,"Exact accepted first float training/score pair")
    adapter=load_module(SEED_SOURCE,SEED_SHA,"_exact_previously_qualified_downstream_seed_adapter")
    programs=adapter.load_programs(ctx.old.CODE)
    require(adapter.INITIALIZATION_SEED==5201 and adapter.TRAINING_SEED==5202
            and programs.worker.SEED==5201 and programs.qualifier.SEED==5201,
            "Only native seed consumers may differ, fixed initialized Q100")
    commit="0"*40
    if config is not None:
        request=json.loads(pinned(config,expected))
        commit=request.get("operation_commit")
        require(commit!="0"*40 and request==request_value(commit),"Exact immutable repeat request")
    worker_ref=ref(SELF)
    ctx.cfg=dict(base_cfg,worker=worker_ref,output_root=OUTPUT.as_posix(),implementation_commit=commit,operation_commit=commit)
    ctx.worker_path=SELF
    ctx.output=OUTPUT
    ctx.worker.AUTHORITY,ctx.worker.WORKER_STATUS,ctx.worker.PAIR_STATUS=AUTHORITY,WORKER_STATUS,PAIR_STATUS
    manifest=dict(original_float_source_manifest_sha256=FLOAT_MANIFEST,downstream_seed_adapter_sha256=SEED_SHA,
                  worker_source_sha256=worker_ref["sha256"],initialization_seed=5201,training_seed=5202)
    ctx.worker.manifest=lambda _entry,_cfg: digest(ctx.worker.encoded(manifest))
    return original,ctx,adapter,programs


def worker_main(argv):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ("code-root","config","scope"):
        parser.add_argument("--"+key,type=Path,required=True)
    for key in ("config-sha256","scope-sha256"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--mode",choices=MODES,required=True)
    args=parser.parse_args(argv)
    original,ctx,adapter,programs=context(args.config,args.config_sha256)
    require(args.code_root==ctx.old.CODE,"Exact unchanged original Windows stock code root")
    original_module=ctx.worker.module

    def worker_config(_entry,path,expected):
        require((path,expected)==(args.config,args.config_sha256),"Exact immutable owned request")
        return ctx.cfg,ctx.output

    with ExitStack() as stack:
        def worker_module(path,expected,name):
            loaded=original_module(path,expected,name)
            if name=="windows_entry":
                stock=loaded.admit_windows_runtime
                stack.enter_context(original.float_population(loaded,lambda scope,integer,ids:
                    original.select_layout(ctx,scope,integer,ids)))
                stack.enter_context(patch.object(loaded,"admit_windows_runtime",
                    lambda scope: bind_repeat_runtime(stock(scope),programs,adapter)))
            return loaded

        stack.enter_context(patch.object(ctx.worker,"module",worker_module))
        stack.enter_context(patch.object(ctx.worker,"read_config",worker_config))
        ctx.worker.run(args)
    return 0


def require_wait(entry,args,legacy,record,config_sha):
    wait=entry.document(pinned(args.original_wait,args.original_wait_sha256))
    expected=dict(status=WAIT_STATUS,original_os_wait=True,original_os_exit=0,owner_birth=record["owner_birth"],
                  owner_config_sha256=config_sha,owner_source_sha256=ref(SELF)["sha256"])
    entry.exact(wait,expected,("combined_stdout_sha256",),reason="Exact captured original owner OS wait")
    require(not legacy.live(record["owner_birth"]),"Original coordinator must have closed")
    stdout=pinned(args.original_stdout,wait["combined_stdout_sha256"])
    original=load_module(SHORT_SOURCE,SHORT_SHA,"_exact_original_float_owner_events")
    original.verify_original_events([entry.document(line) for line in stdout.splitlines() if line])
    return wait


def bind_acceptance(ctx,programs):
    original_module=ctx.old.module
    original_encode=programs.qualifier.encode_receipt
    original_owner_path=Path(ctx.cfg["p333_owner"]["path"])
    raw=pinned(original_owner_path,ctx.cfg["p333_owner"]["sha256"])
    source=raw.decode()
    needle="    visits = entry.document(stock)"
    require(source.count(needle)==1,"Exact P383 closed visit array seam")
    patched=ModuleType("_closed_repeat_array_readback")
    patched.__file__=str(original_owner_path)
    patched.decode_stock_visits=load_module(PARSER,PARSER_SHA,"_exact_validated_visit_array").decode_stock_visits
    exec(compile(source.replace(needle,"    visits = decode_stock_visits(stock, entry.unique, entry.invalid_constant)",1),
                 str(original_owner_path),"exec"),patched.__dict__)

    def owner_module(path,expected,name):
        previous=original_module(path,expected,name)
        if name!="_p333_exact_stock_qualifier":
            return previous

        def encode(receipt):
            if receipt.get("status")==PAIR_STATUS:
                receipt=dict(receipt,initialization_seed=5201,first_float_score_sha256=FIRST_SCORE_SHA,anchor_AP=ANCHOR)
            return original_encode(receipt)

        programs.qualifier.encode_receipt=encode
        return programs.qualifier

    ctx.old.module=owner_module
    ctx.p333.read_common=patched.read_common


def owner_main(argv):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    actions=parser.add_mutually_exclusive_group(required=True)
    for action in ("preflight-cpu","make-request","run","accept"):
        actions.add_argument("--"+action,action="store_true")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--operation-commit")
    parser.add_argument("--original-wait",type=Path)
    parser.add_argument("--original-wait-sha256")
    parser.add_argument("--original-stdout",type=Path)
    args=parser.parse_args(argv)
    require(all((v is not None)==args.accept for v in (args.original_wait,args.original_wait_sha256,args.original_stdout)),
            "Only independent closure uses original wait/stdout")
    if args.make_request:
        require(args.config is not None and args.operation_commit is not None and args.config_sha256 is None,
                "Fresh explicit repeat request destination and verified commit")
        raw=(json.dumps(request_value(args.operation_commit),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        with args.config.open("xb") as stream:
            stream.write(raw)
        print(json.dumps(dict(status="SSDD_FLOAT_REPEAT5202_REQUEST_LOCKED",sha256=digest(raw))),flush=True)
        return 0
    require((args.config is not None)==(args.config_sha256 is not None)
            and (args.preflight_cpu or args.config is not None),"Exact repeat request for actual original run")
    original,ctx,adapter,programs=context(args.config,args.config_sha256)
    short,legacy,runtime=ctx.old.load_guards("short")
    lock=dict(ctx.cfg,stage="short")
    base,replay,ids=ctx.old.inputs(ctx.entry,legacy,runtime,lock)
    original.select_layout(ctx,base,Path(base["fit_layout"]),ids)
    if args.preflight_cpu:
        require(not OUTPUT.exists() and "torch" not in sys.modules,"Fresh no-CUDA exact float preflight")
        ctx.entry.emit(dict(status="SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_SHORT_CPU_PREFLIGHT_PASSED_V1",
             source_manifest_sha256=ctx.worker.manifest(ctx.entry,lock),fit_images=len(ids),fit_rows=1661,
             initialization_seed=adapter.INITIALIZATION_SEED,training_seed=adapter.TRAINING_SEED,anchor_AP=ANCHOR,
             model_forward_calls=0,optimizer_updates=0,torch_imported=False,**SCOPE))
        return 0

    config_ref=legacy.ref(args.config)
    require(config_ref["sha256"]==args.config_sha256,"Exact immutable original request")
    raw=legacy.P189["stable"](legacy.STATE,short.STATE_LIMIT)
    state=json.loads(raw)
    ctx.old.check_idle(state)
    ctx.old.verify_main_handoff(state,ctx.entry,legacy)
    ctx.p333.attach_common_stage(ctx.old,ctx.worker,SELF,args.config,args.config_sha256)
    if args.accept:
        bind_acceptance(ctx,programs)
        cas=legacy.cas
        with patch.object(ctx.p333,"require_wait",require_wait),patch.object(ctx.p333,"PAIR_NAME",PAIR_NAME), \
                patch.object(legacy,"cas",lambda before,new,_name: cas(before,new,"SSDD_FLOAT_REPEAT5202_SHORT_ACCEPT")):
            return ctx.p333.accept(args,ctx.old,ctx.worker,ctx.entry,short,legacy,lock,OUTPUT,config_ref,raw,state,base,replay,ids)
    require(AUTHORITY not in state and not OUTPUT.exists(),"Exactly one fresh seed5202 original short pair")
    return ctx.old.execute(ctx.entry,short,legacy,runtime,lock,OUTPUT,config_ref,raw,state,base,replay,None)


if __name__=="__main__":
    arguments=sys.argv[1:]
    raise SystemExit(worker_main(arguments) if arguments and arguments[0]=="--code-root" else owner_main(arguments))
