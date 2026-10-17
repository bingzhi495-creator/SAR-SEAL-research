"""UNBOUND Torch-free CPU body. Only a future admitted entry may call run."""
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import types


def need(ok,message):
    if not ok:
        raise RuntimeError(message)


def no_torch():
    need(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),
         'This input-only body must never import Torch')


def _write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


class _Tee:
    def __init__(self,original,log): self.original,self.log=original,log
    def write(self,value):
        self.original.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self): self.original.flush(); self.log.flush()


def persist_run(output,report,work,check):
    """One exclusive partial-preserving pass; resource errors are re-raised intact."""
    output=Path(output); output.mkdir()
    started=time.monotonic()
    report.update(status='STARTED',image_records_written=0,qualified=False)
    with (output/'probe.log').open('x',encoding='utf-8',newline='\n') as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
            try:
                with (output/'IMAGE_CHECKS.jsonl').open('x',encoding='utf-8',newline='\n') as rows:
                    def record_image(value):
                        check()
                        rows.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
                        rows.flush()
                        report['image_records_written']+=1
                    check()
                    safe=work(record_image)
                    check()
                _write_new(output/'SAFE_SUMMARY.json',safe)
                report['status']='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
            except BaseException as error:
                report['status']='FAILED'
                report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                traceback.print_exc()
                raise
            finally:
                report['elapsed_seconds']=time.monotonic()-started
                # Deliberately no resource check before this last partial metadata write.
                # It cannot create Q or certify success after the check has rejected work.
                _write_new(output/'REPORT.json',report)
    return report


def _load(c,reference,name):
    path=Path(c.closed(reference)); raw=path.read_bytes()
    need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],
         'Closed scientific source bytes')
    module=types.ModuleType(name); module.__file__=str(path); sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def run(spec,c,pre,context):
    """Actual library imports occur only inside the future original CPU leaf."""
    no_torch()
    output=Path(spec['output_directory'])
    report=dict(schema='U43-HRSID-population-CPU-wrapper-report-pending-v1',
        source=spec['body_source'],population_source=spec['population_source'],checker_source=spec['checker_source'],
        runtime_source=spec['runtime_source'],spec=c.execution_spec_ref,
        CPU_admission=c.execution_admission_ref,phase=spec['phase'],context=spec['context'],
        prelock=spec['prelock'],plan=spec['plan'],population={},actual_import_records=[],
        Torch_imported=False,CUDA_initialized=False,model_forward_executed=False,
        optimizer_created=False,new_training=False,AP=False,DEV=False,TEST_annotations_in_scope=True,TEST_annotations_read=None,
        TEST_used_for_model_selection=False,weights_loaded=False,weights_written=False,
        qualified=False,original_actual0_self_certified=False,semantic_alignment_proven=False,
        automatic_retry=False,automatic_sharding=False,
        source_footprint_scope='Selected actual Python modules and extension files; not all Python/OS/DLL dependencies')
    aliases=('U43_CLOSED_POPULATION','U43_CLOSED_SINGLE_IMAGE_CHECKER')
    absent=object(); previous={name:sys.modules.get(name,absent) for name in aliases}

    def work(record_image):
        try:
            libs={}
            for name,reference in spec['library_sources'].items():
                c.check(); no_torch()
                module=importlib.import_module(name)
                path=Path(c.closed(reference)).resolve()
                need(getattr(module,'__file__',None) is not None and Path(module.__file__).resolve()==path
                     and c.ref(path)==reference,'Actual selected library path/hash mismatch: '+name)
                libs[name]=module
                report['actual_import_records'].append(dict(module=name,source=reference,
                    version=getattr(module,'__version__',None)))
                no_torch()
            population=_load(c,spec['population_source'],aliases[0])
            checker=_load(c,spec['checker_source'],aliases[1])
            no_torch()
            need(libs['PIL.Image'].core is libs['PIL._imaging']
                 and libs['pycocotools.mask']._mask is libs['pycocotools._mask'],
                 'Selected Pillow/official mask native extension objects match imported APIs')
            print('Starting admitted HRSID train3642/test1962 pixel and polygon-mask checks; no model.',flush=True)
            result=population.run_population(libs['PIL.Image'],libs['PIL.ImageFile'],libs['numpy'],
                libs['pycocotools.mask'],context,checker.check_one_image,closed_ref=c.closed,check=c.check,
                record_image=record_image,report=report['population'])
            need(result is report['population'] and result['schema']=='U43-HRSID-population-pixel-mask-check-report-pending-v1'
                 and result['status']=='INPUT_CHECK_COMPLETE_PENDING_ROOT'
                 and all(result[k] is True for k in ('input_check_complete','full_image_visitation',
                     'full_private_audit_coverage','annotation_count_conserved'))
                 and result['qualified'] is False and result['semantic_alignment_proven'] is False
                 and result['new_training'] is False and result['AP'] is False,
                 'Whole-population check must finish successfully; bad images/masks are not filtered')
            totals=result['totals']
            need(totals['processed_images']==totals['recorded_images']==totals['image_checks_passed']==5604
                 and totals['original_annotations_seen']==totals['raster_passed_annotations']==16969
                 and report['image_records_written']==5604,'Actual whole-population counts and private callback log agree')
            no_torch(); c.check()
            for value in [spec[k] for k in ('entry_source','body_source','population_source','checker_source','runtime_source')]+list(spec['library_sources'].values()):
                need(c.ref(value['path'])==value,'Executed selected source unchanged')
            # Only numeric counters and fixed scope words leave the private full report.
            safe=dict(schema='U43-HRSID-population-safe-summary-pending-v1',
                status='INPUT_CHECK_COMPLETE_PENDING_ROOT',dataset='HRSID',
                split_counts={s:{k:result['split_summaries'][s][k] for k in population.COUNTERS} for s in ('train','test')},
                totals={k:totals[k] for k in population.COUNTERS},
                input_check_complete=True,qualified=False,semantic_alignment_proven=False,
                model_forward_executed=False,new_training=False,AP=False,TEST_annotations_read=True,
                TEST_used_for_model_selection=False,blind_test_restored=False,
                public_release_requires_Root_acceptance=True)
            need(all(type(v) is int and v>=0 for v in safe['totals'].values())
                 and all(type(v) is int and v>=0 for row in safe['split_counts'].values() for v in row.values()),
                 'Safe summary contains numeric aggregate counters, never filenames/IDs/references')
            print('All input checks completed; original terminal and Root acceptance still pending.',flush=True)
            return safe
        finally:
            for name,value in previous.items():
                if value is absent: sys.modules.pop(name,None)
                else: sys.modules[name]=value
            report['source_aliases_restored_exact']=all(name not in sys.modules if value is absent else sys.modules.get(name) is value
                                                       for name,value in previous.items())
            report['Torch_imported']=any(n=='torch' or n.startswith('torch.') for n in sys.modules)
            report['CUDA_initialized']=False if not report['Torch_imported'] else None
            # Both split documents precede metadata_validated in the population core.
            # An earlier failure leaves actual TEST read extent unknown, not false.
            if report['population'].get('metadata_validated') is True:
                report['TEST_annotations_read']=True
            report['actual_CPU_affinity']=c.psutil.Process().cpu_affinity()
            report['peak_RSS_bytes']=c.peak_rss
            report['minimum_available_RAM_bytes']=c.minimum_available
            need(report['source_aliases_restored_exact'] and not report['Torch_imported'],
                 'CPU-only source alias cleanup; no Torch import')

    persist_run(output,report,work,c.check)
    no_torch(); c.check()
    return dict(report_ref=c.ref(output/'REPORT.json'),log_ref=c.ref(output/'probe.log'),
        image_records_ref=c.ref(output/'IMAGE_CHECKS.jsonl'),safe_summary_ref=c.ref(output/'SAFE_SUMMARY.json'),
        images=5604,annotations=16969,input_check_complete=True,qualified=False,
        Torch_imported=False,CUDA_initialized=False,new_training=False,AP=False,
        TEST_annotations_read=True,TEST_used_for_model_selection=False,semantic_alignment_proven=False)
