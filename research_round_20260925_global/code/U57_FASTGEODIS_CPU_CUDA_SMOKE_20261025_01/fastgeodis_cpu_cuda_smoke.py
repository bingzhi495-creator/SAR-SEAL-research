
"""Two fixed synthetic FastGeodis CPU/CUDA checks; no SAR/model/AP evaluation."""
import argparse,datetime,gc,importlib.metadata,json,os,resource,sys,time
from pathlib import Path

def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable could not be resolved')

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args()
    out=Path(a.report);assert out.parent.is_dir() and not out.exists()
    assert sys.platform=='linux'
    os.sched_setaffinity(0,{0,1});os.nice(10)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    resource.setrlimit(resource.RLIMIT_CPU,(45,46))
    assert available_ram()>=16*1024**3
    raw=Path('/proc/self/stat').read_text();tail=raw[raw.rfind(')')+2:].split()
    report={'schema':'FastGeodis-two-fixed-synthetic-CPU-CUDA-checks-v1','status':'STARTED',
        'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pid':os.getpid(),'pgid':os.getpgrp(),'start_ticks':int(tail[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'fixture_scope':'Two fixed 9x9 float32 synthetic operator cases; no SAR/model/AP',
        'CPU_affinity':[0,1],'GPU_allocator_cap_bytes':4*1024**3,
        'process_peak_RSS_cap_bytes':16*1024**3,'minimum_available_RAM_bytes':16*1024**3,
        'CPU_seconds_soft_limit':45,'external_wall_timeout_required_seconds':90,
        'atol':1e-5,'rtol':0,'CPU_GPU_fullmap_equality_required':False,
        'no_CUDA_to_CPU_fallback':True,'scientific_training_update_credit':0,'cases':[]}
    with out.open('x') as f:json.dump(report,f,indent=2)
    started=time.monotonic()
    try:
        import torch
        torch.set_num_threads(2)
        assert torch.cuda.is_available(),'CUDA unavailable; CPU fallback is forbidden'
        props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0,4*1024**3/props.total_memory),device=0)
        torch.cuda.reset_peak_memory_stats(0)
        import FastGeodis
        assert importlib.metadata.version('FastGeodis')=='1.0.5'
        report.update(torch_version=str(torch.__version__),FastGeodis_version='1.0.5',
                      CUDA_build_version=torch.version.cuda,CUDA_device_name=props.name)
        def one_case(name,lamb):
            results={}
            for label in ('cpu','cuda:0'):
                image=torch.zeros((1,1,9,9),dtype=torch.float32)
                mask=torch.ones_like(image)
                seed=(4,4) if name=='euclidean' else (4,2)
                if name=='geodesic_step':image[:,:,:,5:]=1.0
                mask[0,0,seed[0],seed[1]]=0.0
                image=image.to(label).contiguous();mask=mask.to(label).contiguous()
                result=FastGeodis.generalised_geodesic2d(image,mask,1e10,lamb,iter=2)
                if label=='cuda:0':torch.cuda.synchronize(0)
                assert tuple(result.shape)==(1,1,9,9) and result.dtype==torch.float32
                assert result.device==torch.device(label)
                assert bool(torch.isfinite(result).all()) and bool((result>=0).all())
                distance=result.detach().cpu()
                seed_value=float(distance[0,0,seed[0],seed[1]])
                assert abs(seed_value)<=1e-5
                if name=='euclidean':
                    check_value=float(distance[0,0,4,5]);assert abs(check_value-1.0)<=1e-5
                else:
                    check_value=float(distance[0,0,4,7]);assert check_value>0.0
                results[label]={'seed_distance':seed_value,'checked_distance':check_value,
                                'map':distance.tolist()}
            cpu=torch.tensor(results['cpu']['map']);gpu=torch.tensor(results['cuda:0']['map'])
            delta=float((cpu-gpu).abs().max())
            return {'name':name,'shape':[1,1,9,9],'dtype':'float32','v':1e10,'lamb':lamb,
                    'iterations':2,'CPU_GPU_max_abs_delta_descriptive_only':delta,'devices':results}
        with torch.inference_mode():
            report['cases'].append(one_case('euclidean',0.0))
            report['cases'].append(one_case('geodesic_step',1.0))
        gc.collect();torch.cuda.synchronize(0);torch.cuda.empty_cache()
        allocated=torch.cuda.memory_allocated(0);peak=torch.cuda.max_memory_allocated(0)
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        assert allocated==0 and peak<=4*1024**3 and rss<16*1024**3
        assert available_ram()>=16*1024**3
        report.update(status='TWO_FIXED_SYNTHETIC_CPU_CUDA_CASES_PASSED',
            all_four_operator_calls_returned=True,CUDA_synchronized=True,
            torch_peak_allocated_bytes=peak,torch_allocated_after_cleanup=allocated,
            process_peak_RSS_bytes=rss,whole_GPU_memory_contract_certified=False,
            no_model_checkpoint_or_SAR_inputs=True,SAMRefiner_inference_or_AP=False)
    except BaseException as e:
        report.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error_type=type(e).__name__,error=str(e))
        raise
    finally:
        report.update(ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started)
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('cases','boot_id')}),flush=True)

if __name__=='__main__':main()
