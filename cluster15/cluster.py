"""Plan once on shared storage, then start a fixed rank on each allocated node."""
import argparse, fcntl, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
WORKFLOW=HERE.parent
sys.path.insert(0,str(WORKFLOW))
import batch_fastprep_anat as batch
def plan(a,path):
    root=a.root
    if path.exists(): raise SystemExit(f'Plan already exists: {path}')
    jobs=[]
    with ThreadPoolExecutor(16) as pool:
        for ds in batch.datasets(root,a.dataset):
            rel=ds.relative_to(root).as_posix()
            subjects=sorted(x for x in ds.glob('sub-*') if x.is_dir())
            units=[u for group in pool.map(batch.subject_units,subjects) for u,reason in group if reason=='eligible']
            for i in range(0,len(units),a.batch_size):
                jobs.append([rel,i//a.batch_size+1,units[i:i+a.batch_size]])
    if not jobs:raise SystemExit('No eligible sessions after exclusions')
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        json.dump(jobs,f); f.write('\n')
    try:os.link(f.name,path)
    finally:Path(f.name).unlink()
    print(f'{len(jobs)} batches, {sum(len(j[2]) for j in jobs)} sessions: {path}')
def run(a,path):
    jobs=json.loads(path.read_text())
    base=path.parent
    def assigned(slot):
        # A log number and GPU assignment refer to the index in the original plan.
        for index in range(a.rank*5+slot,len(jobs),15):
            ds,part,units=jobs[index]
            reason=batch.dataset_exclusion(ds)
            if reason:
                print(f'SKIP {index:07d} {ds} part {part}: {reason}',flush=True)
                continue
            yield index,ds,part,units
    lanes=[list(assigned(slot)) for slot in range(5)]
    if not any(lanes):return
    subprocess.run([str(HERE/'ensure-node.sh')],check=True)
    software = a.software.resolve()
    configured = os.path.expandvars(os.path.expanduser(json.loads(software.read_text())['python']['bfc']))
    python = str(WORKFLOW / configured) if '/' in configured else configured
    count=int(subprocess.check_output([python,'-c','import torch;print(torch.cuda.device_count())'],text=True).strip())
    mask=os.getenv('CUDA_VISIBLE_DEVICES')
    devices=mask.split(',') if mask is not None else list(map(str,range(count)))
    if count!=5 or len(devices)!=5 or len(set(devices))!=5:raise SystemExit(f'Expected 5 allocated GPUs; found {count}, mask={mask}')
    logs=base/'logs';logs.mkdir(exist_ok=True)
    lock=(base/f'.rank{a.rank}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def lane(slot):
        all_complete = True
        for index,ds,part,units in lanes[slot]:
            log=logs/f'{index:07d}.log';ok=log.with_suffix('.done')
            if ok.exists():continue
            args=[sys.executable,str(batch.SCRIPT),'worker','--root',str(a.root),'--dataset',ds,'--part',str(part),
                  '--units',','.join(units),'--isolate-session-errors','--workflow',str(WORKFLOW),'--software',str(software),
                  '--nextflow',str(HERE/'nextflow-one-gpu'),
                  '--work-root',str(base/'work'),'--out-root',str(base/'results'),'--surface-backend','gpu']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':devices[slot].strip(),'FASTPREP_GPU_SLOT':str(slot)}
            print(f'rank {a.rank} GPU {devices[slot]}: {ds} part {part}, {log}',flush=True)
            with log.open('a') as out:
                result=subprocess.run(args,env=env,stdout=out,stderr=subprocess.STDOUT)
            if result.returncode == 3:
                all_complete = False
                print(f'PARTIAL: {log}; continuing this GPU queue', file=sys.stderr, flush=True)
                continue
            if result.returncode:
                print(f'FAILED: {log}; lane stopped',file=sys.stderr,flush=True)
                return False
            ok.touch()
        return all_complete
    with ThreadPoolExecutor(5) as pool: outcomes=list(pool.map(lane,range(5)))
    if not all(outcomes):raise SystemExit(1)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('--dataset');p.add_argument('--batch-size',type=int,default=4)
    r=sub.add_parser('run');r.add_argument('--rank',type=int,required=True,choices=range(3))
    r.add_argument('--software',type=Path,default=os.environ.get('FASTPREP_SOFTWARE', WORKFLOW/'software.local.json'))
    for subparser in (p,r):
        subparser.add_argument('--run-dir',type=Path,required=True)
        subparser.add_argument('--root',type=Path,default=os.environ.get('FASTPREP_BIDS_ROOT','smri'))
    a=parser.parse_args();base=a.run_dir.resolve();a.root=a.root.expanduser().resolve()
    if base.is_relative_to(a.root):parser.error('Run directory must be outside raw inputs')
    path=base/'plan.json'
    if a.action=='plan':
        if a.batch_size<1:parser.error('batch size must be positive')
        base.mkdir(exist_ok=True,parents=True);plan(a,path)
    else:run(a,path)


if __name__ == "__main__":
    main()
