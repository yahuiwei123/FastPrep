import ctypes
import time

import torch

from collision_screen import box_pairs


class SyncMove:
    def __init__(self, faces, library):
        if not faces.is_cuda or faces.ndim!=2 or faces.size(1)!=3:
            raise ValueError('CUDA triangle mesh required')
        self.faces = faces.long().contiguous()
        self.indices = faces.int().contiguous()
        self.classify = ctypes.CDLL(str(library)).classify
        pointer, integer = ctypes.c_void_p,ctypes.c_int
        self.classify.argtypes = [pointer,pointer,integer,pointer,integer,pointer,pointer]
        self.classify.restype = integer

    def pairs(self, start, end):
        lower = torch.minimum(start,end)[self.faces].amin(1).double()-.0011
        upper = torch.maximum(start,end)[self.faces].amax(1).double()+.0011
        keys = [torch.minimum(left,right)*len(self.faces)+torch.maximum(left,right)
                for left,right in box_pairs(lower,upper,self.faces)]
        return torch.unique(torch.cat(keys)) if keys else self.faces.new_empty(0)

    def hits(self, xyz, keys):
        if xyz.dtype!=torch.float32 or not xyz.is_contiguous() or keys.dtype!=torch.int64 or not keys.is_contiguous():
            raise ValueError('wrong kernel tensor layout')
        hits = torch.empty(len(keys),dtype=torch.uint8,device=xyz.device)
        status = self.classify(xyz.data_ptr(),self.indices.data_ptr(),len(self.faces),keys.data_ptr(),len(keys),
                               hits.data_ptr(),torch.cuda.current_stream(xyz.device).cuda_stream)
        if status: raise RuntimeError('CUDA triangle tests failed: '+str(status))
        return hits.bool()

    def seed(self,current,delta,keys):
        return torch.ones_like(current[:,0]),{}

    def clip(self,delta,limit):
        length=delta.square().sum(1,keepdim=True).double().sqrt()
        return delta*(limit/length.clamp_min(1e-30)).clamp_max(1).float()

    def plan(self,current,delta):
        keys=self.pairs(current,current+delta);scales,extra=self.seed(current,delta,keys)
        return delta,keys,scales,extra

    @torch.inference_mode()
    def step(self, current, gradient, old, frozen, dt, momentum, limit):
        if not all(value.is_cuda and value.dtype==torch.float32 and value.shape==current.shape and
                   value.isfinite().all() for value in (current,gradient,old)) or limit<=0:
            raise ValueError('invalid movement state')
        torch.cuda.synchronize(); start=time.perf_counter()
        delta = dt*gradient+momentum*old
        delta = self.clip(delta,limit)
        delta[frozen] = 0
        delta,keys,scales,extra = self.plan(current,delta)
        initial = self.hits(current,keys)
        audits=[]; fallback=False
        torch.cuda.synchronize(); broad_end=time.perf_counter()
        for attempt in range(18):
            motion = scales[:,None]*delta
            middle, end = current+.5*motion,current+motion
            mid_hits,end_hits = self.hits(middle,keys),self.hits(end,keys)
            priority=getattr(self,'priority_commit',False)
            new = keys[(end_hits if priority else (mid_hits|end_hits))&~initial]
            audits.append(len(new))
            if not len(new): break
            if priority:raise ValueError('priority commit produced new endpoint collisions')
            selected = torch.unique(self.faces[torch.cat((new//len(self.faces),new%len(self.faces)))])
            if attempt<5: scales[selected] *= .5
            else: scales[selected] = 0
        else:
            fallback=True;scales.zero_();motion=torch.zeros_like(delta);middle=end=current;mid_hits=end_hits=initial
        torch.cuda.synchronize(); finish=time.perf_counter()
        trace = dict(candidates=len(keys),initial_pairs=int(initial.sum()),unresolved_per_try=audits,
                     reduced=int(((scales<1)&(delta!=0).any(1)).sum()),global_fallback=fallback,
                     broad_and_initial_seconds=broad_end-start,check_backtrack_seconds=finish-broad_end,
                     move_seconds=finish-start,path='priority-commit' if priority else 'straight-sampled',**extra)
        return end,motion,scales,trace,(keys[initial],keys[mid_hits],keys[end_hits])
