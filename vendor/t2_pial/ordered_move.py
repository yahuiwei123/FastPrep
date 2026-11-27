import ctypes

import torch

from sync_move import SyncMove


class OrderedMove(SyncMove):
    def __init__(self,faces,library):
        super().__init__(faces,library);self.ordered=ctypes.CDLL(str(library)).ordered
        ptr,num=ctypes.c_void_p,ctypes.c_int
        self.ordered.argtypes=[ptr,ptr,ptr,num,ptr,num,ptr,ptr,ptr,ptr];self.ordered.restype=num

    def ranks(self,current,delta):
        end=current+delta;low=torch.minimum(current,end);high=torch.maximum(current,end)
        origin=low.amin(0);width=(high.amax(0)-origin)/4;verge=(width.double()*.02).float()+.01
        def region(xyz):
            first=((xyz-verge-origin)/width).int().clamp_max(3);last=((xyz+verge-origin)/width).int().clamp_max(3)
            cell=first[:,0]*16+first[:,1]*4+first[:,2]
            return torch.where((first==last).all(1),cell,64)
        first,last=region(low[self.faces].amin(1)),region(high[self.faces].amax(1))
        cell=torch.where(first==last,first,64).repeat_interleave(3)
        index=self.faces.flatten();lowest=torch.full((len(current),),64,dtype=torch.int32,device=current.device)
        highest=torch.zeros_like(lowest);lowest.scatter_reduce_(0,index,cell,'amin');highest.scatter_reduce_(0,index,cell,'amax')
        shared=(highest!=lowest)|(highest==64)
        return torch.arange(len(current),device=current.device,dtype=torch.int32)+shared.int()*len(current)

    def seed(self,current,delta,keys,rank=None):
        end=current+delta;rank=self.ranks(current,delta) if rank is None else rank
        self.rank=rank
        allowed=torch.ones(len(current),device=current.device,dtype=torch.uint8);sweeps=[]
        for iteration in range(100):
            blocked=torch.zeros_like(rank)
            status=self.ordered(current.data_ptr(),end.data_ptr(),self.indices.data_ptr(),len(self.faces),keys.data_ptr(),len(keys),
                                rank.data_ptr(),allowed.data_ptr(),blocked.data_ptr(),torch.cuda.current_stream().cuda_stream)
            if status:raise RuntimeError('ordered trial kernel failed')
            update=(blocked==0).byte();changed=int((update!=allowed).sum());sweeps.append(changed);allowed=update
            if not changed:break
        else:raise ValueError('ordered proposals did not converge')
        return allowed.float(),{'ordered_changes':sweeps,'ordered_rejects':int((allowed==0).sum())}


class PriorityCommit(OrderedMove):
    def __init__(self,faces,library):
        super().__init__(faces,library);self.priority_commit=True


class NeighborCommit(PriorityCommit):
    def set_one_ring(self, offsets, neighbors):
        if hasattr(self, 'edges'):
            raise ValueError('movement one-ring already initialized')
        count = offsets.numel() - 1
        if (count <= 0 or offsets.ndim != 1 or neighbors.ndim != 1 or
                offsets.dtype != torch.int32 or neighbors.dtype != torch.int32 or
                offsets.device != self.faces.device or neighbors.device != self.faces.device):
            raise ValueError('invalid native one-ring layout')
        offsets = offsets.long()
        sizes = offsets.diff()
        if (offsets[0] != 0 or offsets[-1] != neighbors.numel() or (sizes < 0).any() or
                (neighbors < 0).any() or (neighbors >= count).any() or not self.faces.numel() or
                (self.faces < 0).any() or (self.faces >= count).any()):
            raise ValueError('invalid native one-ring bounds')
        source = torch.arange(count, device=offsets.device).repeat_interleave(sizes, output_size=neighbors.numel())
        other = neighbors.long()
        corners, adjacent = self.faces.flatten(), self.faces.roll(1, 1).flatten()
        expected = torch.unique(torch.cat((corners * count + adjacent, adjacent * count + corners)))
        if (source == other).any() or not torch.equal((source * count + other).sort().values, expected):
            raise ValueError('native one-ring does not match movement faces')
        self.edges = source, other
        self.vertex_count = count

    @staticmethod
    def dot(first,second):
        terms=first*second
        return terms[:,0]+terms[:,1]+terms[:,2]

    def clip(self,delta,limit):
        length=self.dot(delta,delta).sqrt().double()
        return (delta.double()*(limit/length.clamp_min(1e-30)).clamp_max(1)[:,None]).float()

    def project_neighbors(self, delta, vertices, relative, distance):
        if (distance == 0).any():
            raise ValueError('coincident one-ring vertices: undefined projection direction')
        updated = delta.clone()
        if not vertices.numel():
            return updated, 0
        direction = relative / distance[:, None]

        def project(indices, unit):
            remaining = updated[indices]
            along = self.dot(remaining, unit)
            positive = along > 0
            changed = indices[positive]
            updated[changed] = remaining[positive] - along[positive, None] * unit[positive]
            return changed.numel()

        if not (vertices[1:] == vertices[:-1]).any():
            return updated, project(vertices, direction)
        _, sizes = torch.unique_consecutive(vertices, return_counts=True)
        starts = sizes.cumsum(0) - sizes
        corrections = 0
        for depth in range(int(sizes.max())):
            positions = starts[sizes > depth] + depth
            corrections += project(vertices[positions], direction[positions])
        return updated, corrections

    def plan(self,current,delta):
        if len(current) != getattr(self, 'vertex_count', None):
            raise ValueError('movement state does not match native one-ring')
        rank=self.ranks(current,delta)
        source,other=self.edges;offset=current[other]-current[source]
        close=self.dot(offset,offset)<=(.0101+self.dot(delta[other],delta[other]).sqrt()).square()
        source,other=source[close],other[close];trial=delta;corrections=[]
        for iteration in range(12):
            keys=self.pairs(current,current+trial);scales,extra=self.seed(current,trial,keys,rank)
            moved=(rank[other]<rank[source])&(scales[other]!=0)
            relative=torch.where(moved[:,None],current[other]+trial[other],current[other])-current[source]
            distance=self.dot(relative,relative).sqrt();near=distance<=.01
            updated, changed = self.project_neighbors(delta, source[near], relative[near], distance[near])
            corrections.append(changed)
            if torch.equal(trial,updated):return trial,keys,scales,dict(**extra,neighbor_corrections=corrections)
            trial=updated
        raise ValueError('near-neighbor correction did not converge')
