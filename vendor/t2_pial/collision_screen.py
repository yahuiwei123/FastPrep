import torch


def potential_vertices(xyz, faces, radius, cell_mm=1.0):
    triangles = xyz[faces]
    lower = (triangles-radius[faces, None]).amin(1)
    upper = (triangles+radius[faces, None]).amax(1)
    possible = torch.zeros(len(xyz), device=faces.device, dtype=torch.bool)
    for left,right in box_pairs(lower,upper,faces,cell_mm):
        possible.index_fill_(0,faces[left].flatten(),True)
        possible.index_fill_(0,faces[right].flatten(),True)
    return possible


def box_pairs(lower, upper, faces, cell_mm=1.0):
    first = (lower/cell_mm).floor().long()
    last = (upper/cell_mm).floor().long()
    offset = first.amin(0)
    dims = last.amax(0)-offset+1
    spans = last-first+1
    counts = spans.prod(1)
    if int(counts.sum()) > 8_000_000:
        raise ValueError('grid exceeds replay budget; do not use a partial mask')
    starts = counts.cumsum(0)-counts
    ids = torch.arange(len(faces), device=faces.device).repeat_interleave(counts)
    local = torch.arange(len(ids), device=faces.device)-starts[ids]
    shape = spans[ids]
    cells = first[ids]-offset+torch.stack((local//(shape[:,1]*shape[:,2]), local//shape[:,2]%shape[:,1], local%shape[:,2]),1)
    keys, order = ((cells[:,0]*dims[1]+cells[:,1])*dims[2]+cells[:,2]).sort()
    ids = ids[order]
    _, occupancy = torch.unique_consecutive(keys, return_counts=True)
    remaining = occupancy.cumsum(0).repeat_interleave(occupancy)-torch.arange(len(ids), device=faces.device)-1
    pair_end = remaining.cumsum(0)
    total = int(pair_end[-1])
    if total > 60_000_000:
        raise ValueError('pair enumeration exceeds replay budget; no truncation permitted')
    for start in range(0,total,300_000):
        serial = torch.arange(start,min(total,start+300_000),device=faces.device)
        lhs = torch.searchsorted(pair_end,serial,right=True)
        rhs = lhs+1+serial-(pair_end[lhs]-remaining[lhs])
        left,right = ids[lhs],ids[rhs]
        overlap = (lower[left]<=upper[right]).all(1) & (lower[right]<=upper[left]).all(1)
        left,right = left[overlap],right[overlap]
        independent = ~(faces[left,:,None]==faces[right,None,:]).any(dim=(1,2))
        yield left[independent],right[independent]
