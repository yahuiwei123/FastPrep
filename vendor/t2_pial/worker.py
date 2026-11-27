import argparse
import itertools
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch


def nint(value):
    return (value.sign() * (value.abs().double() + .5).floor()).long()


class Volume:
    def __init__(self, root, name, maps, device):
        image = nib.load(root / (name + '.mgh'))
        self.data = torch.as_tensor(np.asarray(image.dataobj).astype('float32'), device=device)
        self.outside, self.spacing, *matrix = maps[name]
        self.matrix = torch.tensor(matrix, device=device, dtype=torch.float64).reshape(3, 4)
        self.limit = torch.tensor(self.data.shape, device=device) - 1

    def vox(self, xyz):
        return (xyz.float().double() @ self.matrix[:, :3].T + self.matrix[:, 3]).float().double()

    def valid(self, index):
        return ((index >= 0) & (index <= self.limit)).all(dim=-1)

    def at(self, index):
        safe = index.clamp(min=0).minimum(self.limit)
        return self.data[safe[..., 0], safe[..., 1], safe[..., 2]]

    def label(self, xyz):
        idx = nint(self.vox(xyz))
        return torch.where(self.valid(idx), self.at(idx), self.outside)

    def sample(self, xyz):
        coord = self.vox(xyz)
        safe = coord.clamp(min=0).minimum(self.limit)
        lower = safe.long()
        frac = safe - lower
        total = torch.zeros_like(frac[..., 0])
        for corner in itertools.product(range(2), repeat=3):
            idx = lower + lower.new_tensor(corner)
            weight = [frac[..., axis] if bit else 1-frac[..., axis] for axis, bit in enumerate(corner)]
            total += weight[0] * weight[1] * weight[2] * self.at(idx)
        return torch.where(self.valid(coord.round()), total, self.outside)


class Frame:
    def __init__(self, root, device):
        self.root, self.device = root, device
        self.maps = {row[0]: list(map(float, row[1:])) for row in (line.split() for line in (root/'matrices.txt').read_text().splitlines())}
        self.input = torch.tensor(np.fromfile(root/'input.bin', '<f4').reshape(-1, 21), device=device)
        self.step, self.maxout, self.sigma, half, left = np.loadtxt(root/'config.txt')
        self.half, self.left = int(half), bool(left)
        self.t2, self.seg = self.vol('t2'), self.vol('rawseg')
        t1=self.vol('t1')
        if any(volume.data.shape!=self.t2.data.shape or not torch.equal(volume.matrix,self.t2.matrix) for volume in (t1,self.seg)):
            raise ValueError('experiment requires identical T1/T2/aseg grids')
        reset = ((self.seg.data == 3) | (self.seg.data == 42)) & (1.1 * t1.data.double() > self.t2.data)
        self.seg.data[reset] = 0
        self.seg_diff = int((self.seg.data != self.vol('seg').data).sum())
        if self.seg_diff:
            raise ValueError(f'{self.seg_diff} differing aseg voxels')

    def vol(self, name):
        return Volume(self.root, name, self.maps, self.device)


def histograms(frame):
    ids = torch.where(frame.input[:, 12] == 0)[0]
    centers = nint(frame.t2.vox(frame.input[ids, :3]))
    offsets = torch.arange(-frame.half, frame.half + 1, device=frame.device)
    offsets = torch.cartesian_prod(offsets, offsets, offsets)
    if len(offsets) > 65535:
        raise ValueError('histogram count exceeds interchange capacity')
    with (frame.root/'gpu.hist').open('wb') as output:
        for vertex_ids, center in zip(ids.split(256), centers.split(256)):
            coords = center[:, None] + offsets
            valid = frame.t2.valid(coords)
            signal = frame.t2.at(coords)
            seg = frame.seg.at(coords)
            lower = torch.where(valid, signal, torch.inf).amin(1)
            upper = torch.where(valid, signal, -torch.inf).amax(1)
            sizes = nint((upper-lower).double()+1)
            if sizes.min() < 2 or sizes.max() > 10000:
                raise ValueError('unsupported local histogram range')
            width = ((upper.double()-lower)/(sizes-1)).float()
            bins = nint((signal-lower[:, None])/width[:, None]).clamp(min=0).minimum(sizes[:, None]-1)
            valid_counts = []
            for labels in ((3, 42), (2, 41)):
                counts = torch.zeros((len(center), int(sizes.max())), device=frame.device, dtype=torch.int32)
                counts.scatter_add_(1, bins, (((seg==labels[0]) | (seg==labels[1])) & valid).int())
                valid_counts.append(counts.cpu().numpy().astype('<u2'))
            headers = torch.stack([vertex_ids, lower, upper, sizes], dim=1).float().cpu().numpy()
            for row, header in enumerate(headers):
                for counts in valid_counts:
                    output.write(header.tobytes()); output.write(counts[row, :int(header[3])].tobytes())
    return {'vertices':len(ids), 'seg_voxel_mismatches':frame.seg_diff}


def search_chunk(frame, state, fit, volumes):
    white_dist, pial_dist, left_dist, right_dist = volumes
    xyz, white, normal, old_pial = (state[:, start:start+3].double() for start in (0,3,6,9))
    delta = (state[:, :3] - state[:, 3:6]).double()
    thickness = delta.square().sum(1).sqrt()
    direction = delta / thickness.clamp(min=1e-30)[:, None]
    active = (state[:, 12] == 0) & (thickness >= np.finfo('f4').eps)
    out = state[:, 13:21].clone(); out[:, :3] = xyz
    mean, sigma, low, high, extlow, exthigh = fit.T
    contra_labels = [42,41,53,54,51,52,50,43,44,47,46,49,62,63,60,58] if frame.left else [3,2,17,18,12,13,11,4,5,8,7,10,30,31,28,26]
    def contra(labels):
        return torch.isin(labels, labels.new_tensor(contra_labels))
    def cerebellum(labels):
        return (labels == 8) | (labels == 47)
    def pick(mask, yes, no):
        return torch.where(mask[:, None], yes, no)
    was_out = torch.zeros_like(active); inward = torch.zeros_like(active)
    previous = torch.zeros_like(thickness); dist = (thickness/2).clamp(max=.5)
    searching = active.clone(); step = frame.step
    for _ in range(int(thickness.max().item()/step)+2):
        eligible = searching & (dist <= thickness)
        if not eligible.any(): break
        point = white + dist[:, None]*direction
        signal = frame.t2.sample(point); white_val = white_dist.sample(point)
        searching &= ~(eligible & (signal > 0) & (white_val <= 0))
        check = eligible & (signal > 0) & (white_val > 1) & ~((signal < low) & (signal > nint(extlow*.75)) & ~was_out)
        first = check & ~was_out & (signal >= low)
        previous = torch.where(first, signal, previous); was_out |= first
        decrease = (signal < previous) & (signal < mean-2*sigma)
        increase = (signal > previous) & (signal > mean+2*sigma) & (frame.t2.sample(white+(dist+1)[:, None]*direction) < low)
        fail = check & (contra(frame.seg.label(point)) | ((white_val > frame.seg.spacing) & (decrease | increase)) | (signal<low) | (signal>high))
        out[:, :3] = pick(fail, point-direction*step/2, out[:, :3]); out[:, 3] = torch.where(fail, dist+step/2-thickness, out[:, 3]); out[:, 4] = torch.where(fail, 1, out[:, 4]); out[:, 5] += fail
        previous = torch.where(check, signal, previous); inward |= fail; searching &= ~fail
        dist += step

    outward = active & ~inward; live = outward.clone(); left_white = torch.zeros_like(live); left_pial = torch.zeros_like(live)
    last_white = torch.zeros_like(thickness); last_pial = pial_dist.sample(xyz)
    success = torch.zeros_like(live); hit = torch.zeros_like(thickness); dist = 0.
    while dist <= frame.maxout:
        point = xyz + dist*normal; white_val = white_dist.sample(point); signal = frame.t2.sample(point); pial_val = pial_dist.sample(point)
        left_white |= ~white_dist.valid(nint(white_dist.vox(point))) | (white_val > 0)
        last_white = torch.where(live & ~left_white, dist, last_white)
        check = live & left_white
        deep = pial_val - last_pial < -step
        found = check & deep & ~((signal > extlow) & (signal < exthigh))
        check &= ~deep; past = pial_val > pial_dist.spacing
        reenter = ~past & left_pial & ((pial_val < pial_dist.spacing) | (pial_val < last_pial))
        stop = (live & left_white & deep) | (check & reenter)
        left_pial |= check & past; check &= ~reenter; last_pial = torch.where(check,pial_val,last_pial)
        reenter_white = check & (white_val <= 0)
        stop |= reenter_white & (dist-last_white > .5); last_white = torch.where(reenter_white,dist,last_white)
        check &= ~stop & (signal >= 0); label = frame.seg.label(point)
        unknown = (label == 0)|(label == 255)|(label == 190)|(label == 191)
        same, opposite = (left_dist, right_dist) if frame.left else (right_dist, left_dist)
        hemi_stop = contra(label) | (unknown & (opposite.sample(point) <= same.sample(point)+.1))
        found |= check & (hemi_stop | (~cerebellum(label) & (((signal < low) & (white_val > 1.2)) | (signal > high))))
        stop |= check & cerebellum(label); hit = torch.where(found,dist-step,hit); success |= found; live &= ~(stop | found)
        dist += step

    probe = xyz + .5*normal; label = frame.seg.label(probe); signal = frame.t2.sample(probe)
    nudge = outward & ~success & ~cerebellum(label) & ~contra(label) & (signal >= extlow) & (signal <= exthigh)
    move = success | nudge; hit = torch.where(nudge,.5,hit)
    target = pick(move, xyz+hit[:, None]*normal,pick(state[:, 18]>0,xyz,old_pial))
    out[:, :3] = pick(outward,target,out[:, :3]); out[:, 3] = torch.where(outward,hit,out[:, 3]); out[:, 4]=torch.where(outward,success.float(),out[:, 4]); out[:, 5] += move
    out[:, 6] = torch.where(active,frame.t2.sample(out[:, :3]),out[:, 6]); out[:, 7]=torch.where(active,frame.sigma,out[:, 7]); return out, int(inward.sum()), int(move.sum())


def search(frame):
    fits = np.fromfile(frame.root/'fits.bin','<f8').reshape(-1,7)
    active = np.flatnonzero(frame.input[:,12].cpu().numpy()==0)
    if not np.array_equal(fits[:,0],active) or not np.isfinite(fits).all(): raise ValueError('invalid fitted thresholds')
    full = np.zeros((len(frame.input),6)); full[active]=fits[:,1:]
    volumes = [frame.vol(key) for key in ('white','pial','lh','rh')]
    fit = torch.as_tensor(full,device=frame.device)
    inward, outward = 0, 0
    with (frame.root/'gpu.fields').open('wb') as output:
        for data, values in zip(frame.input.split(16384),fit.split(16384)):
            fields, inside, outside = search_chunk(frame,data,values,volumes)
            inward += inside; outward += outside
            if not fields.isfinite().all(): raise ValueError('non-finite result')
            output.write(fields.cpu().numpy().astype('<f4').tobytes())
    np.array([inward,outward],dtype='<i4').tofile(frame.root/'gpu.counts')
    return {'vertices':len(full), 'seg_voxel_mismatches':frame.seg_diff, 'num_in':inward, 'num_out':outward}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['hist','search']); parser.add_argument('frame',type=Path); args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter()
    with torch.inference_mode():
        frame=Frame(args.frame,'cuda'); result=(histograms if args.stage=='hist' else search)(frame); torch.cuda.synchronize()
    result.update(seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    (args.frame/(args.stage+'.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
