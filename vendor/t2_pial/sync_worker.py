import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from sync_move import SyncMove


def main():
    root=Path('sync-moves');torch.set_num_threads(1);faces=np.fromfile(root/'faces.i32','<i4').reshape(-1,3)
    from ordered_move import OrderedMove,PriorityCommit,NeighborCommit
    mode = os.environ.get('SYNC_ORDERED', 'neighbor')
    engine={'0':SyncMove,'1':OrderedMove,'commit':PriorityCommit,'neighbor':NeighborCommit}[mode](torch.tensor(faces,device='cuda'),sys.argv[1]);step=0
    if isinstance(engine, NeighborCommit):
        offsets, neighbors = (torch.tensor(np.fromfile(root / name, '<i4'), device=engine.faces.device)
                              for name in ('offsets.i32', 'neighbors.i32'))
        engine.set_one_ring(offsets, neighbors)
        del offsets, neighbors
    with socket.socket(fileno=os.dup(0)) as channel:
        while True:
            token=channel.recv(1)
            if not token:break
            if token!=b'M':raise ValueError('invalid move token')
            folder=root/f'step-{step}';nverts,dt,momentum,limit=map(float,(folder/'config.txt').read_text().split())
            state=torch.tensor(np.fromfile(folder/'input.f32','<f4').reshape(int(nverts),10),device='cuda')
            pos=state[:,:3].contiguous();end,motion,scales,trace,pairs=engine.step(pos,state[:,3:6],state[:,6:9],state[:,9]!=0,dt,momentum,limit)
            for label,keys in zip(('start','mid','end'),pairs):keys.cpu().numpy().tofile(folder/(label+'.keys'))
            if getattr(engine,'priority_commit',False):engine.rank.cpu().numpy().tofile(folder/'rank.i32')
            result=torch.cat((end,motion,(scales<1)[:,None]),dim=1);result.cpu().numpy().tofile(folder/'output.f32')
            (folder/'trace.json').write_text(json.dumps(trace,indent=2)+'\n');channel.sendall(token);step+=1


if __name__=='__main__':main()
