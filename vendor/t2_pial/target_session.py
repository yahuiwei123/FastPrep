import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from worker import Frame, histograms, search


def run(root, channel):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        start = time.perf_counter()
        frame = Frame(root, 'cuda')
        for stage, operation, token in [('hist', histograms, b'H'), ('search', search, b'S')]:
            if stage == 'search':
                if channel.recv(1) != token:
                    raise RuntimeError('missing search barrier')
                start = time.perf_counter()
                maps = {name: list(map(float, values)) for name, *values in
                        (line.split() for line in (root/'matrices.txt').read_text().splitlines())}
                if any(maps.get(name) != values for name, values in frame.maps.items()):
                    raise ValueError('frame geometry changed between histogram and search')
                frame.maps = maps
            result = operation(frame)
            torch.cuda.synchronize()
            result.update(seconds=time.perf_counter()-start, gpu=torch.cuda.get_device_name(),
                          torch=torch.__version__, session_pid=os.getpid())
            (root/(stage+'.json')).write_text(json.dumps(result, indent=2)+'\n')
            channel.sendall(token)


if __name__ == '__main__':
    with socket.socket(fileno=os.dup(0)) as channel:
        run(Path(sys.argv[1]), channel)
