import argparse
import torch
from pathlib import Path
from pytorch3d import _C
from pytorch3d.ops import knn_points
parser = argparse.ArgumentParser(description="Exercise CUDA KNN with the configured SUGAR interpreter")
parser.add_argument("--expected-directory", type=Path, help="Also check the location of the loaded PyTorch3D extension")
expected = parser.parse_args().expected_directory
if expected and Path(_C.__file__).resolve().parent != expected.resolve():
    raise RuntimeError(f"Unexpected PyTorch3D extension: {_C.__file__}")
q=torch.tensor([[[.9,0,0],[0.,2.,0]]],device='cuda:0')
p=torch.tensor([[[0.,0.,0],[1.,0.,0],[0.,1.,0]]],device='cuda:0')
r=knn_points(q,p,K=2)
torch.cuda.synchronize()
assert r.idx.tolist()==[[[1,0],[2,0]]]
torch.testing.assert_close(r.dists,torch.tensor([[[.01,.81],[1.,4.]]],device='cuda:0'))
print(torch.cuda.get_device_name(),torch.__version__,_C.__file__,r.idx.tolist(),flush=True)
