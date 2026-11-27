"""Run in a fresh process against the deployed native libraries; optional --cuda13."""
import sys
import ctypes as c,numpy as np
from pathlib import Path
vendor = Path(__file__).resolve().parents[1] / 'vendor'
def lib(n):
    folder = vendor / {'geometry':'topology/native','signed':'t2_pial/t1/native',
                       'pial-field':'t2_pial/native'}[n]
    if '--cuda13' in sys.argv:
        folder = folder / 'cuda13'
    return c.CDLL(str(folder / (n+'.so')))
def ptr(x):return x.ctypes.data_as(c.c_void_p)
def ints(x):return np.array(x,np.int32)
def fl(x):return np.array(x,np.float32)
a=fl([[1,0,0],[3,0,0]])
lib('signed').smoothSigned(ptr(a),ptr(ints([0,1,2])),ptr(ints([1,0])),ptr(np.zeros(2,np.uint8)),2,1)
np.testing.assert_array_equal(a,[[2,0,0],[2,0,0]])
g=lib('geometry')
g.geomInit(3,1,*map(ptr,[ints([0,1,2]),ints([0,2,4,6]),ints([1,2,0,2,0,1]),ints([0,1,2,3]),ints([0,1,2]),fl(np.arctan(np.arange(100001)/100000))]))
dst=np.zeros(6,np.float32)
g.geomCompute(*map(ptr,[fl([0,1,0]),fl([0,0,1]),fl([0,0,0])]),3,0,1,3,c.c_float(1e-8),ptr(dst),ptr(fl([0]*7)),ptr(fl([0]*18)))
np.testing.assert_allclose(dst,[1,1,1,2**.5,1,2**.5])
mask=np.zeros((5,5,5),np.float32);mask[2,2,2]=1
out=np.empty_like(mask);err=c.create_string_buffer(512)
assert not lib('pial-field').pial_field(ptr(mask),ptr(out),5,5,5,c.c_float(1),1,c.byref(c.c_int()),err,512),err.value
assert np.isfinite(out).all() and np.ptp(out)>0
print('signed smoothing, geometry, pial field OK')
