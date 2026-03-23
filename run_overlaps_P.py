import numpy as np

import glob
import os

import yastn
from yastn.tn import mps
from scripts_fermions.operators import Hamiltonian


Nas = [(256, 0.125)] # [(256, 0.125), (512, 0.0625), (512, 0.125)]
ms = [0, 0.5]
g = 1
# sg2 = 0.25
x0 = 3.0
#
Ps = [0, 1, 2, 3, 4]
Ds = [256]

def fn_mass(fns, **kwargs):
    return [x for x in fns if all( f"{k}={v}" in x for k, v in kwargs.items())][0]


for sg2 in [0.5, 0.25, 0.125, 0.0625]:
    for N, a in Nas:
        for m in ms:
            ts = np.linspace(0, N * a / 2, 9)

            glob_path = os.path.join(os.path.abspath(""), "./results_fermions/g=1.0000/**/state_t=*.npy")
            fnames = glob.glob(glob_path, recursive=True)
            fns = sorted([fname for fname in fnames if all(x in fname for x in ["/g=1.0000/", "dt=0.0625", f"{N=}", f"{a=:0.4f}", f"{x0=:0.4f}", f"{sg2=:0.4f}", f"{m=:0.4f}"])])
            psi_t = {(t, P, D): yastn.from_dict(np.load(fn_mass(fns, t=t, P=P, D=D), allow_pickle=True).item()['psi']) for t in ts for P in Ps for D in Ds}
            # #
            D = 128
            ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
            # #
            data = np.load(f"./results_fermions/construct_P_{m=:0.1f}_{N=}_{a=}.npy", allow_pickle=True).item()
            psi_P = data['state']
            eng_P = data['energy']
            for k, v in psi_P.items():
                psi_P[k] = yastn.from_dict(v)

            probs_R = {}
            for (t, P, D), pt in psi_t.items():
                for PP, pc in psi_P.items():
                    if (t, P, D, PP) not in probs_R:
                        print(t, P, D, PP)
                        probs_R[t, P, D, PP] = np.abs(mps.vdot(pt, pc)) ** 2

            fname = f"./results_fermions/probs_gauss_R_{m=}_{N=}_{a=}_{x0=:0.4f}_{sg2=:0.4f}.npy"
            print(fname)
            np.save(fname, probs_R, allow_pickle=True)
            print(fname)
