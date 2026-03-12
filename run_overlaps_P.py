import numpy as np

import glob
import os

import yastn
from yastn.tn import mps
from scripts_fermions.operators import Hamiltonian


Nas = [(512, 0.125)] # [(256, 0.125), (512, 0.0625), (512, 0.125)]
ms = [0, 0.5]
g = 1
sg2 = 0.25
x0 = 1.5
#
PPs = np.linspace(0, 6, 49)
#
m = 0.0
N, a = 512, 0.0625
#
ts = np.linspace(0, N * a / 2, 17)
Ps = [0, 1, 2, 3, 4]
Ds = [256, 512]

def fn_mass(fns, **kwargs):
    return [x for x in fns if all( f"{k}={v}" in x for k, v in kwargs.items())][0]


for N, a in Nas:
    for m in ms:
        ts = np.linspace(0, N * a / 2, 17)

        glob_path = os.path.join(os.path.abspath(""), "./results_fermions/g=1.0000/**/state_t=*.npy")
        fnames = glob.glob(glob_path, recursive=True)
        fns = sorted([fname for fname in fnames if all(x in fname for x in ["/g=1.0000/", "dt=0.0625", f"{N=}", f"{a=:0.4f}", f"{x0=:0.4f}", f"{sg2=:0.4f}", f"{m=:0.4f}"])])
        psi_t = {(t, P, D): yastn.from_dict(np.load(fn_mass(fns, t=t, P=P, D=D), allow_pickle=True).item()['psi']) for t in ts for P in Ps for D in Ds}
        # #
        # D = 128
        ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
        # #
        # data = np.load(f"./results_fermions/construct_P_{m=}_{N=}_{a=}.npy", allow_pickle=True).item()
        # psi_P = data['state']
        # eng_P = data['energy']
        # for k, v in psi_P.items():
        #     psi_P[k] = yastn.from_dict(v)

        # probs_R = {}
        # for (t, P, D), pt in psi_t.items():
        #     for PP, pc in psi_P.items():
        #         if (t, P, D, PP) not in probs_R:
        #             print(t, P, D, PP)
        #             probs_R[t, P, D, PP] = np.abs(mps.vdot(pt, pc)) ** 2


        # fname = f"./results_fermions/probs_gauss_R_{m=}_{N=}_{a=}.npy"
        # print(fname)
        # np.save(fname, probs_R, allow_pickle=True)
        # print(fname)


        D= 128
        glob_path = os.path.join(os.path.abspath(""), "./results_fermions/g=1.0000/**/gs/**/state_D=*.npy")
        fnames = glob.glob(glob_path, recursive=True)

        # print(fnames)
        print(N,a,m)
        fgs = sorted([fname for fname in fnames if all(x in fname for x in ["/g=1.0000/", f"{N=}", f"{a=:0.4f}", f"{D=}", f"{m=:0.4f}"])])

        d_gs = np.load(fn_mass(fgs, m=m), allow_pickle=True).item()
        psi_gs = yastn.from_dict(d_gs['psi'])
        probs_gs = {(t, PP, D): np.abs(mps.vdot(pt, psi_gs)) ** 2 for (t, PP, D), pt in psi_t.items()}


        fname = f"./results_fermions/probs_gs_gauss_{m=}_{N=}_{a=}.npy"
        print(fname)
        np.save(fname, probs_gs, allow_pickle=True)
        print(fname)


        # H = Hamiltonian(N, m, g, t=0, a=a, v=1, Q=0, ops=ops)
        # E_t = {k: mps.vdot(v, H, v) for k, v in psi_t.items()}

        # fname = f"./results_fermions/Eng_gauss_{m=}_{N=}_{a=}.npy"
        # np.save(fname, E_t, allow_pickle=True)
