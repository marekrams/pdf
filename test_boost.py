import argparse
import csv
import glob
import os.path
from pathlib import Path
import time

import numpy as np

import ray
import yastn
import yastn.tn.mps as mps
from scripts_fermions.operators import HNN, sum_Ln2,nHNN, sum_nLn2



if __name__ == "__main__":
    #
    g = 1
    v, Q = 1, 1
    N = 128
    D = 64
    a = 0.125
    m = 0.
    #
    glob_path = os.path.join(os.path.abspath(""), "./results_fermions/g=1.0000/**/ex/**/state_D=*.npy")
    fnames = glob.glob(glob_path, recursive=True)
    fex = sorted([fname for fname in fnames if all(x in fname for x in ["/g=1.0000/", f"{N=}", f"{a=:0.4f}", f"{D=}"])])
    #
    glob_path = os.path.join(os.path.abspath(""), "./results_fermions/g=1.0000/**/gs/**/state_D=*.npy")
    fnames = glob.glob(glob_path, recursive=True)
    fgs = sorted([fname for fname in fnames if all(x in fname for x in ["/g=1.0000/", f"{N=}", f"{a=:0.4f}", f"{D=}"])])

    def fn_mass(fns, **kwargs):
        return [x for x in fns if all( f"{k}={v}" in x for k, v in kwargs.items()) ][0]

    d_ex = np.load(fn_mass(fex, m=m), allow_pickle=True).item()
    d_gs = np.load(fn_mass(fgs, m=m), allow_pickle=True).item()


    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')


    H0 = HNN(N, a, m, ops=ops)
    e0 = a * g * g / 2
    H1 = e0 * sum_Ln2(N, t=0, a=a, v=1, Q=0, ops=ops)
    H = H0 + H1

    K0 = nHNN(N, a, m, ops=ops)
    K1 = e0 * sum_nLn2(N, ops=ops)
    K = K0 + K1

    psi_ex = yastn.from_dict(d_ex['psi'])
    psi_gs = yastn.from_dict(d_gs['psi'])

    E_ex = mps.vdot(psi_ex, H, psi_ex)
    E_gs = mps.vdot(psi_gs, H, psi_gs)

    print(E_ex)
    print(E_gs)

    psi = psi_ex.copy()

    opts_svd={"D_total": 64}
    Ptimes = np.linspace(0, 1, 21)

    opts_expmv = {'hermitian': True, 'tol': 1e-10}

    evol = mps.tdvp_(psi, [K0, K1], Ptimes, method='12site', opts_svd=opts_svd, dt=0.05, yield_initial=True, precompute=False, subtract_E=True, opts_expmv=opts_expmv)

    EE = []
    EE2 = []


    for info in evol:
        EE.append(mps.vdot(psi, H, psi))
        EE2.append(mps.vdot(psi, H @ H, psi))
        print(f"{info.tf:0.2f}", EE[-1].real, EE2[-1].real - (EE[-1].real) ** 2)
