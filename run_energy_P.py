import numpy as np

import glob
import os

import yastn
from yastn.tn import mps
from scripts_fermions.operators import Hamiltonian


Nas = [(256, 0.125), (512, 0.0625), (512, 0.125)]
ms = [0.0, 0.5]
g = 1
sg2 = 0.25
x0 = 1.5
#
PPs = np.linspace(0, 6, 49)
Ds = [256, 512]
#
#
Ps = [0, 1, 2, 3, 4]

def fn_mass(fns, **kwargs):
    return [x for x in fns if all( f"{k}={v}" in x for k, v in kwargs.items())][0]


for N, a in Nas:
    for m in ms:
        #
        tmp = np.load(f"./results_fermions/construct_P_{m=:0.1f}_{N=}_{a=}.npy", allow_pickle=True).item()
        E_P = tmp['energy']
        #
        ts = np.linspace(0, N * a / 2, 17)
        #
        glob_path = os.path.join(os.path.abspath(""), "./results_fermions/g=1.0000/**/state_t=*.npy")
        fnames = glob.glob(glob_path, recursive=True)
        fns = sorted([fname for fname in fnames if all(x in fname for x in ["/g=1.0000/", "dt=0.0625", f"{N=}", f"{a=:0.4f}", f"{x0=:0.4f}", f"{sg2=:0.4f}", f"{m=:0.4f}"])])
        psi_t = {(t, P, D): yastn.from_dict(np.load(fn_mass(fns, t=t, P=P, D=D), allow_pickle=True).item()['psi']) for t in ts for P in Ps for D in Ds}
        #
        ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
        #
        D= 128
        #
        glob_path = os.path.join(os.path.abspath(""), "./results_fermions/g=1.0000/**/gs/**/state_D=*.npy")
        fnames = glob.glob(glob_path, recursive=True)
        fgs = sorted([fname for fname in fnames if all(x in fname for x in ["/g=1.0000/", f"{N=}", f"{a=:0.4f}", f"{D=}", f"{m=:0.4f}"])])
        d_gs = np.load(fn_mass(fgs, m=m), allow_pickle=True).item()
        psi_gs = yastn.from_dict(d_gs['psi'])
        #
        H = Hamiltonian(N, m, g, t=0, a=a, v=1, Q=0, ops=ops)
        E_t = {k: mps.vdot(v, H, v).item().real for k, v in psi_t.items()}
        Egs = mps.vdot(psi_gs, H, psi_gs).item().real
        I, cp, cm = ops.I(), ops.cp(), ops.c()
        id = mps.generate_mpo(I, N=N)
        dH = H - Egs * id
        #
        stdE = {k: np.sqrt((mps.vdot(v, dH @ dH, v) - mps.vdot(v, dH, v) ** 2)).item().real for k, v in psi_t.items()}
        #
        def Momentum(N=N, a=a):
            terms =  [mps.Hterm( 1j / (4 * a), [n + 2, n], [cp, cm]) for n in range(N - 2)]
            terms += [mps.Hterm(-1j / (4 * a), [n, n + 2], [cp, cm]) for n in range(N - 2)]
            op = mps.generate_mpo(I, terms, N=N)
            return op

        Mom = Momentum()
        stdP = {k: np.sqrt(mps.vdot(v, Mom @ Mom, v)).item().real for k, v in psi_t.items()}
        #
        data = {'stdE': stdE, 'stdP': stdP, 'Egs': Egs, 'E_t': E_t, 'E_P': E_P}
        #
        fname = f"./results_fermions/Eng_gauss_{m=}_{N=}_{a=}.npy"
        np.save(fname, data, allow_pickle=True)
