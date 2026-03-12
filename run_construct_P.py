import numpy as np
from scripts_fermions.construct_P_state import construct_P_state

import glob
import os

import yastn
from yastn.tn import mps


Nas = [(512, 0.0625)] #, (512, 0.0625), (512, 0.125)]
ms = [0, 0.5, 0.6]
g = 1
sg2 = 0.25
x0 = 1.5
#
PPs = np.linspace(0, 6, 49)
m = 0.5


#


for N, a, in Nas:

    D = 128
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    #
    glob_path = os.path.join(os.path.abspath(""), "./results_fermions/g=1.0000/**/gs/**/state_D=*.npy")
    fnames = glob.glob(glob_path, recursive=True)
    fgs = sorted([fname for fname in fnames if all(x in fname for x in ["/g=1.0000/", f"{N=}", f"{a=:0.4f}", f"{D=}", f"{m=:0.4f}"])])
    #
    def fn_mass(fns, **kwargs):
        return [x for x in fns if all( f"{k}={v}" in x for k, v in kwargs.items())][0]
    #
    d_gs = np.load(fn_mass(fgs, m=m), allow_pickle=True).item()
    psi_gs = yastn.from_dict(d_gs['psi'])


    psi_P = {}
    eng_P = {}
    for PP in PPs:
        print(PP)
        psi_P[PP], eng_P[PP] = construct_P_state(PP, N, m, g, a, psi_gs, ops)
        psi_P[PP] = psi_P[PP].to_dict()

    data = {"state": psi_P, 'energy': eng_P}
    np.save(f"./results_fermions/construct_P_{m=}_{N=}_{a=}.npy", data, allow_pickle=True)