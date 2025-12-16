import argparse
import csv
import os.path
from pathlib import Path
import time

import numpy as np

import ray
import yastn
import yastn.tn.mps as mps
from scripts_fermions.operators import HNN, sum_Ln2, measure_local_observables


def folder_gs(g, m, a, N):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/gs/")
    path.mkdir(parents=True, exist_ok=True)
    return path

def folder_ex(g, m, a, N):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/ex/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def folder_evol(g, m, a, N, v, Q, D0, dt, D, tol, method, mkdir=True):
    path = Path(f"./results_fermions/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/{v=:0.4f}/{Q=:0.4f}/{D0=}/{dt=:0.4f}/{D=}/{tol=:0.0e}/{method}")
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


@ray.remote(num_cpus=1)
def run_gs(g, m, a, N, D0, energy_tol=1e-10, Schmidt_tol=1e-8):
    """ initial state at t=0 """
    #
    folder = folder_gs(g, m, a, N)
    fname = folder / f"state_D={D0}.npy"
    finfo = folder / "info.csv"
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    H0 = HNN(N, a, m, ops=ops)
    e0 = a * g * g / 2
    H1 = e0 * sum_Ln2(N, t=0, a=a, v=1, Q=1, ops=ops)
    #
    files = list(folder.glob("*.npy"))
    Ds = [int(f.stem.split("=")[1]) for f in files]
    if any(D <= D0 for D in Ds):
        D = max(D for D in Ds if D <= D0)
        print(f"Loading initial state with {D=}")
        old_data = np.load(folder / f"state_D={D}.npy", allow_pickle=True).item()
        psi_gs = yastn.from_dict(old_data["psi"])
    else:
        print(f"Random initial state.")
        psi_gs = mps.random_mps(H0, D_total=D0, n=(N // 2))
    # 2 sweeps of 2-site dmrg
    info = mps.dmrg_(psi_gs, [H0, H1], max_sweeps=200,
                     method='2site', opts_svd={"D_total": D0, "tol": 1e-6},
                     energy_tol=energy_tol, Schmidt_tol=Schmidt_tol, precompute=False)
    #
    data = {}
    data["psi"] = psi_gs.to_dict()
    data["bd"] = psi_gs.get_bond_dimensions()
    data["entropy"] = psi_gs.get_entropy()
    sch = psi_gs.get_Schmidt_values()
    data["schmidt"] = [x.data for x in sch]
    np.save(fname, data, allow_pickle=True)

    fieldnames = ["D", "energy", "sweeps", "denergy", "dSchmidt", "min_Schmidt"]
    out = {"D" : max(data["bd"]),
           "energy": info.energy,
           "sweeps": info.sweeps,
           "denergy": info.denergy,
           "dSchmidt": info.max_dSchmidt,
           "min_Schmidt": min(data["schmidt"][N // 2])}
    file_exists = os.path.isfile(finfo)
    with open(finfo, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)


@ray.remote(num_cpus=1)
def run_ex(g, m, a, N, D0, energy_tol=1e-10, Schmidt_tol=1e-8):
    """ initial state at t=0 """
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    try:
        fname = folder_gs(g, m, a, N) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi_gs = yastn.from_dict(data["psi"])
    except FileNotFoundError:
        return None

    folder = folder_ex(g, m, a, N)
    fname = folder / f"state_D={D0}.npy"
    finfo = folder / "info.csv"
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    H0 = HNN(N, a, m, ops=ops)
    e0 = a * g * g / 2
    H1 = e0 * sum_Ln2(N, t=0, a=a, v=1, Q=1, ops=ops)
    #
    files = list(folder.glob("*.npy"))
    Ds = [int(f.stem.split("=")[1]) for f in files]
    if any(D <= D0 for D in Ds):
        D = max(D for D in Ds if D <= D0)
        print(f"Loading initial state with {D=}")
        old_data = np.load(folder / f"state_D={D}.npy", allow_pickle=True).item()
        psi_ex = yastn.from_dict(old_data["psi"])
    else:
        print(f"Random initial state.")
        psi_ex = mps.random_mps(H0, D_total=D0, n=(N // 2))
    # 2 sweeps of 2-site dmrg
    info = mps.dmrg_(psi_ex, [H0, H1], max_sweeps=200, project=[psi_gs],
                     method='2site', opts_svd={"D_total": D0, "tol": 1e-6},
                     energy_tol=energy_tol, Schmidt_tol=Schmidt_tol, precompute=False)
    #
    data = {}
    data["psi"] = psi_ex.to_dict()
    data["bd"] = psi_ex.get_bond_dimensions()
    data["entropy"] = psi_ex.get_entropy()
    sch = psi_ex.get_Schmidt_values()
    data["schmidt"] = [x.data for x in sch]
    np.save(fname, data, allow_pickle=True)

    fieldnames = ["D", "energy", "sweeps", "denergy", "dSchmidt", "min_Schmidt"]
    out = {"D" : max(data["bd"]),
           "energy": info.energy,
           "sweeps": info.sweeps,
           "denergy": info.denergy,
           "dSchmidt": info.max_dSchmidt,
           "min_Schmidt": min(data["schmidt"][N // 2])}
    file_exists = os.path.isfile(finfo)
    with open(finfo, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)

def save_psi(fname, psi):
    data = {}
    data["psi"] = psi.to_dict()
    data["bd"] = psi.get_bond_dimensions()
    np.save(fname, data, allow_pickle=True)

@ray.remote(num_cpus=1)
def run_evol(g, m, a, N, D0, v, Q, dt, D, tol, method, snapshots, snapshots_states):
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    #
    try:
        fname = folder_gs(g, m, a, N) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi = yastn.from_dict(data["psi"])
    except FileNotFoundError:
        return None
    #
    e0 = a * g * g / 2
    folder = folder_evol(g, m, a, N, v, Q, D0, dt, D, tol, method)
    H0 = HNN(N, a, m, ops=ops)
    Ht = lambda t: [H0, e0 * sum_Ln2(N, t, a, v, Q, ops=ops)]

    times = np.linspace(0, N * a / (2 * v), snapshots + 1)
    sps = snapshots // snapshots_states

    data = {}
    data['entropy_1'] = np.zeros((snapshots + 1, N + 1), dtype=np.float64)
    data['entropy_2'] = np.zeros((snapshots + 1, N + 1), dtype=np.float64)
    data['entropy_3'] = np.zeros((snapshots + 1, N + 1), dtype=np.float64)
    data['Ln'] = np.zeros((snapshots + 1, N), dtype=np.float64)
    data['T00'] = np.zeros((snapshots + 1, N), dtype=np.float64)
    data['T11'] = np.zeros((snapshots + 1, N), dtype=np.float64)
    data['T01'] = np.zeros((snapshots + 1, N), dtype=np.float64)
    data['j0'] = np.zeros((snapshots + 1, N // 2), dtype=np.float64)
    data['j1'] = np.zeros((snapshots + 1, N // 2), dtype=np.float64)
    data['nu'] = np.zeros((snapshots + 1, N // 2), dtype=np.float64)
    data['energy'] = np.zeros(snapshots + 1, dtype=np.float64)
    data['time'] = np.zeros(snapshots + 1, dtype=np.float64) - 1  # times not calculated are < 0
    data['min_Schmidt'] = np.zeros(snapshots + 1, dtype=np.float64) - 1  # times not calculated are < 0

    evol = mps.tdvp_(psi, Ht, times,
                    method=method, dt=dt,
                    opts_svd={"D_total": D, "tol": tol},
                    yield_initial=True, precompute=False, subtract_E=True)

    tref0 = time.time()
    print(times)
    for ii, step in enumerate(evol):
        data['time'][ii] = step.tf
        data['entropy_1'][ii, :] = psi.get_entropy(alpha=1)
        data['entropy_2'][ii, :] = psi.get_entropy(alpha=2)
        data['entropy_3'][ii, :] = psi.get_entropy(alpha=3)
        data['energy'][ii] = mps.vdot(psi, Ht(step.tf), psi).real
        data["min_Schmidt"][ii] = min(psi.get_Schmidt_values()[N // 2].data)

        T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi, step.tf, a, g, m, v, Q, ops)
        data['T00'][ii, :] = T00
        data['T11'][ii, :] = T11
        data['T01'][ii, :] = T01
        data['j0'][ii, :] = j0
        data['j1'][ii, :] = j1
        data['nu'][ii, :] = nu
        data['Ln'][ii, :] = Ln
        print(f"t={step.tf:0.2f}  st={time.time() - tref0:0.1f} sek.")

        if ii % sps == 0:
            np.save(folder / f"results.npy", data, allow_pickle=True)
            save_psi(folder / f"state_t={step.tf:0.4f}.npy", psi)


if __name__ == "__main__":
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-g", type=float, default=1.0)
    # parser.add_argument("-D0", type=int, default=256)
    # parser.add_argument("-mm", type=float, default=1.0)
    # parser.add_argument("-N", type=int, default=1024)
    # parser.add_argument("-a", type=float, default=0.125)
    # parser.add_argument("-Q", type=float, default=1.0)

    # args = parser.parse_args()
    # print(args)

    # tref0 = time.time()
    # run_gs(args.g, args.mm, args.a, args.N, args.D0)
    # print(f"GS found in: {time.time() - tref0}")

    # v = 1
    # D, tol, method = args.D0, 1e-6, '12site'
    # snapshots = args.N // 2
    # dt = min(1 / 16, args.N * args.a / (2 * v * snapshots))
    # tref0 = time.time()
    # run_evol(args.g, args.mm, args.a, args.N, args.D0, v, args.Q, dt, D, tol, method, snapshots, 4)
    # print(f"Evolution finished in: {time.time() - tref0}")

    g=1
    ray.init()

    v = 1
    Q = 1
    dt = 1/16
    tol = 1e-6
    method = '12site'
    snapshots_states = 16
    refs = []
    for m in [0, 0.5]:
        for (N, a) in [(128, 0.25), (256, 0.125)]:
            for D0 in [64, 128]:
                snapshots = N // 2
                job = run_evol.remote(g, m, a, N, D0, v, Q, dt, D0, tol, method, snapshots, snapshots_states)
                # job = run_ex.remote(g, m, a, N, D0, energy_tol=1e-10, Schmidt_tol=1e-10)
                refs.append(job)
                # mlat = m - g * g * a / 8
                # # job = run_evol.remote(g, mlat, a, N, D0, v, Q, dt, D0, tol, method, snapshots, snapshots_states)
                # job = run_ex.remote(g, mlat, a, N, D0, energy_tol=1e-10, Schmidt_tol=1e-8)
                # refs.append(job)

    ray.get(refs)
