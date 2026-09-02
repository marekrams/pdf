import argparse
import csv
import os.path
from pathlib import Path
import time

import numpy as np

import ray
import yastn
import yastn.tn.mps as mps
from scripts_fermions.operators import HNN, sum_Ln2, measure_local_observables, Hamiltonian, Boost, fermionP, project_Ln


def core_folder(mlat):
    if mlat:
        return "results_mlat"
    return "results_fermions"

def folder_gs(g, m, a, N, mlat):
    ss = core_folder(mlat)
    path = Path(f"./{ss}/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/gs/")
    path.mkdir(parents=True, exist_ok=True)
    return path

def folder_ex(g, m, a, N, mlat):
    ss = core_folder(mlat)
    path = Path(f"./{ss}/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/ex/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def folder_exx(g, m, a, N, mlat):
    ss = core_folder(mlat)
    path = Path(f"./{ss}/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/exx/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def folder_evol(g, m, a, N, v, Q, D0, dt, D, tol, method, mlat, mkdir=True):
    ss = core_folder(mlat)
    path = Path(f"./{ss}/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/{v=:0.4f}/{Q=:0.4f}/{D0=}/{dt=:0.4f}/{D=}/{tol=:0.0e}/{method}")
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def folder_gauss(g, m, a, N, P, x0, sg2, D0, dt, D, tol, method, mlat, mkdir=True):
    ss = core_folder(mlat)
    path = Path(f"./{ss}/{g=:0.4f}/{m=:0.4f}/{N=}/{a=:0.4f}/{P=:0.4f}/{x0=:0.4f}/{sg2=:0.4f}/{D0=}/{dt=:0.4f}/{D=}/{tol=:0.0e}/{method}")
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


@ray.remote(num_cpus=1)
def run_gs(g, m, a, N, D0, mlat, energy_tol=1e-10, Schmidt_tol=1e-8):
    """ initial state at t=0 """
    #
    folder = folder_gs(g, m, a, N, mlat)
    fname = folder / f"state_D={D0}.npy"
    finfo = folder / "info.csv"
    #
    if mlat:
        m = m - g * g * a / 8
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

    T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi_gs, 0, a, g, m, v=1, Q=0, ops=ops)
    data['T00'] = T00
    data['T11'] = T11
    data['T01'] = T01
    data['j0'] = j0
    data['j1'] = j1
    data['nu'] = nu
    data['Ln'] = Ln

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
def run_ex(g, m, a, N, D0, mlat, energy_tol=1e-10, Schmidt_tol=1e-8):
    """ initial state at t=0 """
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    try:
        fname = folder_gs(g, m, a, N, mlat) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi_gs = yastn.from_dict(data["psi"])
    except FileNotFoundError:
        return None

    folder = folder_ex(g, m, a, N, mlat)
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



@ray.remote(num_cpus=1)
def run_exx(g, m, a, N, D0, mlat, energy_tol=1e-10, Schmidt_tol=1e-8):
    """ initial state at t=0 """
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    try:
        fname = folder_gs(g, m, a, N, mlat) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi_gs = yastn.from_dict(data["psi"])
    except FileNotFoundError:
        return None

    try:
        fname = folder_ex(g, m, a, N, mlat) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi_ex = yastn.from_dict(data["psi"])
    except FileNotFoundError:
        return None


    folder = folder_exx(g, m, a, N, mlat)
    fname = folder / f"state_D={D0}.npy"
    finfo = folder / "info.csv"
    #
    if mlat:
        m = m - g * g * a / 8
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
        psi_exx = yastn.from_dict(old_data["psi"])
    else:
        print(f"Random initial state.")
        psi_exx = mps.random_mps(H0, D_total=D0, n=(N // 2))
    # 2 sweeps of 2-site dmrg
    info = mps.dmrg_(psi_exx, [H0, H1], max_sweeps=200, project=[psi_gs, psi_ex],
                     method='2site', opts_svd={"D_total": D0, "tol": 1e-6},
                     energy_tol=energy_tol, Schmidt_tol=Schmidt_tol, precompute=False)
    #
    data = {}
    data["psi"] = psi_exx.to_dict()
    data["bd"] = psi_exx.get_bond_dimensions()
    data["entropy"] = psi_exx.get_entropy()
    sch = psi_exx.get_Schmidt_values()
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


def get_psiLn_pair(psi, n0):
    # returns components of psi with (L(n0-1)+L(n0))/2 = -2, -3/2, -1, -1/2, 0, 1/2, 1, 3/2, 2
    assert n0 % 2 == 0

    psi3 = project_Ln(psi, 3, n0-1)
    psi2 = project_Ln(psi, 2, n0-1)
    psi1 = project_Ln(psi, 1, n0-1)
    psi0 = project_Ln(psi, 0, n0-1)
    psim1 = project_Ln(psi, -1, n0-1)
    psim2 = project_Ln(psi, -2, n0-1)
    psim3 = project_Ln(psi, -3, n0-1)

    assert project_Ln(psi2, 1, n0).norm() < 1e-10
    assert project_Ln(psi1, 0, n0).norm() < 1e-10
    assert project_Ln(psi0, -1, n0).norm() < 1e-10
    assert project_Ln(psim1, -2, n0).norm() < 1e-10

    psip6 = project_Ln(psi3, 3, n0)
    psip5 = project_Ln(psi2, 3, n0)
    psip4 = project_Ln(psi2, 2, n0)
    psip3 = project_Ln(psi1, 2, n0)
    psip2 = project_Ln(psi1, 1, n0)
    psip1 = project_Ln(psi0, 1, n0)
    psip0 = project_Ln(psi0, 0, n0)
    psipm1 = project_Ln(psim1, 0, n0)
    psipm2 = project_Ln(psim1, -1, n0)
    psipm3 = project_Ln(psim2, -1, n0)
    psipm4 = project_Ln(psim2, -2, n0)
    psipm5 = project_Ln(psim3, -2, n0)
    psipm6 = project_Ln(psim3, -2, n0)

    psis2 = [psipm6, psipm5, psipm4, psipm3, psipm2, psipm1, psip0, psip1, psip2, psip3, psip4, psip5, psip6]
    pr2 = np.array([x.norm() ** 2 for x in psis2], dtype=np.float64)

    return pr2


def get_psiLn(psi, n0):

    psi3 = project_Ln(psi, 3, n0-1)
    psi2 = project_Ln(psi, 2, n0-1)
    psi1 = project_Ln(psi, 1, n0-1)
    psi0 = project_Ln(psi, 0, n0-1)
    psim1 = project_Ln(psi, -1, n0-1)
    psim2 = project_Ln(psi, -2, n0-1)
    psim3 = project_Ln(psi, -3, n0-1)

    psis1 = [psim3, psim2, psim1, psi0, psi1, psi2, psi3]
    pr1 = np.array([x.norm() ** 2 for x in psis1], dtype=np.float64)
    en1 = np.array([x.get_entropy()[n0] for x in psis1], dtype=np.float64)
    return pr1, en1


@ray.remote(num_cpus=8)
def run_evol(g, m, a, N, D0, v, Q, dt, D, tol, method, mlat, snapshots, snapshots_states):
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    #
    try:
        fname = folder_gs(g, m, a, N, mlat) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi = yastn.from_dict(data["psi"])
    except FileNotFoundError:
        return None
    #
    folder = folder_evol(g, m, a, N, v, Q, D0, dt, D, tol, method, mlat)
    #
    if mlat:
        m = m - g * g * a / 8
    #
    e0 = a * g * g / 2
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
    data['evol_time'] = np.zeros(snapshots + 1, dtype=np.float64)  # times not calculated are < 0
    data['time'] = np.zeros(snapshots + 1, dtype=np.float64) - 1  # times not calculated are < 0

    data['min_Schmidt'] = np.zeros(snapshots + 1, dtype=np.float64) - 1  # times not calculated are < 0

    data['pr1a'] = np.zeros((snapshots + 1, 7), dtype=np.float64)
    data['pr1b'] = np.zeros((snapshots + 1, 7), dtype=np.float64)
    data['pr1c'] = np.zeros((snapshots + 1, 7), dtype=np.float64)
    data['ent1a'] = np.zeros((snapshots + 1, 7), dtype=np.float64)
    data['ent1b'] = np.zeros((snapshots + 1, 7), dtype=np.float64)
    data['ent1c'] = np.zeros((snapshots + 1, 7), dtype=np.float64)

    data['pr2'] = np.zeros((snapshots + 1, 13), dtype=np.float64)

  # times not calculated are < 0

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

        n0 = N // 2
        data['pr2'][ii, :] = get_psiLn_pair(psi, n0)
        data['pr1a'][ii, :], data['ent1a'][ii, :] = get_psiLn(psi, n0 - 1)
        data['pr1b'][ii, :], data['ent1b'][ii, :] = get_psiLn(psi, n0)
        data['pr1c'][ii, :], data['ent1c'][ii, :] = get_psiLn(psi, n0 + 1)

        data['evol_time'][ii] = time.time() - tref0

        print(f"t={step.tf:0.2f}  st={data['evol_time'][ii]:0.1f} sek.")

        if ii % sps == 0:
            np.save(folder / f"results.npy", data, allow_pickle=True)
            # save_psi(folder / f"state_t={step.tf:0.4f}.npy", psi)


@ray.remote(num_cpus=2)
def run_gauss(g, m, a, N, P, x0, sg2, D0, dt, D, tol, method, snapshots, snapshots_states):
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    #
    try:
        fname = folder_gs(g, m, a, N) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi = yastn.from_dict(data["psi"])
        print(f" Loaded {fname=}")
    except FileNotFoundError:
        print(f" Not found {fname=}")
        return None
    #
    e0 = a * g * g / 2
    folder = folder_gauss(g, m, a, N, P, x0, sg2, D0, dt, D, tol, method)
    H = [HNN(N, a, m, ops=ops), e0 * sum_Ln2(N, 0, a, v=1, Q=0, ops=ops)]
    #
    fermionR = fermionP(N, a, P, x0, sg2, 'cp', parity=0, ops=ops)
    fermionL = fermionP(N, a, -P, -x0, sg2, 'cm', parity=1, ops=ops)
    #
    psi = fermionL @ fermionR @ psi
    psi = psi / psi.norm()
    psi.canonize_(to='last')
    psi.truncate_(to='first', opts_svd={'tol': tol, 'D_total': D})
    #
    times = np.linspace(0, N * a / 2, snapshots + 1)
    sps = snapshots // snapshots_states
    #
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

    evol = mps.tdvp_(psi, H, times,
                    method=method, dt=dt,
                    opts_svd={"D_total": D, "tol": tol},
                    yield_initial=True, precompute=False, subtract_E=True)

    tref0 = time.time()
    for ii, step in enumerate(evol):
        data['time'][ii] = step.tf
        data['entropy_1'][ii, :] = psi.get_entropy(alpha=1)
        data['entropy_2'][ii, :] = psi.get_entropy(alpha=2)
        data['entropy_3'][ii, :] = psi.get_entropy(alpha=3)
        data['energy'][ii] = mps.vdot(psi, H, psi).real
        data["min_Schmidt"][ii] = min(psi.get_Schmidt_values()[N // 2].data)

        T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi, step.tf, a, g, m, v=1, Q=0, ops=ops)
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





@ray.remote(num_cpus=4)
def run_boost(g, m, a, N, D0, D, tol):
    ops = yastn.operators.SpinlessFermions(sym='U1', tensordot_policy='no_fusion')
    #
    try:
        fname = folder_ex(g, m, a, N) / f"state_D={D0}.npy"
        data = np.load(fname, allow_pickle=True).item()
        psi = yastn.from_dict(data["psi"])
        print(f" Loaded {fname=}")
    except FileNotFoundError:
        print(f" Not found {fname=}")
        return None
    #
    t = 0
    v = 1
    Q = 0
    H = Hamiltonian(N, m, g, t, a, v, Q, ops)
    K = Boost(N, m, g, a, ops)

    chis = np.linspace(0, 3, 61)


    snapshots = len(chis) - 1
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

    opts_expmv = {'hermitian': True, 'tol': 1e-10}
    opts_svd={"D_total": D, "tol": tol},

    evol = mps.tdvp_(psi, K, chis,
                     method='12site', opts_svd=opts_svd, dt=0.05,
                     yield_initial=True, precompute=False, subtract_E=True,
                     opts_expmv=opts_expmv)

    data['chis'] = chis
    data['psi'] = {}
    tref0 = time.time()
    for ii, step in enumerate(evol):
        chi = step.tf
        assert abs(chi - chis[ii]) < 1e-6, 'rapidity mismatch'
        data['time'][ii] = step.tf
        data['entropy_1'][ii, :] = psi.get_entropy(alpha=1)
        data['entropy_2'][ii, :] = psi.get_entropy(alpha=2)
        data['entropy_3'][ii, :] = psi.get_entropy(alpha=3)
        data['energy'][ii] = mps.vdot(psi, H, psi).real
        data["min_Schmidt"][ii] = min(psi.get_Schmidt_values()[N // 2].data)

        T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi, step.tf, a, g, m, v, Q, ops)
        data['T00'][ii, :] = T00
        data['T11'][ii, :] = T11
        data['T01'][ii, :] = T01
        data['j0'][ii, :] = j0
        data['j1'][ii, :] = j1
        data['nu'][ii, :] = nu
        data['Ln'][ii, :] = Ln

        data['psi'][ii] = psi.to_dict()

        print(f"chi={step.tf:0.2f}  st={time.time() - tref0:0.1f} sek.")


    fname = folder_ex(g, m, a, N) / f"state_D={D0}_boosted.npy"
    np.save(fname, data, allow_pickle=True)


if __name__ == "__main__":
    ray.init()
    g = 1
    v = 1
    Q = 1
    dt = 1 / 16
    tol = 1e-6
    method = '12site'
    snapshots_states = 16
    mlat = True
    refs = []
    for m in [0.0, 0.1, 0.2, 0.3183, 0.4, 0.5, 0.6, 0.7]:
        for (N, a) in [(1024, 1/16)]:
            for D0 in [256]:
                snapshots = N // 2
                # job = run_gs.remote(g, m, a, N, D0, mlat, energy_tol=1e-10, Schmidt_tol=1e-10)
                # job = run_ex.remote(g, mlat, a, N, D0, energy_tol=1e-10, Schmidt_tol=1e-8)
                job = run_evol.remote(g, m, a, N, D0, v, Q, dt, D0, tol, method, mlat, snapshots, snapshots_states)
                refs.append(job)
    ray.get(refs)





# if __name__ == "__main__":
#     #
#     g = 1
#     ray.init()
#     dt = 1 / 16
#     tol = 1e-6
#     method = '12site'
#     snapshots_states = 8
#     refs = []

#     x0 = 3.0
#     sg2 = 0.25

#     for m in [0.0, 0.5]:
#         for P in [0, 1, 2, 3, 4]:
#             for (N, a) in [(256, 1/8)]:  # (256, 1/8)
#                 D0 = 128
#                 for D in [512]:
#                     snapshots = N // 2
#                     job = run_gauss.remote(g, m, a, N, P, x0, sg2, D0, dt, D, tol, method, snapshots, snapshots_states)
#                     refs.append(job)
#     ray.get(refs)
