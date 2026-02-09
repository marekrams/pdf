import numpy as np
import yastn
import yastn.tn.mps as mps

def S(n):
    return f"S({n})"

def s2i(k):
    return int(k[2:-1])

def all_sites(N):
    return list(range(N))

def all_sites_lesser_equal(n):
    return list(range(n + 1))

def set_n0(N):
    return (N - 1) / 2

def Hamiltonian(N, m, g, t, a, v, Q, ops=None):
    H0 = HNN(N, a, m, ops=ops)
    H1 = (a * g * g / 2) * sum_Ln2(N, t=t, a=a, v=v, Q=Q, ops=ops)
    return H0 + H1

def Boost(N, m, g, a, ops=None):
    K0 = (-a) * nHNN(N, a, m, ops=ops)
    K1 = (-1 * a * a * g * g / 2) * sum_nLn2(N, ops=ops)
    return K0 + K1


def fermionP(N, a, P, x0, sg2, op, parity=-1, ops=None):
    #
    I = ops.I()
    if op == 'cp':
        op = ops.cp()
    elif op == 'cm':
        op = ops.c()
    else:
        raise ValueError()

    n0 = set_n0(N)
    if parity == -1:
        ns = np.arange(N)
    if parity == 0:
        ns = np.arange(0, N, 2)
    if parity == 1:
        ns = np.arange(1, N, 2)
    Hterms = []
    for n in ns:
        amp = np.exp(-((n-n0)*a-x0)**2/(2*sg2)) * np.exp(1j*P*(n-n0)*a)
        if abs(amp) > 1e-10:
            Hterms.append(mps.Hterm(amplitude=amp, positions=n, operators=op))
    return mps.generate_mpo(I, Hterms, N=N)


def HNN(N, a, m, ops=None):
    I, cp, cm, d = ops.I(), ops.cp(), ops.c(), ops.n()
    sites = all_sites(N)
    terms =  [mps.Hterm( 1j / (2 * a), [S(n + 1), S(n)], [cp, cm]) for n in sites[:-1]]
    terms += [mps.Hterm(-1j / (2 * a), [S(n), S(n + 1)], [cp, cm]) for n in sites[:-1]]
    terms += [mps.Hterm(m * (-1) ** n, [S(n)], [d]) for n in sites]
    terms = [mps.Hterm(v, tuple(s2i(k) for k in p), o) for v, p, o in terms]
    H = mps.generate_mpo(I, terms, N=N)
    return H

def nHNN(N, a, m, ops=None):
    I, cp, cm, d = ops.I(), ops.cp(), ops.c(), ops.n()
    sites = all_sites(N)
    n0 = set_n0(N)
    terms  = [mps.Hterm( 1j * (n + 1/2 - n0) / (2 * a), [S(n + 1), S(n)], [cp, cm]) for n in sites[:-1]]
    terms += [mps.Hterm(-1j * (n + 1/2 - n0) / (2 * a), [S(n), S(n + 1)], [cp, cm]) for n in sites[:-1]]
    terms += [mps.Hterm(m * (n - n0) * (-1) ** n, [S(n)], [d]) for n in sites]
    terms = [mps.Hterm(v, tuple(s2i(k) for k in p), o) for v, p, o in terms]
    H = mps.generate_mpo(I, terms, N=N)
    return H

def Qp(n, t, n0, a, v, Q):
    return Q * np.maximum(1 - np.abs(n0 + v * t / a - n), 0)

def Qm(n, t, n0, a, v, Q):
    return Q * np.maximum(1 - np.abs(n0 - v * t / a - n), 0)

def dLn(n, t, n0, a, v, Q):
    return Qp(n, t, n0, a, v, Q) - Qm(n, t, n0, a, v, Q) - (1 - (-1) ** n) / 2

def cLn(n, t=None, n0=None, a=None, v=None, Q=None):
    ns = np.array(all_sites_lesser_equal(n), dtype=np.float64)
    cc = - np.sum((1 - (-1) ** ns) / 2)
    if t is not None:
        cc += np.sum(Qp(ns, t, n0, a, v, Q) - Qm(ns, t, n0, a, v, Q))
    return cc

def cLns(N, t, n0, a, v, Q):
    sites = all_sites(N)
    ns = np.array(sites, dtype=np.float64)
    tmp = np.cumsum(Qp(ns, t, n0, a, v, Q) - Qm(ns, t, n0, a, v, Q) - (1 - (-1) ** ns) / 2)
    return dict(zip(sites, tmp))


def t_to_L(tt, n, N, time=None, a=None, v=None, Q=None):
    if isinstance(tt, (tuple)):
        tt = tt[0]
    return N // 2 - tt + cLn(n, time, set_n0(N), a, v, Q)


def L_to_t(L, n, N, time=None, a=None, v=None, Q=None):
    return N // 2 - L + cLn(n, time, set_n0(N), a, v, Q)


def project_Ln(psi, L, n, time=None, a=None, v=None, Q=None):
    N = psi.N
    tt = L_to_t(L, n, N, time, a, v, Q)
    tt = (int(np.round(tt)),)

    leg = psi[n].get_legs(axes=2)
    D = leg.tD[tt]

    leg0 = yastn.Leg(psi.config, s=leg.s, t=[tt], D=[D])
    proj = yastn.eye(psi.config, legs=leg0)

    psi = psi.shallow_copy()
    psi[n] = yastn.apply_mask(proj, psi[n], axes=2)
    psi[n + 1] = yastn.apply_mask(proj, psi[n + 1], axes=0)
    psi.canonize_(to='first', normalize=False)
    return psi


def Ln(n, N, t, a, v, Q, ops=None):
    I, d = ops.I(), ops.n()
    sites = all_sites_lesser_equal(n)
    n0 = set_n0(N)
    dLns = dLn(np.array(sites, dtype=np.float64), t, n0, a, v, Q)
    terms = [mps.Hterm(1, [S(k)], [d + cst * I]) for k, cst in zip(sites, dLns)]
    terms = [mps.Hterm(v, tuple(s2i(k) for k in p), o) for v, p, o in terms]
    return mps.generate_mpo(I, terms, N=N)

def sum_Ln2(N, t, a, v, Q, ops=None):
    """ sum_{n=0}^{N-2} Ln^2 """
    #
    I, d = ops.I(), ops.n()
    sites = all_sites(N)
    nlast = sites[-1]
    n0 = set_n0(N)
    dLns = dLn(np.array(sites, dtype=np.float64), t, n0, a, v, Q)
    ds  = [d + cst * I for cst in dLns]
    d2s = [x @ x for x in ds]
    ds  = [x.add_leg(axis=0, s=-1).add_leg(axis=2, s=1) for x in ds]
    d2s = [x.add_leg(axis=0, s=-1).add_leg(axis=2, s=1) for x in d2s]
    I = I.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    H = mps.Mpo(N)
    # This encodes Hamiltonian of the form sum_{n<n'} A_n B_n' + sum_n C_n
    for i, n in enumerate(sites):
        An = ds[i]
        xx = (nlast + 1 - n)
        Bn = 2 * xx * ds[i]
        Cn = xx * d2s[i]

        if i == 0:
            H[i] = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn}, common_legs=(1, 3))
        elif i == N - 1:
            H[i] = yastn.block({(0, 2): Cn,
                                (1, 2): Bn,
                                (2, 2): I}, common_legs=(1, 3))
        else:
            H[i] = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn,
                                           (1, 1): I,  (1, 2): Bn,
                                                       (2, 2): I}, common_legs=(1, 3))
    return H

def sum_nLn2(N, ops=None):
    """ sum_{n=1}^{N} n Ln^2 """
    t, a, v, Q = 0, 1, 0, 0
    I, d = ops.I(), ops.n()
    sites = all_sites(N)
    nlast = sites[-1]
    n0 = set_n0(N)
    dLns = dLn(np.array(sites, dtype=np.float64), t, n0, a, v, Q)
    ds  = [d + cst * I for cst in dLns]
    d2s = [x @ x for x in ds]
    ds  = [x.add_leg(axis=0, s=-1).add_leg(axis=2, s=1) for x in ds]
    d2s = [x.add_leg(axis=0, s=-1).add_leg(axis=2, s=1) for x in d2s]
    I = I.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    H = mps.Mpo(N)
    # This encodes Hamiltonian of the form sum_{n<n'} A_n B_n' + sum_n C_n
    for i, n in enumerate(all_sites(N)):
        An = ds[i]
        xx = (nlast + 1 - n) * (nlast + 1 + n - 2 * n0) / 2
        Bn = 2 * xx * ds[i]
        Cn = xx * d2s[i]
        #
        if i == 0:
            H[i] = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn}, common_legs=(1, 3))
        elif i == N - 1:
            H[i] = yastn.block({(0, 2): Cn,
                                (1, 2): Bn,
                                (2, 2): I}, common_legs=(1, 3))
        else:
            H[i] = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn,
                                           (1, 1): I,  (1, 2): Bn,
                                                       (2, 2): I}, common_legs=(1, 3))
    return H

def measure_local_observables(psi, t, a, g, m, v, Q, ops):
    #
    psi = psi / psi.norm()
    #
    N = psi.N
    I, cp, cm, d = ops.I(), ops.cp(), ops.c(), ops.n()
    n0 = set_n0(N)
    #
    sites = all_sites(N)
    dLns = dLn(np.array(sites, dtype=np.float64), t, n0, a, v, Q)
    #
    ecpcm = mps.measure_2site(psi, cp, cm, psi, bonds='r1r-1r2r-2')
    ed = {k: v.real for k, v in mps.measure_1site(psi, d, psi).items()}
    #
    #
    Hdd = mps.product_mpo(I, N=N)
    env = mps.Env(psi, [Hdd, psi])
    env.setup_(to='first')
    #
    eLn2 = {-1: 0.,
             0: mps.measure_1site(psi, (d + dLns[0] * I) @ (d + dLns[0] * I), psi, sites=0).real}
    #
    L, C, R = dd_mpo_elements(I, d + dLns[0] * I)
    Hdd[0] = L
    for i in range(1, N):
        L, C, R = dd_mpo_elements(I, d + dLns[i] * I)
        env.update_env_(i-1, to='last')
        Hdd[i] = R
        env.update_env_(i, to='first')
        eLn2[i] = env.measure(bd=(i-1, i)).real
        Hdd[i] = C
    #
    T00 = np.zeros(N, dtype=np.float64)
    T11 = np.zeros(N, dtype=np.float64)
    for i in range(N):
        tmp = (-1j / (4 * a * a)) * (ecpcm.get((i-1, i), 0) + ecpcm.get((i, i+1), 0))
        tmp += (1j / (4 * a * a)) * (ecpcm.get((i, i-1), 0) + ecpcm.get((i+1, i), 0))
        T00[i] += tmp.real
        T11[i] += tmp.real

        T00[i] += (m / a) * ((-1) ** i) * ed[i]
        T00[i] += (g * g / 4) * (eLn2[i-1] + eLn2[i])
        T11[i] -= (g * g / 4) * (eLn2[i-1] + eLn2[i])

    T01 = np.zeros(N, dtype=np.float64)
    # in the old, there was zeros(N-2), without the first and the last sites.
    for i in range(N):
        T01[i] = (1j / (4 * a * a) * (ecpcm.get((i+1, i-1), 0) - ecpcm.get((i-1, i+1), 0))).real

    j0 = np.zeros(N // 2, dtype=np.float64)
    j1 = np.zeros(N // 2, dtype=np.float64)
    nu = np.zeros(N // 2, dtype=np.float64)
    for i in range(N // 2):
        j0[i] = (ed[2 * i] + ed[2 * i + 1]) / (2 * a)
        j1[i] = (ecpcm[2 * i, 2 * i + 1] + ecpcm[2 * i + 1, 2 * i]).real / (2 * a)
        nu[i] = (ed[2 * i] - ed[2 * i + 1]) / (2 * a)

    Ln = np.zeros(N, dtype=np.float64)
    Ln[0] = ed[0] + dLns[0]
    for i in range(1, N):
        Ln[i] = Ln[i-1] + ed[i] + dLns[i]

    return T00, T11, T01, j0, j1, nu, Ln

def dd_mpo_elements(I, d):
    I = I.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    # This encodes Hamiltonian of the form sum_{n<n'} A_n B_n' + sum_n C_n
    #
    An = d
    Bn = 2 * d
    Cn = d @ d
    #
    An = An.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    Bn = Bn.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    Cn = Cn.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    L = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn}, common_legs=(1, 3))
    R = yastn.block({(0, 2): Cn,
                     (1, 2): Bn,
                     (2, 2): I}, common_legs=(1, 3))
    C = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn,
                                (1, 1): I,  (1, 2): Bn,
                                            (2, 2): I}, common_legs=(1, 3))
    return L, C, R