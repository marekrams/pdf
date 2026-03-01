import numpy as np

import yastn
from yastn.tn import mps
from scipy.optimize import minimize

from .operators import Hamiltonian


def op1(P, N, a, ops=None):
    I, cp, cm, d = ops.I(), ops.cp(), ops.c(), ops.n()
    ks = np.arange(N)
    cs = -(1- (-1) ** ks) / 2
    if P==0:
        cfs = (N - ks) / 2
    else:
        cfs = (np.exp(1j*a*N*P) - np.exp(1j*a*ks*P))/(np.exp(1j*a*P) - 1)/2
    terms = [mps.Hterm(cf, [k], [d + cst * I]) for cf, k, cst in zip(cfs, ks, cs)]
    op = mps.generate_mpo(I, terms, N=N)
    return op


def op2(P, N, a, ops=None):
    I, cp, cm, d = ops.I(), ops.cp(), ops.c(), ops.n()
    terms =  [mps.Hterm( 1j * np.exp(1j*a*P*n)/ (2 * a), [n + 1, n], [cp, cm]) for n in range(N-1)]
    terms += [mps.Hterm( 1j * np.exp(1j*a*P*n)/ (2 * a), [n, n + 1], [cp, cm]) for n in range(N-1)]
    op = mps.generate_mpo(I, terms, N=N)
    return op



def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def simp(psi, D=128):
    psi.canonize_(to='last')
    err = psi.truncate_(to='first', opts_svd={'D_total': D, 'tol': 1e-6})


def construct_P_state(P, N, m, g, a, psi_gs, ops=None, Dmax=128):

    ap1 = op1(P, N, a, ops)
    ap2 = op2(P, N, a, ops)

    H = Hamiltonian(N, m, g, t=0, a=a, v=1, Q=0, ops=ops)

    N = psi_gs.N
    I, cp, cm, d = ops.I(), ops.cp(), ops.c(), ops.n()

    # ensure that ap1 will be orthogonal to the ground state

    c = mps.vdot(psi_gs, ap1, psi_gs)
    ap1 = ap1 - c * mps.generate_mpo(I, [], N=N)

    # act on ground state, optimize relative coefficient

    psi1 = ap1 @ psi_gs
    psi2 = ap2 @ psi_gs


    psi1.canonize_(to='last')
    psi1.truncate_(to='first', opts_svd={'D_total': Dmax, 'tol': 1e-6})

    psi2.canonize_(to='last')
    psi2.truncate_(to='first', opts_svd={'D_total': Dmax, 'tol': 1e-6})

    psi1 = psi1 / psi1.norm()
    psi2 = psi2 / psi2.norm()

    psis = [psi1, psi2]
    npsis = len(psis)

    nrms = np.zeros((npsis, npsis), dtype=np.complex128)
    for i in range(npsis):
        for j in range(npsis):
            nrms[i, j] = yastn.tn.mps.vdot(psis[i], psis[j])

    hams = np.zeros((npsis, npsis), dtype=np.complex128)
    for i in range(npsis):
        for j in range(npsis):
            hams[i, j] = mps.vdot(psis[i], H, psis[j])

    def fun(x):
        z = real_to_complex(x)
        z = np.concatenate(([1.0], z))
        vev = np.conj(z.T).dot(hams).dot(z)
        nrm = np.conj(z.T).dot(nrms).dot(z)
        # print(vev/nrm)
        return np.real(vev/nrm)

    x0 = np.zeros(npsis-1)   # assume that the first wavefunction (here the original Banuls state) has coefficient 1

    res = minimize(fun, complex_to_real(x0))

    def construct_psi(x):
        z = real_to_complex(x)
        z = np.concatenate(([1.0], z))
        nrm = np.conj(z.T).dot(nrms).dot(z)
        z = z/np.sqrt(nrm)
        psi = z[0]*psis[0]
        for i in range(1, npsis):
            psi += z[i]*psis[i]
        return psi

    psi_opt = construct_psi(res.x)
    simp(psi_opt)

    return psi_opt, fun(res.x).real