import numpy as np
from operators import dLn, cLn, cLns, all_sites
from operators import measure_local_observables, sum_Ln2, Ln, sum_nLn2, set_n0
import yastn
import yastn.tn.mps as mps


def test_cLn():
    for t, a, v, Q, N in [[3, 0.2, 0.75, 1, 40],
                          [0, 1, 1, 0, 40],
                          [5, 0.1, 0.2, 10, 32]]:

        n0 = set_n0(N)
        assert sum(range(N)) / N == n0

        for n in [0, 1, 4, 7, 20]:
            assert cLn(n, t, n0, a, v, Q) == sum(dLn(k, t, n0, a, v, Q) for k in range(0, n+1))

        sites = all_sites(N)
        aa = {n: cLn(n, t, n0, a, v, Q) for n in sites}
        bb = cLns(N, t, n0, a, v, Q)
        assert len(aa) == len(bb)
        for k in aa:
            assert np.allclose(aa[k], bb[k])


def test_sum_Ln2(N):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    for t, a, v, Q in [(2, 1, 1, 1), (4.5, 0.25, 0.3, 2.2)]:
        H1 = sum_Ln2(N, t=t, a=a, v=v, Q=Q, ops=ops)

        sites = all_sites(N)
        Lns = {n: Ln(n, N, t=t, a=a, v=v, Q=Q, ops=ops) for n in sites}

        LL = Lns[0] @ Lns[0]
        for n in sites[1:]:
            LL += Lns[n] @ Lns[n]

        print((H1 - LL).norm() / H1.norm())
        assert (H1 - LL).norm() < 1e-13 * H1.norm()



def test_sum_nLn2(N):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    t, a, v, Q = 0, 1, 0, 0

    H1 = sum_nLn2(N, ops=ops)

    sites = all_sites(N)
    Lns = {n: Ln(n, N, t=t, a=a, v=v, Q=Q, ops=ops) for n in sites}
    n0 = set_n0(N)

    LL = (0 + 1/2 - n0) * Lns[0] @ Lns[0]
    for n in sites[1:]:
        LL += (n + 1/2 - n0) * Lns[n] @ Lns[n]

    print((H1 - LL).norm() / H1.norm())
    assert (H1 - LL).norm() < 1e-13 * H1.norm()




def test_measure_local_observables():
    t, a, g, m, v, Q, N = 3, 0.2, 0.2, 0.1, 0.75, 1, 40
    ops = yastn.operators.SpinlessFermions(sym='U1')
    I = ops.I()
    HI = mps.product_mpo(I, N=N)
    psi = mps.random_mps(HI, n=N//2, D_total=64, dtype='complex128')
    psi.canonize_(to='first').canonize_(to='last')

    oT00, oT11, oT01, oj0, oj1, onu, oLn = measure_local_observables_old(psi, t, a, g, m, v, Q, ops)
    T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi, t, a, g, m, v, Q, ops)

    assert np.allclose(oT00, T00)
    assert np.allclose(oT11, T11)
    assert np.allclose(oT01, T01)
    assert np.allclose(oj0, j0)
    assert np.allclose(oj1, j1)
    assert np.allclose(onu, nu)
    assert np.allclose(oLn, Ln)






if __name__ == "__main__":
    test_cLn()
    test_sum_Ln2(N=10)
    test_sum_Ln2(N=16)
    test_sum_nLn2(N=10)
    test_sum_nLn2(N=16)

    # test_measure_local_observables()
