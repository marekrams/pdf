import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_heatmaps(ev, title, data, NaDdt, ms, g, f_analytic=None, subtract_t0=True, avarage_nn=False, tmax=None):
    nx, ny = len(NaDdt), len(ms)
    if f_analytic:
        nx += 1
    fig, ax = plt.subplots(nx, ny, figsize=(ny * 5, nx * 5), squeeze=False)  # sharex=True, sharey=True,

    zmin, zmax = 0, 0
    for m in ms:
        for j, (N, a, D, dt) in enumerate(NaDdt):
            try:
                ee = data[m, N, a, D, dt][ev][data[m, N, a, D, dt]["time"] > -1]
                ee = ee - ee[0, :]
                zmax = max(zmax, np.max(ee))
                zmin = min(zmin, np.min(ee))
            except KeyError:
                pass

    zlim = max(abs(zmin), abs(zmax))

    for i, m in enumerate(ms):
        for j, (N, a, D, dt) in enumerate(NaDdt):
            #try:
                tm = data[m, N, a, D, dt]["time"]
                mask = tm > -1
                tm = tm[mask]
                ee = data[m, N, a, D, dt][ev][mask]

                if subtract_t0:
                    ee = ee - ee[0, :]
                if avarage_nn:
                    ee = (ee[:, 0::2] + ee[:, 1::2]) / 2  # avarage over 2*n and 2*n+1

                xmax = N * a / 2
                im = ax[j, i].imshow(ee, extent=(-xmax, xmax, 0, tm[-1]),
                                    origin='lower', aspect='auto',
                                    vmin=-zlim, vmax=zlim,
                                    cmap = cm.seismic)
                ax[j, i].set_title(f"{a=:0.2f} {m/g=:0.2f} {N=} {D=}")
                ax[j, i].set_xticks([-xmax , -xmax / 2, 0, xmax / 2, xmax])
                ax[j, i].set_ylim([0, xmax])
                if tmax:
                    ax[j, i].set_ylim([0, tmax ])
                    ax[j, i].set_xlim([-tmax, tmax])
            # except KeyError:
            #     pass




    if f_analytic:
        N = 1024
        t = np.linspace(0, xmax, N).reshape(-1, 1)
        x = np.linspace(-xmax, xmax, 2 * N + 1).reshape(1, -1)
        analytical = f_analytic(g, t, x)

        j, i = nx - 1, 0
        ax[j, i].imshow(analytical, extent=(-xmax, xmax, 0, xmax),
                                    origin='lower', aspect='auto',
                                    vmin=-zlim, vmax=zlim,
                                    cmap = cm.seismic)
        ax[j, i].set_title(f"analytic solution, m=0")
        ax[j, i].set_xticks([-xmax , -xmax / 2, 0, xmax / 2, xmax])
        ax[j, i].set_ylim([0, xmax])

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.10, 0.02, 0.8])
    cb = fig.colorbar(im, cax=cbar_ax)
    # cb.ax.set_title(title)
    cb.ax.set_ylim([zmin, zmax])

    fig.text(0.43, -0.01, 'position  (n - n0) * a', ha='center')
    fig.text(-0.01, 0.5, 'time', va='center', rotation='vertical')
    fig.text(0.43, 1.00, title, ha='center')


def plot_comparison(ev, times, data, Nas, m, g, D, dt, f_analytic=None, subtract_t0=True, avarage_nn=False, ylim=None):
    nx, ny = 1, len(times)
    fig, ax = plt.subplots(nx, ny, sharex=True, sharey=True, figsize=(ny * 4, nx * 4))

    for j, t_target in enumerate(times):
        ax[j].set_title(f"{t_target=:0.1f}")
        ax[j].set_xlabel("position")

        ylabel = ev
        if subtract_t0:
            ylabel += " - " + ev + '(t=0)'
        ax[j].set_ylabel(ylabel)
        for N, a in Nas:
            tm = data[m, N, a, D, dt]["time"]
            ii = np.argmin(np.abs(tm - t_target))
            if abs(tm[ii] -  t_target) < 1e-6:
                ee0 = data[m, N, a, D, dt][ev][0]
                ee = data[m, N, a, D, dt][ev][ii]
                ns = a * (np.arange(len(ee0))) * (N / len(ee0))
                ns = ns - np.mean(ns)
                if subtract_t0:
                    ee = ee - ee0
                if avarage_nn:
                    ee = (ee[0::2] + ee[1::2]) / 2  # avarage over 2*n and 2*n+1
                    ns = (ns[0::2] + ns[1::2]) / 2  # avarage over 2*n and 2*n+1
                ax[j].plot(ns, ee, label=f"{N=} {a=:0.3f}")
            # np.savetxt(f"{ev:s}_{m=}_{N=}_{a=}_t={t_target}.txt", np.column_stack([ns, ee]))
        if f_analytic:
            xmax = np.max(ns)
            x = np.linspace(-xmax, xmax, 1024)
            t = t_target
            ax[j].plot(x, f_analytic(g, t, x), '--', label='analytic')
            # np.savetxt(f"{ev:s}_analytic_t={t_target}.txt", np.column_stack([x, f_analytic(g, t, x)]))
        ax[j].legend()
        if ylim:
            ax[j].set_ylim(ylim)
    fig.tight_layout()
    return fig, ax
