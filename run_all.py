# Runs all cases of the assignment with devsim_pn.py and saves publication-style PNG plots + CSV files.
# Usage:  python run_all.py               (full run, white background  -> use this for the report)
#         python run_all.py dark          (full run, dark background, files get a _dark suffix)
#         python run_all.py quick         (2 cases only, to test that everything works)
#         python run_all.py quick dark
#         python run_all.py serial        (no parallel processes - use this if the parallel run gives problems)
# NOTE: the 1-D device has an area of 1 cm2, so I [A] = J [A/cm2] numerically.
#
# SPEED: all simulations are independent, so they run in parallel (one process per CPU core) BEFORE any plotting.
#        devsim_pn.py also uses a coarser far-field mesh (PS_FAR), prints nothing and uses a single thread per process.
import os, sys, time
import numpy as np
import devsim_pn as P

QUICK = 'quick' in sys.argv[1:]
SERIAL = 'serial' in sys.argv[1:]
DARK = False                      # True = dark background, False = white background
DARK = DARK or ('dark' in sys.argv[1:])
SIGN = 1.0                        # set to -1.0 if forward-bias current comes out NEGATIVE
Vf = np.arange(0, 1.0001, 0.025)  # forward voltage points for the dark curves
Vr = np.array([0, -0.025, -0.05, -0.1, -0.2, -0.4, -0.6, -0.8, -1.0, -1.25, -1.5, -1.75, -2.0])   # reverse points
Vs = np.arange(0, 0.7001, 0.01)   # voltage points for the solar cell
WS = (2, 10, 50, 200)             # thicknesses [um]
SUNS = (0.25, 0.5, 1.0)

def exp10(N): return int(round(np.log10(N)))

# ------------------------------------------------------------------ run all simulations (parallel)
def run_jobs(specs):
    n = 1 if SERIAL else min(len(specs), os.cpu_count() or 1)
    out, errors, t0 = {}, [], time.time()
    print(f"Running {len(specs)} simulations on {n} process(es) ...", flush=True)
    def done(tag):
        name = f"doping N = {tag[1]:.0e}" if tag[0] == "dark" else (f"{tag[1]} thickness {tag[2]} um" if tag[0] == "thick"
               else f"solar cell, {tag[1]*1000:.0f} W/m2")
        print(f"   [{len(out):2d}/{len(specs)}] done {name}   ({time.time()-t0:.1f} s)", flush=True)
    if n <= 1:
        for s in specs:
            with P.quiet():
                tag, res = P.job(s)
            if isinstance(res, str): errors.append((tag, res)); print(f"   FAILED {tag}", flush=True); continue
            out[tag] = res; done(tag)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n, initializer=P.silence) as pool:
            for tag, res in pool.imap_unordered(P.job, specs):
                if isinstance(res, str): errors.append((tag, res)); print(f"   FAILED {tag}", flush=True); continue
                out[tag] = res; done(tag)
    if errors:                                              # show the real DEVSIM error text, then stop
        for tag, msg in errors[:2]:
            print(f"\n----- ERROR in simulation {tag} -----\n{msg}")
        print(f"{len(errors)} of {len(specs)} simulations failed. Please send the error text above.")
        sys.exit(1)
    return out

def make_specs():
    Ns = (1e17,) if QUICK else (1e15, 1e16, 1e17, 1e18)
    specs = []
    for sun in ((1.0,) if QUICK else (1.0, 0.5, 0.25, 0.0)):                     # heaviest jobs first
        specs.append((("sun", sun), "run_solar", dict(sun=sun, volts=(np.arange(0, 0.7001, 0.02) if QUICK else Vs))))
    if not QUICK:
        for w in (200, 50, 10, 2):
            for label in ("Wp", "Wn"):
                Wn, Wp = (50e-4, w*1e-4) if label == "Wp" else (w*1e-4, 50e-4)
                specs.append((("thick", label, w), "run_dark", dict(Nd=1e17, Na=1e17, Wn=Wn, Wp=Wp, volts=Vf, volts_rev=Vr)))
    for N in Ns:
        specs.append((("dark", N), "run_dark", dict(Nd=N, Na=N, Wn=50e-4, Wp=50e-4, volts=Vf, volts_rev=Vr, want_bands=True)))
        heavy = lambda s: 0 if (s[0][0] == "thick" and s[0][2] == 200) else 1     # the two 200 um jobs are the slowest: start them first
    specs.sort(key=heavy)                                                      # (stable sort: all other jobs keep their order)
    return Ns, specs

def main():
    t_start = time.time()
    Ns, specs = make_specs()
    R = run_jobs(specs)
    t_sim = time.time() - t_start

    # ------------------------------------------------------------------ plotting
    import matplotlib
    matplotlib.use("Agg")                                  # file output only: faster, no GUI windows
    import matplotlib.pyplot as plt, matplotlib.patheffects as pe
    if DARK:
        plt.style.use("dark_background")
        FG, EVC, SUF, OUTLINE = "w", "#A0AAB4", "_dark", "#000000"
        NBOX, PBOX, NTXT, PTXT = "#0C4A6E", "#7C2D12", "#7DD3FC", "#FDBA74"
    else:
        FG, EVC, SUF, OUTLINE = "k", "#0F766E", "", "#FFFFFF"
        NBOX, PBOX, NTXT, PTXT = "#E0F2FE", "#FFEDD5", "#0369A1", "#C2410C"
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.labelsize": 11, "legend.fontsize": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                         "grid.linestyle": "--", "lines.linewidth": 2.2, "mathtext.default": "regular"})
    PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]      # colour-blind-safe (Okabe-Ito)
    AREA = "device area 1 cm$^2$"
    from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator

    def iv_axes(ax):
        """Signed current, reverse + forward bias in ONE graph.
        y: symmetric-log axis (smooth through 0 A, no log-|I| 'needle').
        x: the reverse side (-2 ... 0 V) is drawn compressed to REV_SHARE of its true width, with true voltage labels,
           so the forward curves keep most of the width and stay easy to tell apart."""
        REV_SHARE = 0.2
        fwd = lambda x: np.where(np.asarray(x) < 0, np.asarray(x) * REV_SHARE, x)
        inv = lambda x: np.where(np.asarray(x) < 0, np.asarray(x) / REV_SHARE, x)
        ax.set_xscale("function", functions=(fwd, inv))
        ax.set_xlim(-2.0, 1.02)
        ax.xaxis.set_major_locator(FixedLocator([-2, -1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ("−" if v < 0 else "") + f"{abs(v):g}"))
        yt = [-1e-8, 0, 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2]
        ax.set_yscale("symlog", linthresh=1e-10, linscale=0.4)
        ax.set_ylim(-5e-8, 3e3)
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else ("−" if v < 0 else "") + f"$10^{{{int(round(np.log10(abs(v))))}}}$"))
        ax.axvspan(-2.0, 0, color="#94A3B8", alpha=0.12, lw=0, zorder=0)         # reverse-bias region
        ax.axhline(0, color="#999999", lw=0.9, zorder=1); ax.axvline(0, color="#64748B", lw=1.0, zorder=1)
        lab = dict(transform=ax.get_xaxis_transform(), ha="center", fontsize=9, style="italic", color="#64748B")
        ax.text(-1.0, 0.30, "Reverse bias\n(axis compressed)", va="bottom", **lab)
        ax.text(0.62, 0.05, "Forward bias", va="bottom", **lab)
        ax.set_xlabel("Voltage (V)"); ax.set_ylabel(f"Current (I) [A]   ({AREA})")

    def save(fig, fname):
        """Save a figure; retry if Windows has the file locked (image viewer / antivirus / sync)."""
        for i in range(6):
            try:
                fig.savefig(fname, dpi=300, bbox_inches="tight"); return
            except OSError as e:
                print(f"   [cannot write {fname} ({e}); retry {i+1}/6 - close the image if it is open]", flush=True)
                time.sleep(2)
        alt = fname.replace(".png", f"_{int(time.time())}.png")
        fig.savefig(alt, dpi=300, bbox_inches="tight"); print(f"   [saved as {alt} instead of {fname}]", flush=True)

    def draw_bands(ax, d, Nexp, hw, legend=True, tag=True):
        X, Ec, Ev, Efn, Efp = d
        ax.plot(X, Ec, color="#111827", lw=2.0, label="$E_c$ (conduction band)")
        ax.plot(X, Ev, color=EVC, lw=2.0, label="$E_v$ (valence band)")
        # quasi-Fermi levels coincide at equilibrium: wide translucent EFn underneath, dashed outlined EFp on top
        ax.plot(X, Efn, color="#3FA9F5", lw=6.0, alpha=0.65, solid_capstyle="butt", zorder=3, label="$E_{Fn}$ (electrons)")
        ax.plot(X, Efp, color="#E4380F", lw=2.0, ls=(0, (4, 2.5)), zorder=5,
                path_effects=[pe.withStroke(linewidth=3.6, foreground=OUTLINE)], label="$E_{Fp}$ (holes)")
        ax.axvline(0, color="#64748B", ls=":", lw=1.3)
        if tag:
            ax.text(0.25, 1.02, "n-type", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                    color=NTXT, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=NBOX, ec="none"))
            ax.text(0.75, 1.02, "p-type", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                    color=PTXT, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=PBOX, ec="none"))
        ax.set_xlim(-hw, hw); ax.set_ylim(-0.98, 0.98)
        ax.set_xlabel("Distance from the junction (µm)   [n side ← 0 → p side]"); ax.set_ylabel("Energy (eV)")
        if legend: ax.legend(loc="upper left", framealpha=0.9, handlelength=2.6)

    # ============================================================== (a1) effect of doping
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bands = {}
    for k, N in enumerate(Ns):
        V, J, b = R[("dark", N)]; J = SIGN * J
        ax.plot(V, J, color=PAL[k], label=f"$N_A = N_D = 10^{{{exp10(N)}}}$ cm$^{{-3}}$")
        np.savetxt(f"dark_doping_{N:.0e}.csv", np.c_[V, J], header="V(V) I(A)  [device area 1 cm2]")
        x, Ec, Ev, Efn, Efp = b
        bands[N] = (((x - 50e-4) * 1e4, Ec, Ev, Efn, Efp), 5 if N <= 1e15 else 3)
    iv_axes(ax)
    ax.set_title("I–V characteristics (dark): effect of doping"); ax.legend(loc="upper left")
    save(fig, "JV_doping" + SUF + ".png"); plt.close(fig)

    for N, (d, hw) in bands.items():                         # one band diagram per doping
        fig, ax = plt.subplots(figsize=(7, 4.5))
        draw_bands(ax, d, exp10(N), hw)
        ax.set_title(f"Energy band diagram, equilibrium ($N_A = N_D = 10^{{{exp10(N)}}}$ cm$^{{-3}}$)", pad=26)
        save(fig, f"bands_doping_{N:.0e}" + SUF + ".png"); plt.close(fig)

    if len(bands) == 4:                                      # all four in one figure (handy for the report)
        fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
        for a, (N, (d, hw)) in zip(axs.ravel(), bands.items()):
            draw_bands(a, d, exp10(N), hw, legend=False)
            a.set_title(f"$N_A = N_D = 10^{{{exp10(N)}}}$ cm$^{{-3}}$", pad=24)
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=True, handlelength=2.6)
        fig.suptitle("Equilibrium band diagrams: effect of doping", fontweight="bold", fontsize=14)
        fig.tight_layout(rect=[0, 0.06, 1, 0.97]); save(fig, "bands_doping_all" + SUF + ".png"); plt.close(fig)

    if QUICK:
        V, J, _ = R[("sun", 1.0)]
        print('QUICK TEST (expect I(0V) about +/-31 mA under 1 sun):')
        print('V  :', np.round(V[:4], 3), '...'); print('I  :', np.round(J[:4]*1e3, 2), 'mA ...')
        print(f"Time: simulations {t_sim:.1f} s, total {time.time()-t_start:.1f} s")
        return

    # ============================================================== (a2) effect of thickness (NA = ND = 1e17)
    # I-V curves only (thin lines, different styles so overlapping curves stay visible);
    # the current at VREF for each thickness is printed and saved in thickness_table.csv instead of a second panel
    VREF = 0.7
    STY = {2: dict(ls="-", lw=2.0), 10: dict(ls="-", lw=2.0), 50: dict(ls="--", lw=2.0), 200: dict(ls=":", lw=2.8)}
    trows = []
    for label in ("Wp", "Wn"):
        fig, ax1 = plt.subplots(figsize=(6.4, 4.6))
        for k, w_um in enumerate(WS):
            V, J, _ = R[("thick", label, w_um)]; J = SIGN * J
            ax1.plot(V, J, color=PAL[k], label=f"${label[0]}_{label[1]}$ = {w_um} µm", zorder=2+k, **STY[w_um])
            trows.append((0 if label == "Wp" else 1, w_um, float(np.interp(VREF, V, J))))
        side = "p-side" if label == "Wp" else "n-side"
        iv_axes(ax1); ax1.legend(loc="upper left")
        ax1.set_title(f"I–V (dark): {side} thickness")
        fig.tight_layout(); save(fig, f"JV_thickness_{label}{SUF}.png"); plt.close(fig)
    print(f"Current I at {VREF} V (A):")
    for side_id, w_um, j in trows: print(f"   {'Wp' if side_id == 0 else 'Wn'} = {w_um:>3} um :  {j:.3g} A")
    np.savetxt("thickness_table.csv", np.array(trows), delimiter=",",
               header=f"side(0=Wp,1=Wn), thickness(um), I at {VREF} V (A)   [area 1 cm2]")

    # ============================================================== (b) optimal solar cell
    V, J, _ = R[("sun", 0.0)]; Idark = -SIGN*1e3*J
    res, curves = [], {}
    for sun in SUNS:
        V, J, _ = R[("sun", sun)]; I = -SIGN*1e3*J                             # mA, photocurrent positive
        k = np.where(I < 0)[0]
        Voc = V[k[0]-1] + (V[k[0]]-V[k[0]-1])*I[k[0]-1]/(I[k[0]-1]-I[k[0]]) if len(k) else float('nan')
        Pw = V*I; m = int(np.argmax(Pw)); eta = Pw[m]/(100*sun)*100; FF = Pw[m]/(I[0]*Voc)*100
        curves[sun] = (V, I, Pw, m, Voc)
        res.append((sun*1000, I[0], Voc, FF, Pw[m], eta))
        print(f"solar cell, {sun*1000:.0f} W/m2:  Isc={I[0]:.2f} mA (Jsc = same number in mA/cm2, area 1 cm2)  Voc~{Voc:.3f} V  FF={FF:.1f} %  eta={eta:.2f} %")
    np.savetxt("solar_results.csv", np.array(res), delimiter=",",
               header="irradiance(W/m2), Isc(mA), Voc(V), FF(%), Pmax(mW), efficiency(%)   [area 1 cm2]")

    fig, ax1 = plt.subplots(figsize=(6.6, 4.8))
    ax1.plot(Vs, Idark, color=FG, lw=2.2, ls="--", label="dark", zorder=3)
    for c, (sun, (V, I, Pw, m, Voc)) in zip(PAL, curves.items()):
        ax1.plot(V, I, color=c, label=f"{sun*1000:.0f} W/m$^2$", zorder=2)
    ax1.axhline(0, color="#999999", lw=0.9, zorder=1)
    V1, I1, P1, m1, Voc1 = curves[1.0]                                       # annotate the 1-sun curve
    ax1.plot([0], [I1[0]], "o", color=PAL[2], ms=7, mec=FG, zorder=5); ax1.plot([Voc1], [0], "o", color=PAL[2], ms=7, mec=FG, zorder=5)
    ax1.annotate(f"$I_{{sc}}$ = {I1[0]:.1f} mA", (0, I1[0]), (0.04, I1[0]+3.5), fontsize=10, fontweight="bold")
    ax1.annotate(f"$V_{{oc}}$ = {Voc1:.3f} V", (Voc1, 0), (Voc1+0.04, 7), fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=FG, lw=1))
    ax1.set_xlim(0, 0.7); ax1.set_ylim(-50, 42)
    ax1.set_xlabel("Voltage (V)"); ax1.set_ylabel(f"Current (I) [mA]   ({AREA})")
    ax1.set_title("Solar cell I–V: dark vs illuminated"); ax1.legend(loc="upper right")
    fig.tight_layout(); save(fig, "JV_solar" + SUF + ".png"); plt.close(fig)
    print(f"Done. Time: simulations {t_sim:.1f} s, plots {time.time()-t_start-t_sim:.1f} s, total {time.time()-t_start:.1f} s")
    print("PNG and CSV files are in this folder (solar summary: solar_results.csv).")

if __name__ == "__main__":          # required: the parallel worker processes re-import this file
    main()