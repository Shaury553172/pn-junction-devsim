# DEVSIM 1-D Si pn junction: dark/illuminated I-V and band diagram  (cgs units, T = 300 K)
import os, sys, contextlib
for _v in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   
    os.environ.setdefault(_v, "1")
import numpy as np, devsim as ds
from devsim.python_packages.simple_physics import *
from devsim.python_packages.model_create import *
try: ds.set_parameter(name="threads_available", value=1)  
except Exception: pass
dev, reg = "pn", "Si"
PS_FAR = 2e-4       
CELL = (1e19, 2e16, 0.3e-4, 200e-4)   # proposed solar cell: Nd, Na, Wn, Wp
ABS_ERR, REL_ERR = 1e6, 1e-8         

ABS_FALLBACK = (1e8, 3e8, 1e9, 3e9, 1e10)   # extra steps between 1e8 and 1e10 so the garbage 1e10 is almost never reached
BANDS=[(0.35,46,1.1e6),(0.45,150,2.5e4),(0.55,160,7.0e3),(0.65,150,2.8e3),
       (0.75,115,1.3e3),(0.85,95,5.3e2),(0.95,70,1.6e2),(1.05,55,4.0)]   

def photogen(sun, R=0.05):                       
    t=[f"{(1-R)*P*sun*1e-4*lam*1e-6/1.9864e-25*a:.6e}*exp(-{a}*x)" for lam,P,a in BANDS]
    return "+".join(t) if sun>0 else "0"

def build(Nd, Na, Wn, Wp, sun=0.0, tau0=1e-5, Nref=1e16):
    for f, k in ((ds.delete_device, "device"), (ds.delete_mesh, "mesh")):   
        try: f(**{k: dev if k == "device" else "m"})
        except Exception: pass
    ds.create_1d_mesh(mesh="m")                                   # --- mesh (graded, cm)
    
    ps_top = 2e-7 if sun > 0 else min(PS_FAR, Wn/20)
    ps_bot = min(PS_FAR, Wp/20)
    ds.add_1d_mesh_line(mesh="m", pos=0,     ps=ps_top, tag="top")
    ds.add_1d_mesh_line(mesh="m", pos=Wn,    ps=min(2e-7, Wn/20, Wp/20), tag="mid")
    ds.add_1d_mesh_line(mesh="m", pos=Wn+Wp, ps=ps_bot, tag="bot")
    ds.add_1d_contact(mesh="m", name="top", tag="top", material="metal")
    ds.add_1d_contact(mesh="m", name="bot", tag="bot", material="metal")
    ds.add_1d_region(mesh="m", material="Si", region=reg, tag1="top", tag2="bot")
    ds.finalize_mesh(mesh="m"); ds.create_device(mesh="m", device=dev)
    SetSiliconParameters(dev, reg, 300)
    ds.set_parameter(device=dev, region=reg, name="Cn", value=2.8e-31)   
    ds.set_parameter(device=dev, region=reg, name="Cp", value=9.9e-32)
    CreateSolution(dev, reg, "Potential")                          # --- doping
    CreateNodeModel(dev, reg, "Acceptors", f"{Na}*step(x-{Wn})")
    CreateNodeModel(dev, reg, "Donors",    f"{Nd}*step({Wn}-x)")
    CreateNodeModel(dev, reg, "NetDoping", "Donors-Acceptors")
    CreateNodeModel(dev, reg, "Ntot", "Donors+Acceptors")
    # Caughey-Thomas mobility and doping-dependent lifetime as node models -> edge averages
    CreateNodeModel(dev, reg, "mun_n", "68.5+(1414-68.5)/(1+pow(Ntot/9.2e16,0.711))")
    CreateNodeModel(dev, reg, "mup_n", "44.9+(470.5-44.9)/(1+pow(Ntot/2.23e17,0.719))")
    CreateNodeModel(dev, reg, "tau_n", f"{tau0}/(1+Ntot/{Nref})")
    for m in ("mun_n", "mup_n"): ds.edge_from_node_model(device=dev, region=reg, node_model=m)
    ds.edge_model(device=dev, region=reg, name="mu_n", equation="0.5*(mun_n@n0+mun_n@n1)")
    ds.edge_model(device=dev, region=reg, name="mu_p", equation="0.5*(mup_n@n0+mup_n@n1)")
    CreateSiliconPotentialOnly(dev, reg)                           # --- equilibrium (Poisson only)
    for c in ds.get_contact_list(device=dev):
        ds.set_parameter(device=dev, name=GetContactBiasName(c), value=0.0)
        CreateSiliconPotentialOnlyContact(dev, reg, c)
    ds.solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)
    CreateSolution(dev, reg, "Electrons"); CreateSolution(dev, reg, "Holes")
    ds.set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
    ds.set_node_values(device=dev, region=reg, name="Holes",     init_from="IntrinsicHoles")
    CreateSiliconDriftDiffusion(dev, reg, mu_n="mu_n", mu_p="mu_p")  # --- drift-diffusion
    
    U=("(Electrons*Holes-n_i^2)/(tau_n*(Electrons+n_i)+tau_n*(Holes+n_i))"
       "+(Cn*Electrons+Cp*Holes)*(Electrons*Holes-n_i^2)")
    CreateNodeModel(dev, reg, "PhotoGen", photogen(sun))
    for nm in ("USRH",):
        CreateNodeModel(dev, reg, nm, U); CreateNodeModelDerivative(dev, reg, nm, U, "Electrons", "Holes")
    # sign convention of simple_physics: ElectronGeneration=-q(U-G), HoleGeneration=+q(U-G)
    for nm, sg in (("ElectronGeneration", "-"), ("HoleGeneration", "+")):
        e=f"{sg}ElectronCharge*(USRH-PhotoGen)"
        CreateNodeModel(dev, reg, nm, e); CreateNodeModelDerivative(dev, reg, nm, e, "Electrons", "Holes")
    for c in ds.get_contact_list(device=dev):
        CreateSiliconDriftDiffusionAtContact(dev, reg, c)
    ds.solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=30)   # equilibrium: original tolerance

def _save():
    return {n: ds.get_node_model_values(device=dev, region=reg, name=n) for n in ("Potential", "Electrons", "Holes")}
def _restore(sv):
    for n, v in sv.items(): ds.set_node_values(device=dev, region=reg, name=n, values=v)
def _current():
    return (ds.get_contact_current(device=dev, contact="bot", equation="ElectronContinuityEquation")
          + ds.get_contact_current(device=dev, contact="bot", equation="HoleContinuityEquation"))

def _current_dark(V):
    """Dark diode: |I| from the TOP contact (the bottom read-out loses nA currents to round-off for thick p-regions),
    sign from the bias (I and V always have the same sign in the dark), so it does not depend on DEVSIM's sign convention."""
    t = (ds.get_contact_current(device=dev, contact="top", equation="ElectronContinuityEquation")
       + ds.get_contact_current(device=dev, contact="top", equation="HoleContinuityEquation"))
    return float(np.sign(V)) * abs(t)

def _solve():
    """Newton solve with the strict tolerance first; if it fails, continue with looser absolute limits."""
    err = None
    for ae in (ABS_ERR,) + ABS_FALLBACK:
        try:
            ds.solve(type="dc", absolute_error=ae, relative_error=REL_ERR, maximum_iterations=30); return
        except Exception as e:
            err = e
    raise err

def sweep(volts, h0=0.02, min_step=1e-3, dark=False, hmax=0.1):
    """Bias the p contact ("bot") through `volts` with automatic step halving.
    Returns (V, J): the voltages actually reached and current density [A/cm2]."""
    bias = GetContactBiasName("bot"); cur = ds.get_parameter(device=dev, name=bias)
    Vout, J = [], []
    for target in volts:
        h = float(np.clip(target - cur, -hmax, hmax)) or h0  
        while abs(target - cur) > 1e-12:
            h = np.sign(target - cur)*min(abs(h), abs(target - cur))
            sv = _save()
            ds.set_parameter(device=dev, name=bias, value=cur + h)
            try:
                _solve()
                cur += h; h *= 1.5
            except Exception:
                _restore(sv); ds.set_parameter(device=dev, name=bias, value=cur); h /= 2
                if abs(h) < min_step:
                    print(f"   [stopped: no convergence beyond V = {cur:.3f} V]")
                    return np.array(Vout), np.array(J)
        Vout.append(cur); J.append(_current_dark(cur) if dark else _current())
    return np.array(Vout), np.array(J)

def bands():                                     # E_C, E_V, E_Fn, E_Fp [eV] at current bias
    x=np.array(ds.get_node_model_values(device=dev, region=reg, name="x"))
    psi=np.array(ds.get_node_model_values(device=dev, region=reg, name="Potential"))
    n=np.array(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
    p=np.array(ds.get_node_model_values(device=dev, region=reg, name="Holes")); Vt=0.025852; ni=1e10
    Efn=-(psi-Vt*np.log(n/ni)); Efp=-(psi+Vt*np.log(p/ni))
    return x, -psi+0.56, -psi-0.56, Efn, Efp

# ---------------------------------------------------------------- speed helpers (used by run_all.py)
def silence():
    """Discard DEVSIM's console output for good (used in worker processes).
    Done at Python level (sys.stdout): redirecting the OS file descriptor breaks Python's console writer on Windows."""
    sys.stdout = open(os.devnull, "w")

@contextlib.contextmanager
def quiet():
    """Temporarily discard DEVSIM's console output (printing every Newton iteration is slow, especially on Windows)."""
    saved, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close(); sys.stdout = saved

def run_dark(Nd, Na, Wn, Wp, volts, want_bands=False, volts_rev=None):
    """Dark I-V. volts = forward points (starting at 0), volts_rev = reverse points (starting at 0, going negative).
    Returns V (ascending, reverse + forward), J and the equilibrium bands (if requested)."""
    build(Nd, Na, Wn, Wp)
    b = bands() if want_bands else None        # equilibrium bands: the device is at 0 V right after build()
    Vr, Jr = np.array([]), np.array([])
    if volts_rev is not None:
        Vr, Jr = sweep(volts_rev, dark=True)              # 0 -> negative voltages
        build(Nd, Na, Wn, Wp)                  # fresh equilibrium for the forward sweep
    V, J = sweep(volts, dark=True)
    if len(Vr) > 1: V, J = np.r_[Vr[:0:-1], V], np.r_[Jr[:0:-1], J]    # reverse part (without its 0 V point) + forward part
    return V, J, b

def run_solar(sun, volts):
    build(*CELL, sun=sun)
    V, J = sweep(volts)
    return V, J, None

def job(spec):
    """spec = (tag, function name, kwargs) -> (tag, result). Top-level so worker processes can run it."""
    tag, fname, kw = spec
    try:
        return tag, globals()[fname](**kw)
    except BaseException as e:                
        import traceback
        return tag, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

if __name__ == "__main__":
    # (a1) doping, (a2) thickness, (b) optimal cell: Nd=1e19, Na=2e16, Wn=0.3 um, Wp=200 um
    build(1e17, 1e17, 50e-4, 50e-4);          Vd, Jdark = sweep(np.arange(0, 1.0001, 0.05))
    build(1e19, 2e16, 0.3e-4, 200e-4, sun=1); Vl, Jlight = sweep(np.arange(0, 0.7001, 0.01))
