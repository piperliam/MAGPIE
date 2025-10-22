
# ============================================================
# MAGPIE Mars Transfer & MOI Simulator
# ============================================================
# Liam Piper 2025
# Determines Delta V for Orbiter S/C Bus, with all burns and determines launch vehicle


import warnings
warnings.filterwarnings('ignore', message=r'ERFA function')

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Sun, Earth
from poliastro.iod.izzo import lambert
from poliastro.twobody import Orbit
from skyfield.api import load
import datetime
import os

# ===================== USER TOGGLES =====================
# Launch Vehicle selection:
USE_FALCON_HEAVY = False      # False = Falcon 9, True = Falcon Heavy (expendable)
LV_RESERVE_FRAC  = 0.03       # Fraction of LV Stage-2 propellant held as reserve (never touched)

# ---- LV2 underload model ----
LV2_ASCENT_SCALING   = "linear"   # "linear" or "sqrt"
LV2_PROP_TO_ORBIT_AT_CAP_FRAC = 0.78
ALLOW_LV2_POST_INSERTION_BURN = True

# STAR kickstage:
USE_STAR           = False
STAR_MODE          = "fixed"   # "fixed" | "autosize" | "off"
AUTO_DISABLE_STAR  = True
FORCE_KEEP_STAR    = False

# Aerocapture + MOI shaping:
USE_AEROCAPTURE     = False         # Enable aerocapture option for MOI
MINIMUM_ORBIT_MODE  = False        # If True, skip dv3 circularize

# Porkchop search (days from seed, and number of candidates to report):
# Porkchop just finds minimal delta v needed to do the burns
PORKCHOP_SEARCH_DAYS = 600
PORKCHOP_LIST_TOP    = 12

# GO-ANYWAY behavior
GO_ANYWAY_IF_NO_C3   = True

# Plot control
SHOW_PLOTS = True

# ===================== CONSTANTS & VEHICLE DATA =====================
g0      = 9.80665
mu_e    = Earth.k.to(u.m**3/u.s**2).value
mu_m    = 4.282837e13
R_e     = 6371e3
R_m     = 3389.5e3
h_LEO   = 350e3
h_MOI   = 250e3
GM_sun  = 1.32712440018e20

# Spacecraft (MONARC bus) (need better name) 
m_sc_wet = 7507.0    # kg   (includes MONARC prop when "full")
m_sc_dry =  3562.0    # kg
isp_monarc     = 235.0  # s
thrust_monarc  = 445.0  # N

# STAR kickstage (I think a star 26 - but this is never really used) 
star_isp      = 292.0   # s
star_prop_max = 3470.0  # kg
star_dry      =  230.0  # kg

# Launch vehicle Stage-2 (approximate)
LV2_adaptor = 1000.0      # Extra Gubins
LV2_ISP     = 348.0     # Merlin Vac ~348 s
LV2_PROP    = 110000.0  # kg prop in stage-2 (generic)
LV2_DRY     = 4000.0 + LV2_adaptor   # kg dry stage-2
LEO_CAP_F9  = 22800.0   # kg
LEO_CAP_FH  = 63800.0   # kg

# Mission geometry
EI_ALT_M            = 250e3
PERIAPSIS_ALT_M     =  40e3
PLANE_CHANGE_DEG    = 120.0
APOAPSIS_RATIO      = 150.0

# Burn discretization (perigee-pass “chunks”)
PER_PASS_TIME_MIN_DEP = 10.0
PER_PASS_TIME_MIN_MOI = 8.0
MAX_PASSES_PER_PHASE  = 200

# Porkchop seed date
depart_dt_seed = datetime.datetime(2028, 11, 25, tzinfo=datetime.timezone.utc)

# ===================== AEROCAPTURE PHYSICS TOGGLES =====================
AEROCAPTURE_PHYSICS  = True    # if True, replace energy bookkeeping with atmosphere simulation
ENTRY_USE_LIFT       = False   # False=ballistic; True=lifting entry (false just means worse sim, but lift is not properly implemnented yet)
ENTRY_BANK_DEG       = 0.0     # constant bank (deg)
CL                   = 0.3     # set ~0.2–0.4 (need to determine better) if ENTRY_USE_LIFT=True
CD                   = 1.5
REF_AREA_M2          = 12.0
NOSE_RADIUS_M        = 0.5     # m (Sutton–Graves radius)
SG_K                 = 1.2e-4  # Sutton–Graves constant (SI): q_dot = K*sqrt(rho/Rn)*V^3

# Atmosphere model selection: "exponential" | "layered_exp" | "tabulated"
ATMOSPHERE_MODEL     = "layered_exp"
# density scaling factor to emulate season/lat (applies to all models) (needs to be expanded)
ATM_DENSITY_SCALE    = 1.0

# Exponential model params
MARS_RHO0            = 0.020   # kg/m^3 near surface
MARS_SCALE_HEIGHT    = 11100.0 # m

# Layered exponential (alt_km upper bounds, and scale heights per layer in m)
_LAY_ALT_KM   = np.array([7, 20, 40, 60, 80, 120, 160, 200, 300], dtype=float)
_LAY_H_M      = np.array([10e3, 11e3, 12e3, 13e3, 15e3, 18e3, 22e3, 30e3, 40e3], dtype=float)
_LAY_RHO0     = 0.020  # surface reference; layers are continuous via cumulative construction below

# Tabulated profile (coarse, mean atmosphere; alt[km], rho[kg/m^3], T[K])
_TAB_ALT_KM = np.array(
    [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300],
    dtype=float
)
_TAB_RHO = np.array(
    [0.0200, 0.0155, 0.0120, 0.0092, 0.0070, 0.0053, 0.0040, 0.0023, 0.0013, 7.5e-4,
     4.4e-4, 2.5e-4, 1.4e-4, 8.0e-5, 3.0e-5, 1.2e-5, 5.5e-6, 2.6e-6, 1.3e-6, 4.0e-7, 1.5e-7],
    dtype=float
)
_TAB_TEMP = np.array(
    [210, 205, 200, 195, 190, 185, 180, 175, 170, 165,
     160, 155, 150, 150, 150, 150, 150, 150, 150, 150, 150],
    dtype=float
)

# Entry integration
ENTRY_DT             = 0.25    # s
ENTRY_MAX_DURATION   = 3*3600  # s
ENTRY_FPA_SOLVE      = True    # shoot for target periapsis altitude
FPA_BOUNDS_DEG       = (-15.0, -1.0)
FPA_SOLVE_TOL_M      = 1000.0  # 1 km

# Safety / design limits (these are just guesses)
Q_LIMIT_PA           = 30e3     # 30 kPa
HEAT_LIMIT_WM2       = 1.0e6    # 1 MW/m^2
G_LIMIT              = 8.0      # 8 g

# ===================== HELPERS =====================
def _as_quantity_kms(x):
    if hasattr(x, "unit"):
        return x.to(u.km/u.s)
    return np.asarray(x) * (u.km/u.s)

def vec_norm(q):
    if hasattr(q, "unit"):
        arr = q.to(u.km/u.s).value
        return np.sqrt(np.sum(arr*arr)) * (u.km/u.s)
    arr = np.asarray(q)
    return np.sqrt(np.sum(arr*arr))

def solve_lambert(mu, r0, r1, tof):
    res = lambert(mu, r0, r1, tof)
    import types
    try:
        if isinstance(res, types.GeneratorType):
            first = next(res)
            if isinstance(first, tuple) and len(first) == 2:
                return _as_quantity_kms(first[0]), _as_quantity_kms(first[1])
            if hasattr(first, "v0") and hasattr(first, "v1"):
                return _as_quantity_kms(first.v0), _as_quantity_kms(first.v1)
            arr = np.array(first)
            if arr.shape == (2, 3):
                return _as_quantity_kms(arr[0]), _as_quantity_kms(arr[1])
            raise RuntimeError("Unsupported lambert() generator element type")
    except StopIteration:
        raise RuntimeError("lambert() generator yielded no solutions")
    if isinstance(res, tuple) and len(res) == 2:
        return _as_quantity_kms(res[0]), _as_quantity_kms(res[1])
    if hasattr(res, "v0") and hasattr(res, "v1"):
        return _as_quantity_kms(res.v0), _as_quantity_kms(res.v1)
    if isinstance(res, (list, tuple)) and len(res) > 0:
        first = res[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            return _as_quantity_kms(first[0]), _as_quantity_kms(first[1])
    arr = np.array(res)
    if arr.shape == (2, 3):
        return _as_quantity_kms(arr[0]), _as_quantity_kms(arr[1])
    raise RuntimeError(f"Unsupported lambert() return type: {type(res)}")

def rocket_eq_delta_v(isp_s, m0, mf):
    if mf <= 0 or mf >= m0: return 0.0
    return isp_s * g0 * np.log(m0 / mf)

def rocket_eq_prop_for_dv(isp_s, m_dry, dv_mps, m0_guess=None):
    mass_ratio = np.exp(dv_mps / (isp_s * g0))
    m0_needed  = mass_ratio * m_dry
    return max(0.0, m0_needed - m_dry)

def perigee_pass_burns_required(dv_target_kms, m0, isp_s, thrust_n, per_pass_min, max_passes, phase_name=""):
    dv_target_mps = dv_target_kms * 1000.0
    if dv_target_mps <= 1e-9:
        return [], 0.0, m0, 0.0
    passes = []
    delivered = 0.0
    t_total_min = 0.0
    m = m0
    dt = per_pass_min * 60.0
    for i in range(1, max_passes+1):
        a = thrust_n / m
        dv_mps = a * dt
        dm = m * (1.0 - np.exp(-dv_mps / (isp_s * g0)))
        m_new = m - dm
        delivered += dv_mps
        t_total_min += per_pass_min
        passes.append({
            "pass": i,
            "dv_kms": dv_mps/1000.0,
            "time_min": per_pass_min,
            "m_start": m,
            "m_end": m_new,
            "m_used": dm
        })
        m = m_new
        if delivered >= dv_target_mps - 1e-9:
            break
    return passes, delivered/1000.0, m, t_total_min

def _savefig(name):
    try:
        plt.savefig(f"{name}.png", dpi=130)
    except Exception as e:
        print("[plot save warning]", name, e)

# ===================== EPHEMERIDES =====================
planets = load('de421.bsp')
sun   = planets['sun']
earth = planets['earth']
mars  = planets['mars']
ts    = load.timescale()

# ================== PORKCHOP (yum) ===================
def porkchop_candidates(seed_dt, search_days, top_n):
    results = []
    for dd in range(0, search_days, 2):
        dep_dt = seed_dt + datetime.timedelta(days=dd)
        t0 = ts.utc(dep_dt.year, dep_dt.month, dep_dt.day, 0, 0, 0)

        rE_helio = (earth - sun).at(t0).position.km * 1e3
        rM_helio_guess = (mars - sun).at(t0).position.km * 1e3
        rE_norm  = np.linalg.norm(rE_helio)
        rM_norm  = np.linalg.norm(rM_helio_guess)
        a_trans  = 0.5 * (rE_norm + rM_norm)
        tof_s    = np.pi * np.sqrt(a_trans**3 / GM_sun)
        arr_dt   = dep_dt + datetime.timedelta(seconds=float(tof_s))
        t1 = ts.utc(arr_dt.year, arr_dt.month, arr_dt.day, 0, 0, 0)

        r0_km = (earth - sun).at(t0).position.km * u.km
        r1_km = (mars  - sun).at(t1).position.km * u.km
        try:
            v_dep, v_arr = solve_lambert(Sun.k, r0_km, r1_km, tof_s * u.s)
        except Exception:
            continue
        vE_kms = (earth - sun).at(t0).velocity.km_per_s * (u.km/u.s)
        v_inf_dep = vec_norm(v_dep - vE_kms).value

        r_LEO = R_e + h_LEO
        v_circ_e = np.sqrt(mu_e / r_LEO)
        v_esc_e  = np.sqrt(2.0 * mu_e / r_LEO)
        v_inf_m  = v_inf_dep * 1000.0
        dv_dep_mps = np.sqrt(v_inf_m**2 + v_esc_e**2) - v_circ_e
        results.append({
            "dep": dep_dt,
            "arr": arr_dt,
            "tof_days": int(tof_s//86400),
            "v_inf_dep": v_inf_dep,
            "dv_dep_kms": dv_dep_mps/1000.0
        })
    results.sort(key=lambda x: x["dv_dep_kms"])
    return results[:top_n]

# ------------- globals to feed plots -------------
_g_dv_lv2_used_kms = 0.0
_g_dv_star_used_kms = 0.0
_g_dv_monarc_dep_kms = 0.0

# ===================== ATMOSPHERE MODELS =====================
def atm_density_temp(h_m):
    """Return (rho, T) for Mars at altitude h [m] using selected ATMOSPHERE_MODEL."""
    h_km = max(0.0, h_m) / 1000.0
    if ATMOSPHERE_MODEL == "exponential":
        rho = MARS_RHO0 * np.exp(-h_m / MARS_SCALE_HEIGHT)
        T   = 200.0  # simple constant
        return ATM_DENSITY_SCALE * rho, T

    if ATMOSPHERE_MODEL == "layered_exp":
        # Build a continuous layered exponential using cumulative matching
        # Precompute base densities per layer edges from reference surface rho0
        alt_edges = np.concatenate(([0.0], _LAY_ALT_KM))
        Hs = _LAY_H_M
        rho = _LAY_RHO0
        base_rho = [_LAY_RHO0]
        for i in range(len(Hs)):
            dh = (alt_edges[i+1] - alt_edges[i]) * 1000.0
            rho = rho * np.exp(-dh / Hs[i])
            base_rho.append(rho)
        # find layer index
        idx = np.searchsorted(_LAY_ALT_KM, h_km)
        idx = min(idx, len(Hs)-1)
        # density within layer
        h0_km = alt_edges[idx]
        rho0_layer = base_rho[idx]
        H = Hs[idx]
        dh = (h_km - h0_km) * 1000.0
        rho = rho0_layer * np.exp(-dh / H)
        # crude temperature lapse (optional, just helpful)
        T = 210.0 - 0.5*h_km
        T = max(145.0, min(230.0, T))
        return ATM_DENSITY_SCALE * rho, T

    # "tabulated" (interpolate)
    rho = np.interp(h_km, _TAB_ALT_KM, _TAB_RHO, left=_TAB_RHO[0], right=_TAB_RHO[-1])
    T   = np.interp(h_km, _TAB_ALT_KM, _TAB_TEMP, left=_TAB_TEMP[0], right=_TAB_TEMP[-1])
    return ATM_DENSITY_SCALE * rho, T

def sutton_graves_heat_flux_Wm2(rho, V, Rn=NOSE_RADIUS_M, K=SG_K):
    Rn_eff = max(Rn, 1e-4)
    return K * np.sqrt(max(rho, 0.0) / Rn_eff) * (V**3)

# ===================== ENTRY DYNAMICS =====================
def entry_dynamics(t, state, planet_mu=mu_m, R_body=R_m, CD=CD, CL=CL, A=REF_AREA_M2,
                   m=m_sc_wet, bank_deg=ENTRY_BANK_DEG, use_lift=ENTRY_USE_LIFT):
    """
    3-DOF point-mass in planet-centric radial-velocity-flightpath frame.
    State = [r, v, gamma] (radius [m], speed [m/s], flight-path angle [rad])
    """
    r, v, gamma = state
    h = r - R_body
    rho, _T = atm_density_temp(h)
    q   = 0.5 * rho * v*v
    D   = q * CD * A
    L   = (q * CL * A) if use_lift else 0.0
    bank = np.deg2rad(bank_deg)
    L_n  = L * np.cos(bank)   # normal component in vertical plane
    g    = planet_mu / (r*r) # any mu'ers out there?

    r_dot     = v * np.sin(gamma)
    v_dot     = -(D/m) - g * np.sin(gamma)
    gamma_dot = 0.0
    if v > 1e-3:
        gamma_dot = (L_n/(m*v)) + (v/r - g/v) * np.cos(gamma)

    # Diagnostics
    heat_Wm2 = sutton_graves_heat_flux_Wm2(rho, v)
    # Use vertical-plane resultant accel for g-load proxy
    a_mag    = np.sqrt((D/m + g*np.sin(gamma))**2 + (L_n/(m))**2)

    return np.array([r_dot, v_dot, gamma_dot]), {
        "rho": rho, "q": q, "D": D, "L": L_n, "heat": heat_Wm2, "a": a_mag
    }
#me when rk4 is not in matlab so I have to make
def rk4_step(fun, t, y, dt): 
    k1, d1 = fun(t, y)
    k2, d2 = fun(t+0.5*dt, y + 0.5*dt*k1)
    k3, d3 = fun(t+0.5*dt, y + 0.5*dt*k2)
    k4, d4 = fun(t+dt,     y + dt*k3)
    y_next = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y_next, d4

def simulate_entry_pass(v_inf_arr_mps, gamma0_deg, rp_target_alt_m,
                        EI_alt_m=EI_ALT_M, dt=ENTRY_DT, max_dur=ENTRY_MAX_DURATION,
                        R_body=R_m, mu=mu_m, m=m_sc_wet):
    """Integrate from EI with initial FPA gamma0 to exit above EI. Return diagnostics & histories."""
    r_EI = R_body + EI_alt_m
    v_EI = np.sqrt(v_inf_arr_mps**2 + 2*mu / r_EI)
    gamma0 = np.deg2rad(gamma0_deg)

    t = 0.0
    y = np.array([r_EI, v_EI, gamma0], dtype=float)

    max_q = 0.0
    max_g = 0.0
    max_heat = 0.0
    heat_int = 0.0
    min_alt = EI_alt_m
    atmo_dv = 0.0

    hist_t=[]; hist_alt=[]; hist_v=[]; hist_q=[]; hist_g=[]; hist_heat=[]

    last_v = v_EI
    exited = False
    while t < max_dur:
        y, diag = rk4_step(lambda tt, yy: entry_dynamics(tt, yy, planet_mu=mu, R_body=R_body,
                                                         CD=CD, CL=CL, A=REF_AREA_M2, m=m,
                                                         bank_deg=ENTRY_BANK_DEG, use_lift=ENTRY_USE_LIFT),
                           t, y, dt)
        r, v, gamma = y
        alt = r - R_body
        if alt < 0:
            # impact (invalid for aerocapture), impact = bad
            break

        # accumulate with basic heat model
        q = diag["q"]; gload = diag["a"] / g0; heat = diag["heat"]
        max_q   = max(max_q, q)
        max_g   = max(max_g, gload)
        max_heat= max(max_heat, heat)
        heat_int += heat * dt
        min_alt = min(min_alt, alt)
        atmo_dv += max(0.0, last_v - v)
        last_v = v

        # store
        hist_t.append(t)
        hist_alt.append(alt)
        hist_v.append(v)
        hist_q.append(q)
        hist_g.append(gload)
        hist_heat.append(heat)

        # exit condition: back out above EI altitude after at least some time
        if alt > EI_alt_m and t > 10.0:
            exited = True
            break

        t += dt

    # rough measure: achieved periapsis alt ~ min_alt encountered
    results = {
        "time_s": np.array(hist_t),
        "alt_m": np.array(hist_alt),
        "v_mps": np.array(hist_v),
        "q_Pa": np.array(hist_q),
        "g_load": np.array(hist_g),
        "heat_Wm2": np.array(hist_heat),
        "max_q_Pa": max_q,
        "max_g": max_g,
        "max_heat_Wm2": max_heat,
        "heat_load_Jm2": heat_int,
        "min_alt_m": min_alt,
        "atmo_dv_mps": atmo_dv,
        "exited": exited,
        "gamma0_deg": gamma0_deg
    }
    return results

def solve_entry_fpa_for_target(v_inf_arr_mps, rp_target_alt_m, fpa_bounds_deg=FPA_BOUNDS_DEG,
                               tol_m=FPA_SOLVE_TOL_M, max_iter=18):
    """Bisection on gamma0 to match min altitude to rp_target_alt_m (monotonic assumption)."""
    lo, hi = fpa_bounds_deg
    # evaluate ends
    r_lo = simulate_entry_pass(v_inf_arr_mps, lo, rp_target_alt_m)
    r_hi = simulate_entry_pass(v_inf_arr_mps, hi, rp_target_alt_m)
    def alt_err(res): return (res["min_alt_m"] - rp_target_alt_m)

    e_lo = alt_err(r_lo)
    e_hi = alt_err(r_hi)

    # If both on same side, return the one closer
    if e_lo*e_hi > 0:
        return r_lo if abs(e_lo) < abs(e_hi) else r_hi

    best = r_lo if abs(e_lo) < abs(e_hi) else r_hi
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        r_mid = simulate_entry_pass(v_inf_arr_mps, mid, rp_target_alt_m)
        e_mid = alt_err(r_mid)
        if abs(e_mid) < tol_m:
            return r_mid
        # choose bracket
        if e_lo*e_mid <= 0:
            hi = mid; r_hi = r_mid; e_hi = e_mid
        else:
            lo = mid; r_lo = r_mid; e_lo = e_mid
        best = r_mid if abs(e_mid) < abs(alt_err(best)) else best
    return best

# ===================== MAIN RUN =====================
def main():
    global _g_dv_lv2_used_kms, _g_dv_star_used_kms, _g_dv_monarc_dep_kms

    # --- Porkchop search ---
    cands = porkchop_candidates(depart_dt_seed, PORKCHOP_SEARCH_DAYS, PORKCHOP_LIST_TOP)
    print("\n=== Porkchop Top (by departure Δv only) ===")
    for i, c in enumerate(cands, 1):
        print(f"{i:2d}. dep {c['dep'].date()} | TOF {c['tof_days']:3d} d | v∞dep {c['v_inf_dep']:.3f} | "
              f"dep Δv {c['dv_dep_kms']:.3f} km/s")
    best = cands[0]
    depart_dt = best["dep"]
    arrival_dt = best["arr"]
    tof_s   = (arrival_dt - depart_dt).total_seconds()

    print(f"\n[Auto-apply] Using dep {depart_dt.date()} → arr {arrival_dt.date()} from porkchop best.\n")

    t0 = ts.utc(depart_dt.year, depart_dt.month, depart_dt.day, 0, 0, 0)
    t1 = ts.utc(arrival_dt.year, arr_dt.month, arr_dt.day, 0, 0, 0) if False else ts.utc(arrival_dt.year, arrival_dt.month, arrival_dt.day, 0, 0, 0)
    r0_km   = (earth - sun).at(t0).position.km * u.km
    r1_km   = (mars  - sun).at(t1).position.km * u.km
    v_dep, v_arr = solve_lambert(Sun.k, r0_km, r1_km, tof_s * u.s)
    vE_kms = (earth - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    vM_kms = (mars  - sun).at(t1).velocity.km_per_s * (u.km/u.s)
    v_inf_dep = vec_norm(v_dep - vE_kms)   # km/s
    v_inf_arr = vec_norm(v_arr - vM_kms)   # km/s

    # Required LEO→C3 (from earth)
    r_LEO    = R_e + h_LEO
    v_circ_e = np.sqrt(mu_e / r_LEO)
    v_esc_e  = np.sqrt(2.0 * mu_e / r_LEO)
    v_inf_m  = v_inf_dep.to(u.m/u.s).value
    dv_dep_mps = np.sqrt(v_inf_m**2 + v_esc_e**2) - v_circ_e
    c3 = (v_inf_dep.value)**2  # km^2/s^2

    # ===== Print high-level transfer numbers =====
    days  = int(tof_s // 86400)
    hours = int((tof_s % 86400) // 3600)
    mins  = int((tof_s % 3600) // 60)
    print(f"TOF: {days}d {hours}h {mins}m")
    print(f"C3 (km^2/s^2):                {c3:,.3f}")
    print(f"Departure v_inf:              {v_inf_dep.value:.3f} km/s")
    print(f"Departure burn Δv (LEO→C3):   {dv_dep_mps/1000.0:.3f} km/s")

    # ===== Assemble stack at LEO =====
    stack_mass = m_sc_wet
    star_prop = 0.0
    star_used = False
    star_auto_dropped = False

    if USE_STAR and STAR_MODE != "off":
        if STAR_MODE == "fixed":
            star_prop = min(star_prop_max, star_prop_max)
        elif STAR_MODE == "autosize":
            star_prop = star_prop_max
        stack_mass += (star_prop + star_dry)

    leo_cap = LEO_CAP_FH if USE_FALCON_HEAVY else LEO_CAP_F9
    lv_name = "FH" if USE_FALCON_HEAVY else "F9"
    if stack_mass > leo_cap:
        print(f"[ABORT] Stack @LEO {stack_mass:.1f} kg exceeds {lv_name} LEO capacity {leo_cap:.1f} kg.")
        return

    # ==== LV2 Underload Model ====
    payload_frac = max(0.0, min(1.0, stack_mass / leo_cap))
    exp = 1.0 if LV2_ASCENT_SCALING.lower().startswith("lin") else 0.5
    prop_to_orbit_at_cap = LV2_PROP_TO_ORBIT_AT_CAP_FRAC * LV2_PROP
    prop_to_orbit_actual = prop_to_orbit_at_cap * (payload_frac ** exp)
    prop_to_orbit_actual = max(0.0, min(LV2_PROP, prop_to_orbit_actual))

    prop_reserve = max(0.0, min(LV2_PROP, LV_RESERVE_FRAC * LV2_PROP))
    prop_leftover = max(0.0, LV2_PROP - prop_to_orbit_actual - prop_reserve)

    print("\n[LV Stage-2 Ascent Ledger]")
    print(f"  Vehicle: {lv_name}  | LEO cap: {leo_cap:.0f} kg")
    print(f"  Payload(stack) mass @LEO: {stack_mass:.1f} kg  ({payload_frac*100:.1f}% of cap)")
    print(f"  LV2 prop tank: {LV2_PROP:.0f} kg   | LV2 dry: {LV2_DRY:.0f} kg")
    print(f"  Prop to reach LEO (model): {prop_to_orbit_actual:.0f} kg [at cap {prop_to_orbit_at_cap:.0f} kg; scaling={LV2_ASCENT_SCALING}]")
    print(f"  Prop held as reserve:      {prop_reserve:.0f} kg")
    print(f"  Prop leftover (post-insert): {prop_leftover:.0f} kg")

    # LV2 post-insertion burn toward C3
    lv2_dv_used_mps = 0.0
    stack_after_lv = stack_mass
    if ALLOW_LV2_POST_INSERTION_BURN and prop_leftover > 1e-6:
        m0_lv_burn = LV2_DRY + prop_leftover + stack_mass
        mf_lv_burn = LV2_DRY + stack_mass
        lv2_dv_full = rocket_eq_delta_v(LV2_ISP, m0_lv_burn, mf_lv_burn)
        lv2_dv_used_mps = min(lv2_dv_full, dv_dep_mps)
        mf_if_used = m0_lv_burn / np.exp(lv2_dv_used_mps / (LV2_ISP * g0))
        prop_burned_now = (m0_lv_burn - mf_if_used)
        prop_leftover -= prop_burned_now
        stack_after_lv = mf_if_used - LV2_DRY
        print(f"\n[LV2 → C3] Using leftover toward LEO→C3:")
        print(f"  LV2 Δv avail (leftover): {lv2_dv_full/1000.0:.3f} km/s")
        print(f"  LV2 Δv used:             {lv2_dv_used_mps/1000.0:.3f} km/s")
        print(f"  LV2 prop burned now:     {prop_burned_now:.0f} kg")
    else:
        stack_after_lv = stack_mass
        if not ALLOW_LV2_POST_INSERTION_BURN:
            print("\n[LV2 → C3] Post-insertion burn disabled by toggle.")
        else:
            print("\n[LV2 → C3] No leftover prop available after reserve.")

    _g_dv_lv2_used_kms = lv2_dv_used_mps / 1000.0
    dv_dep_remain_mps = max(0.0, dv_dep_mps - lv2_dv_used_mps)

    # ===== STAR logic =====
    star_dv_used_mps = 0.0
    if dv_dep_remain_mps > 1e-6 and USE_STAR and STAR_MODE != "off":
        m_start = stack_after_lv
        dv_star_all = 0.0
        if star_prop > 0.0:
            m_end_if_all = m_start - star_prop
            dv_star_all = rocket_eq_delta_v(star_isp, m_start, m_end_if_all)
        dv_to_use = min(dv_dep_remain_mps, dv_star_all)
        if dv_to_use > 1e-6:
            mf_req = m_start / np.exp(dv_to_use / (star_isp * g0))
            star_prop_used = (m_start - mf_req)
            star_prop = max(0.0, star_prop - star_prop_used)
            star_used = True
            stack_after_lv = mf_req
            star_dv_used_mps = dv_to_use
            print(f"\n[STAR → C3] Δv used: {star_dv_used_mps/1000.0:.3f} km/s | prop burned: {star_prop_used:.0f} kg")
        if star_used or (STAR_MODE in ["fixed","autosize"] and not FORCE_KEEP_STAR):
            stack_after_lv -= star_dry
        else:
            if AUTO_DISABLE_STAR and dv_dep_remain_mps <= 1e-6 and not FORCE_KEEP_STAR:
                stack_after_lv -= (star_prop + star_dry)
                star_auto_dropped = True
    else:
        if USE_STAR and STAR_MODE != "off" and AUTO_DISABLE_STAR and not FORCE_KEEP_STAR and dv_dep_remain_mps <= 1e-6:
            stack_after_lv -= (star_prop + star_dry)
            star_auto_dropped = True

    _g_dv_star_used_kms = star_dv_used_mps / 1000.0
    dv_dep_remain_mps = max(0.0, dv_dep_remain_mps - star_dv_used_mps)

    # Departure ledger 
    print("\n[Departure Δv Ledger]")
    print(f"  Required LEO→C3 (raw):   {dv_dep_mps/1000.0:.3f} km/s")
    print(f"  LV2 leftover provided:    {_g_dv_lv2_used_kms:.3f} km/s")
    if USE_STAR and STAR_MODE != "off":
        if star_auto_dropped:
            print("  STAR: not required — auto-dropped from stack.")
        elif _g_dv_star_used_kms > 0:
            print(f"  STAR provided:            {_g_dv_star_used_kms:.3f} km/s")
        else:
            print("  STAR present, no Δv used.")
    else:
        print("  STAR: disabled.")
    print(f"  Remaining for MONARC:     {dv_dep_remain_mps/1000.0:.3f} km/s")

    monarc_prop_full = m_sc_wet - m_sc_dry

    # MONARC departure perigee-pass burns if needed
    dep_passes, dep_deliv_kms, m_after_dep, dep_t_min = [], 0.0, None, 0.0
    monarc_prop_used_dep = 0.0
    if dv_dep_remain_mps > 1e-6:
        dep_passes, dep_deliv_kms, m_after_dep, dep_t_min = perigee_pass_burns_required(
            dv_dep_remain_mps/1000.0, m_sc_wet, isp_monarc, thrust_monarc,
            PER_PASS_TIME_MIN_DEP, MAX_PASSES_PER_PHASE, "Departure MONARC"
        )
        monarc_prop_used_dep = m_sc_wet - m_after_dep
        print("\n[MONARC (LEO→C3)] Perigee-pass burns:")
        for p in dep_passes[:80]:
            print(f"  pass {p['pass']:02d}: Δv={p['dv_kms']:.4f} km/s, time={p['time_min']:.2f} min, "
                  f"mass {p['m_start']:.1f}→{p['m_end']:.1f} kg (used {p['m_used']:.1f} kg)")
        if len(dep_passes) > 80:
            print(f"  ... {len(dep_passes)-80} more passes not shown")
        if dep_deliv_kms >= dv_dep_remain_mps/1000.0 - 1e-6:
            print("\n--- MONARC Departure Summary ---")
            print(f"Delivered Δv:   {dep_deliv_kms:.3f} km/s in {len(dep_passes)} passes ({dep_t_min:.1f} min)")
            print(f"Prop used:      {monarc_prop_used_dep:.1f} kg ({100*monarc_prop_used_dep/monarc_prop_full:.1f}% of MONARC prop)")
    else:
        m_after_dep = m_sc_wet

    _g_dv_monarc_dep_kms = dep_deliv_kms

    achieved_escape = (dv_dep_remain_mps <= 1e-6) or (dep_deliv_kms >= dv_dep_remain_mps/1000.0 - 1e-6)
    if not achieved_escape and not GO_ANYWAY_IF_NO_C3:
        print("\n=== FINAL STATUS: Escape NOT achieved ===")
        print(f"Required dep Δv (net): {(dv_dep_remain_mps/1000.0):.3f} km/s")
        print(f"Delivered by MONARC:    {dep_deliv_kms:.3f} km/s")
        shortfall = max(0.0, dv_dep_remain_mps/1000.0 - dep_deliv_kms)
        print(f"Shortfall:              {shortfall:.3f} km/s")
        print("Consider enabling STAR or switching to FH, or widening porkchop search.")
        return

    if not achieved_escape and GO_ANYWAY_IF_NO_C3:
        m_final = m_sc_dry
        r_p = R_e + h_LEO
        v_p = np.sqrt(mu_e / r_p)
        v_p_new = v_p + dep_deliv_kms*1000.0
        a_new = 1.0 / (2.0/r_p - (v_p_new**2)/mu_e)
        e_new = 1.0 - r_p / a_new
        r_a = a_new * (1+e_new)
        print("\n=== FINAL STATUS: Escape NOT achieved ===")
        print("Earth-bound orbit achieved:")
        print(f"  rp = {r_p/1000.0:,.1f} km (alt {h_LEO/1000.0:,.1f} km)")
        print(f"  ra = {r_a/1000.0:,.1f} km (alt {r_a/1000.0 - R_e/1000.0:,.1f} km)")
        print(f"  a  = {a_new/1000.0:,.1f} km, e = {e_new:.6f}")
        T = 2*np.pi*np.sqrt(a_new**3 / mu_e)
        print(f"  Period = {T/3600.0:.2f} h")
        make_all_plots(depart_dt, arrival_dt, t0, tof_s, r0_km, v_dep, vE_kms, False, False,
                       final_orbit=("Earth", r_p, r_a),
                       dv_breakdown=(_g_dv_lv2_used_kms, _g_dv_star_used_kms, _g_dv_monarc_dep_kms),
                       entry_results=None)
        return

    # =============== Proceed to MOI (we reached escape) ===============
    transfer0 = Orbit.from_vectors(Sun, r0_km, v_dep)
    rE0 = (earth - sun).at(t0).position.km * u.km
    vE0 = (earth - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    rM0 = (mars  - sun).at(t0).position.km * u.km
    vM0 = (mars  - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    earth_orb0 = Orbit.from_vectors(Sun, rE0, vE0)
    mars_orb0  = Orbit.from_vectors(Sun, rM0, vM0)

    r_p = R_m + h_MOI
    r_a = APOAPSIS_RATIO * r_p
    a   = 0.5 * (r_p + r_a)
    v_esc_m     = np.sqrt(2.0 * mu_m / r_p)
    v_hyp_p     = np.sqrt(v_esc_m**2 + (v_inf_arr.to(u.m/u.s).value)**2)
    v_peri_ellip= np.sqrt(mu_m * (2.0/r_p - 1.0/a))
    v_apo_ellip = np.sqrt(mu_m * (2.0/r_a - 1.0/a))
    v_circ_m    = np.sqrt(mu_m / r_p)

    theta   = np.deg2rad(PLANE_CHANGE_DEG)

    entry_results = None
    if USE_AEROCAPTURE and AEROCAPTURE_PHYSICS:
        v_inf_arr_mps = v_inf_arr.to(u.m/u.s).value
        # Solve for FPA to aim for desired periapsis altitude during pass
        if ENTRY_FPA_SOLVE:
            entry_results = solve_entry_fpa_for_target(v_inf_arr_mps, PERIAPSIS_ALT_M,
                                                       fpa_bounds_deg=FPA_BOUNDS_DEG, tol_m=FPA_SOLVE_TOL_M)
        else:
            entry_results = simulate_entry_pass(v_inf_arr_mps, gamma0_deg=-5.0, rp_target_alt_m=PERIAPSIS_ALT_M)

        print("\n=== Aerocapture Pass (physics) ===")
        print(f"  gamma0:                 {entry_results['gamma0_deg']:.2f} deg")
        print(f"  min altitude:           {entry_results['min_alt_m']/1000:.1f} km")
        print(f"  Δv removed by atmosphere {entry_results['atmo_dv_mps']/1000:.3f} km/s")
        print(f"  max dynamic pressure q: {entry_results['max_q_Pa']/1000:.2f} kPa")
        print(f"  peak heat flux:         {entry_results['max_heat_Wm2']:.0f} W/m^2")
        print(f"  total heat load:        {entry_results['heat_load_Jm2']:.0f} J/m^2")
        print(f"  peak g-load:            {entry_results['max_g']:.2f} g")

        # Safety checks
        unsafe = []
        if entry_results['max_q_Pa'] > Q_LIMIT_PA:        unsafe.append("q")
        if entry_results['max_heat_Wm2'] > HEAT_LIMIT_WM2: unsafe.append("heat")
        if entry_results['max_g'] > G_LIMIT:               unsafe.append("g-load")
        if unsafe:
            print(f"  [WARN] Aerocapture exceeded limits: {', '.join(unsafe)}")

        # Atmosphere handled capture; set dv1 propulsive to 0
        dv1_mps = 0.0

    else:
        # placeholder energy bookkeeping (legacy)
        rp = R_m + PERIAPSIS_ALT_M
        v_inf = v_inf_arr.to(u.m/u.s).value
        v_p   = np.sqrt(v_inf**2 + 2*mu_m/rp)
        v_esc = np.sqrt(2*mu_m/rp)
        dv_atm_min = max(0.0, v_p - v_esc)
        v_peri_target = v_peri_ellip
        dv_atm_to_target = max(0.0, v_p - v_peri_target)
        print("\n=== Aerocapture Requirements (minimal-assumption) ===")
        print(f"Periapsis radius rp:          {rp/1000.0:,.1f} km (alt {PERIAPSIS_ALT_M/1000.0:.1f} km)")
        print(f"Δv_atm_min_to_capture:        {dv_atm_min/1000.0:.3f} km/s")
        print(f"Δv_atm_to_target_ellipse:     {dv_atm_to_target/1000.0:.3f} km/s  (to reach R≈{APOAPSIS_RATIO:.1f})")
        dv1_mps = 0.0  # keeping aerocapture assumption

    # Remaining MOI pieces (mars orbit inspemection (insertion))
    dv2_mps = 2.0 * v_apo_ellip * np.sin(theta / 2.0)
    dv3_mps = 0.0 if MINIMUM_ORBIT_MODE else abs(v_circ_m - v_peri_ellip)

    dv1_k = dv1_mps/1000.0; dv2_k = dv2_mps/1000.0; dv3_k = dv3_mps/1000.0
    dv_moi_total = dv1_k + dv2_k + dv3_k

    print(f"\n[MOI Planner] (Aerocapture={'True' if USE_AEROCAPTURE else 'False'} | Physics={'True' if AEROCAPTURE_PHYSICS else 'False'} | MinOrbit={MINIMUM_ORBIT_MODE})")
    print(f"  dv1 (capture at peri):      {dv1_k:.3f} km/s")
    print(f"  dv2 (plane change @ apo):   {dv2_k:.3f} km/s")
    print(f"  dv3 (circularize @ peri):   {dv3_k:.3f} km/s")
    print(f"Total Mars insertion Δv:      {dv_moi_total:.3f} km/s")

    # MONARC for MOI
    m0_moi = m_after_dep
    monarc_prop_full = m_sc_wet - m_sc_dry

    def print_passes(title, passes, prop_start, prop_end):
        print(f"\n[{title}] Perigee-pass burns:")
        for p in passes[:120]:
            prop_now = max(0.0, p['m_end'] - m_sc_dry)
            frac_left = (prop_now / monarc_prop_full * 100.0) if monarc_prop_full>0 else 0.0
            print(f"  pass {p['pass']:02d}: Δv={p['dv_kms']:.4f} km/s, time={p['time_min']:.2f} min, "
                  f"mass {p['m_start']:.1f}→{p['m_end']:.1f} kg (used {p['m_used']:.1f} kg) | "
                  f"MONARC prop left: {prop_now:.1f} kg ({frac_left:.1f}%)")
        if len(passes) > 120:
            print(f"  ... {len(passes)-120} more passes not shown")

    total_monarc_used = monarc_prop_used_dep

    # dv1 (propulsive) skipped under aerocapture physics 
    dv1_passes=[]; dv1_deliv=0.0; t1_min=0.0
    m_after1 = m0_moi

    # dv2 @ apo
    dv2_deliv=0.0; t2_min=0.0
    m_after2 = m_after1
    if dv2_k > 1e-9:
        dt_sec = 0.0
        v_acc  = 0.0
        m      = m_after1
        while v_acc < dv2_k*1000.0 and dt_sec < 3*3600:
            aacc = thrust_monarc / m
            dv = aacc * 1.0
            dm = m * (1.0 - np.exp(-dv / (isp_monarc * g0)))
            m -= dm
            v_acc += dv
            dt_sec += 1.0
        dv2_deliv = min(v_acc/1000.0, dv2_k)
        t2_min = dt_sec/60.0
        m_after2 = m
        used2 = (m_after1 - m_after2); total_monarc_used += used2
        print(f"\n[dv2 @ apo] Δv={dv2_deliv:.3f} km/s, time≈{t2_min:.2f} min, mass {m_after1:.1f}→{m_after2:.1f} kg (used {used2:.1f} kg)")
    else:
        print("\n[dv2 @ apo] Skipped (≈0).")

    # dv3 @ peri (if not minimum orbit mode)
    dv3_passes=[]; dv3_deliv=0.0; t3_min=0.0
    m_after3 = m_after2
    if dv3_k > 1e-9:
        dv3_passes, dv3_deliv, m_after3, t3_min = perigee_pass_burns_required(
            dv3_k, m_after2, isp_monarc, thrust_monarc, PER_PASS_TIME_MIN_MOI, MAX_PASSES_PER_PHASE, "dv3 circularize"
        )
        print_passes("dv3 (circularize at peri)", dv3_passes, m_after2, m_after3)
        used3 = (m_after2 - m_after3); total_monarc_used += used3
        print("\n[dv3 (circularize) Summary]")
        print(f"  Total Δv:     {dv3_deliv:.3f} km/s")
        print(f"  Total time:   {t3_min:.1f} min")
        print(f"  Prop used:    {used3:.1f} kg")
        print(f"  Passes:       {len(dv3_passes)}")
    else:
        print("\n[dv3 (circularize)] Skipped (Minimum-orbit mode).")

    monarc_prop_remaining = max(0.0, m_after3 - m_sc_dry)
    print("\n--- MONARC Propellant Usage (Total) ---")
    print(f"Prop total (design): {monarc_prop_full:.1f} kg")
    print(f"Prop used (actual):  {total_monarc_used:.1f} kg")
    print(f"Prop remaining:      {monarc_prop_remaining:.1f} kg ({100*monarc_prop_remaining/monarc_prop_full:.1f}% left)")

    monarc_dv_total = _g_dv_monarc_dep_kms + dv1_deliv + dv2_deliv + dv3_deliv
    opt_prop = rocket_eq_prop_for_dv(isp_monarc, m_sc_dry, monarc_dv_total*1000.0)
    opt_wet  = m_sc_dry + opt_prop
    potential_savings = max(0.0, monarc_prop_full - opt_prop)
    print("\n--- Optimal MONARC Prop Estimate ---")
    print(f"Total MONARC Δv delivered: {monarc_dv_total:.3f} km/s")
    print(f"Optimal prop to match Δv:  {opt_prop:.1f} kg  (wet={opt_wet:.1f} kg, dry={m_sc_dry:.1f} kg)")
    print(f"Potential prop savings:    {potential_savings:.1f} kg (vs design {monarc_prop_full:.1f} kg)")

    if MINIMUM_ORBIT_MODE:
        print("\n=== FINAL STATUS: Arrived at Mars ===")
        print(f"Final (captured) ellipse:")
        print(f"  rp = {r_p/1000.0:,.1f} km")
        print(f"  ra = {r_a/1000.0:,.1f} km")
        print(f"  a  = {(0.5*(r_p+r_a))/1000.0:,.1f} km, e = { (r_a-r_p)/(r_a+r_p):.6f}")
        Tm = 2*np.pi*np.sqrt((0.5*(r_p+r_a))**3/mu_m)
        print(f"  Period = {Tm/3600.0:,.2f} h")
        final_orbit = ("Mars", r_p, r_a)
    else:
        print("\n=== FINAL STATUS: Arrived at Mars ===")
        print(f"Final (circularized near rp):")
        print(f"  r  = {r_p/1000.0:,.1f} km (e≈0)")
        final_orbit = ("Mars", r_p, r_p)

    make_all_plots(depart_dt, arrival_dt, t0, tof_s, r0_km, v_dep, vE_kms, USE_AEROCAPTURE, True,
                   final_orbit=final_orbit,
                   dv_breakdown=(_g_dv_lv2_used_kms, _g_dv_star_used_kms, _g_dv_monarc_dep_kms),
                   entry_results=entry_results)

# ===================== PLOTS =====================
def make_all_plots(depart_dt, arrival_dt, t0, tof_s, r0_km, v_dep, vE_kms, use_aerocapture, reached_mars, final_orbit,
                   dv_breakdown=(0.0,0.0,0.0), entry_results=None):
    transfer0 = Orbit.from_vectors(Sun, r0_km, v_dep)
    rE0 = (earth - sun).at(t0).position.km * u.km
    vE0 = (earth - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    rM0 = (mars  - sun).at(t0).position.km * u.km
    vM0 = (mars  - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    earth_orb0 = Orbit.from_vectors(Sun, rE0, vE0)
    mars_orb0  = Orbit.from_vectors(Sun, rM0, vM0)

    N = 400
    tsamp = np.linspace(0.0, tof_s, N) * u.s
    r_sc = np.zeros((N,3)); v_sc = np.zeros((N,3))
    r_ea = np.zeros((N,3)); r_ma = np.zeros((N,3))
    nu_sc = np.zeros(N); speed_sc = np.zeros(N)

    for i, dt_ in enumerate(tsamp):
        o_sc = transfer0.propagate(dt_)
        r_sc[i,:] = o_sc.r.to_value(u.km)
        v_sc[i,:] = o_sc.v.to_value(u.km/u.s)
        nu_sc[i]  = o_sc.nu.to(u.rad).value
        speed_sc[i] = np.linalg.norm(v_sc[i,:])
        r_ea[i,:] = earth_orb0.propagate(dt_).r.to_value(u.km)
        r_ma[i,:] = mars_orb0 .propagate(dt_).r.to_value(u.km)

    d_sc = np.linalg.norm(r_sc, axis=1)
    d_ea = np.linalg.norm(r_ea, axis=1)
    d_ma = np.linalg.norm(r_ma, axis=1)
    t_days = (tsamp.to(u.day)).value

    # 3D heliocentric transfer
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_sc[:,0], r_sc[:,1], r_sc[:,2], label='Spacecraft transfer')
    ax.plot(r_ea[:,0], r_ea[:,1], r_ea[:,2], '--', alpha=0.6, label='Earth arc')
    ax.plot(r_ma[:,0], r_ma[:,1], r_ma[:,2], '--', alpha=0.6, label='Mars arc')
    ax.scatter(r_sc[0,0], r_sc[0,1], r_sc[0,2], s=40, label='Earth @ dep')
    ax.scatter(r_sc[-1,0], r_sc[-1,1], r_sc[-1,2], s=40, label='Mars @ arr')
    ax.scatter(0,0,0, s=80, label='Sun')
    ax.set_title('Heliocentric Transfer (Conic Propagation)')
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_zlabel('z (km)')
    ax.legend(); plt.tight_layout(); _savefig("heliocentric_3d")

    # 2D XY heliocentric
    plt.figure(figsize=(8,6))
    plt.plot(r_sc[:,0], r_sc[:,1], label='Transfer (conic)')
    plt.plot(r_ea[:,0], r_ea[:,1], '--', alpha=0.6, label='Earth arc')
    plt.plot(r_ma[:,0], r_ma[:,1], '--', alpha=0.6, label='Mars arc')
    plt.scatter(0,0, s=80, label='Sun')
    plt.axis('equal'); plt.grid(True); plt.legend(); plt.title('XY Heliocentric View (Conics)')
    plt.tight_layout(); _savefig("heliocentric_xy")

    # Distance vs time
    plt.figure()
    plt.plot(t_days, d_ea/1e6, label='Earth')
    plt.plot(t_days, d_sc/1e6, label='Spacecraft')
    plt.plot(t_days, d_ma/1e6, label='Mars')
    plt.xlabel('Time since departure (days)'); plt.ylabel('Distance from Sun (10^6 km)')
    plt.title('Heliocentric Distance vs Time')
    plt.grid(True); plt.legend(); plt.tight_layout(); _savefig("heliocentric_distance")

    # True anomaly vs time
    plt.figure()
    nu_deg = np.degrees((nu_sc + 2*np.pi) % (2*np.pi))
    plt.plot(t_days, nu_deg)
    plt.xlabel('Time since departure (days)'); plt.ylabel('True anomaly (deg)')
    plt.title('Spacecraft True Anomaly vs Time')
    plt.grid(True); plt.tight_layout(); _savefig("heliocentric_true_anomaly")

    # Speed vs time
    plt.figure()
    plt.plot(t_days, speed_sc)
    plt.xlabel('Time since departure (days)'); plt.ylabel('Speed (km/s)')
    plt.title('Spacecraft Speed vs Time (Heliocentric)')
    plt.grid(True); plt.tight_layout(); _savefig("heliocentric_speed")

    # Δv breakdown (departure)
    plt.figure()
    bars = ['LV2 leftover', 'STAR', 'MONARC (dep)']
    vals = [dv_breakdown[0], dv_breakdown[1], dv_breakdown[2]]
    plt.bar(bars, vals)
    plt.ylabel('Δv (km/s)'); plt.title('Δv Breakdown — Departure Contributions')
    plt.tight_layout(); _savefig("dv_breakdown")

    # Final orbit plot (Mars or Earth, hopefully mars)
    body, rp, ra = final_orbit
    plt.figure()
    if body == "Mars":
        R = R_m
        title = "Final Orbit around Mars"
    else:
        R = R_e
        title = "Final Orbit around Earth"
    a = 0.5*(rp+ra)
    e = (ra - rp)/(ra + rp + 1e-12)
    th = np.linspace(0, 2*np.pi, 400)
    r = (a*(1-e**2)) / (1 + e*np.cos(th))
    x = r*np.cos(th)/1000.0; y = r*np.sin(th)/1000.0
    plt.plot(x, y, label='Final orbit')
    circ = plt.Circle((0,0), R/1000.0, color='orange', alpha=0.3, label=f'{body} radius')
    ax2 = plt.gca(); ax2.add_patch(circ)
    plt.axis('equal'); plt.grid(True); plt.legend(); plt.title(title)
    plt.xlabel('x (km)'); plt.ylabel('y (km)')
    plt.tight_layout(); _savefig("final_orbit")

    # Entry plots
    if entry_results is not None and len(entry_results["time_s"]) > 0:
        tmin = entry_results["time_s"]/60.0
        plt.figure(); plt.plot(tmin, entry_results["alt_m"]/1000.0)
        plt.xlabel('Time (min)'); plt.ylabel('Altitude (km)')
        plt.title('Aerocapture Altitude vs Time'); plt.grid(True); _savefig("entry_altitude")

        plt.figure(); plt.plot(tmin, entry_results["v_mps"]/1000.0)
        plt.xlabel('Time (min)'); plt.ylabel('Speed (km/s)')
        plt.title('Aerocapture Speed vs Time'); plt.grid(True); _savefig("entry_speed")

        plt.figure(); plt.plot(tmin, entry_results["q_Pa"]/1000.0)
        plt.xlabel('Time (min)'); plt.ylabel('Dynamic Pressure q (kPa)')
        plt.title('Dynamic Pressure vs Time'); plt.grid(True); _savefig("entry_q")

        plt.figure(); plt.plot(tmin, entry_results["heat_Wm2"])
        plt.xlabel('Time (min)'); plt.ylabel('Stagnation Heat Flux (W/m²)')
        plt.title('Heat Flux vs Time'); plt.grid(True); _savefig("entry_heat")

        plt.figure(); plt.plot(tmin, entry_results["g_load"])
        plt.xlabel('Time (min)'); plt.ylabel('g-load')
        plt.title('Deceleration vs Time'); plt.grid(True); _savefig("entry_g")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')

if __name__ == "__main__":
    main()
