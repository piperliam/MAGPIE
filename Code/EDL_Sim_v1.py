"""
MAGPIE Mars EDL / Landing Simulator (v2.0k (this has had a few revisions + debug + chute corridor + staging snapshots)

Author: Liam Piper
2025
"""

import math
import json
import csv
import os
import matplotlib.pyplot as plt

##############################################
# ========= USER CONFIG / CONSTANTS =========
##############################################

DT                      = 0.1        # [s] integrator step
MAX_TIME_S              = 20000.0    # cutoff safety

WRITE_TELEMETRY_CSV     = True
WRITE_EVENTS_CSV        = True
WRITE_SUMMARY_JSON      = True

TELEM_CSV_PATH          = "telemetry.csv"
EVENTS_CSV_PATH         = "events.csv"
SUMMARY_JSON_PATH       = "summary.json"

PRINT_DEBUG             = True

# Plot controls
PLOT_MODE   = "save"    # "show", "save", "none"
SAVE_DIR    = "."
SAVE_PREFIX = "edl_"

SHOW_PLOTS  = (PLOT_MODE.lower() == "show")
SAVE_PLOTS  = (PLOT_MODE.lower() == "save")

# --- Planet model (Mars only) ---
MARS_MU         = 4.282837e13     # [m^3/s^2]
MARS_RADIUS     = 3389500.0       # [m]
MARS_G0         = 3.711           # [m/s^2] near surface
MARS_RHO0       = 0.020           # [kg/m^3]
MARS_HSCALE     = 11100.0         # [m]
MARS_GAMMA      = 1.29            # [-] cp/cv guess for mach calc
MARS_A0         = 240.0           # [m/s] approx sound speed in lower Mars atm

# Dust storm / density scaling
DUST_STORM_ENABLE   = False       # global "thicken atm" hack - not ideal but it is good to have as a test
DUST_RHO_SCALE      = 1.5         # multiplier on density if storm is on

# --- Entry targeting / orbit geometry ---
START_ALT_CIRC_M       = 250_000.0    # [m] start circular orbit alt
TARGET_PERIAPSIS_ALT_M = 35_000.0     # [m] periapsis alt after deorbit
ENTRY_INTERFACE_ALT_M  = 125_000.0    # [m] EI def

# --- Mass / stage bookkeeping ---
BUS_DRY_KG              = 22.9    # kg bus dry (lands)
BUS_TERMINAL_PROP_KG    = 0.0     # kg prop available for terminal retro
MAIN_CHUTE_MASS_KG      = 4.2     # kg chute mass carried to ground
HONEYCOMB_PAD_MASS_KG   = 6.0       # kg crush pad mass carried to ground

ABLATOR_MASS_KG         = 0.5     # kg ablator in aeroshell
AEROSHELL_MASS_KG       = 12.2    # kg backshell + structure etc
DEORBIT_PACK_DRY_KG     = 0.0     # kg retro pack (discarded pre-EI)
DEORBIT_PROP_MASS_KG    = 5.0     # kg prop (consumed pre-EI)

# --- Terminal retro pack / terminal DV ---
TERMINAL_RETRO_ENABLE       = True

G12_COUNT                   = 0
G12_TOTAL_IMPULSE_NS        = 144.0      # per motor
G12_PROPELLANT_KG           = 0.082
G12_SCALE_TO_IMPULSE        = True

RETRO_MODE                  = "at_altitude"  # "at_altitude" or "after_chute_full"
RETRO_TRIGGER_ALT_M         = 80.0           # [m] if mode == "at_altitude"
ADVANCED_THRUST_ORIENTATION = False          # (placeholder, no vectoring yet)

TERMINAL_REVERSE_DV_MPS     = 20.0           # m/s vertical impulse when firing
TERMINAL_PROP_USED_ON_FIRE  = 0.25           # kg prop spent when we fire

# --- Aero / drag geometry ---
AEROSHELL_CD        = 1.8513620
AEROSHELL_AREA_M2   = 0.785294
PROBE_CD            = 2.363940
PROBE_AREA_M2       = 0.57145

# --- Drogue chute model ---
DROGUE_ENABLE           = True
DROGUE_DEPLOY_MACH      = 1.95
DROGUE_AREA_M2          = 0.073
DROGUE_CD               = 0.97
DROGUE_CUT_AT_JETTISON  = True

# --- Main chute reefing model ---
PARA_AREA_M2        = 42.0 #23.7
PARA_CD             = 2.2
PARA_SPILL_HOLE_FR  = 0.05
PARA_LINE_LENGTH_M  = 8.0

PARA_DEPLOY_ALT_M   = 10_000.0

# Main chute deployment corridor in Mach-q space.
# Widened to actually intersect the current trajectory (subsonic, low-q)
# so we can see the chute behavior in this test case.
MACH_MIN, MACH_MAX  = 0.3, 2.0
Q_MIN_PA, Q_MAX_PA  = 50.0, 2000.0

REEF1_TARGET_F      = 0.35
REEF2_TARGET_F      = 0.70
FULL_TARGET_F       = 1.00

TAU_LINE_STRETCH_S  = 0.78
TAU_REEF1_S         = 1.6
TAU_REEF2_S         = 2.0
TAU_FULL_S          = 3.5

Q_REEF1_TO_2_MAX     = 900.0
Q_REEF2_TO_FULL_MAX  = 700.0

MAX_RISER_FORCE_N   = 25500.0
SNATCH_OVERSHOOT    = 1.25
ALLOW_CUTAWAY       = False

# Hard safety: force main chute below this altitude even if Mach/q corridor is missed.
# Set to None to disable this backup.
FORCE_MAIN_DEPLOY_BELOW_ALT_M = 9000.0  # [m]

# survivability checks
SURVIVABLE_PEAK_G_LIMIT         = 300.0   # payload limit in g at impact
SURVIVABLE_RISER_FACTOR         = 1.0     # scaling chute limit
SURVIVABLE_ENTRY_PEAK_G_LIMIT   = 50.0    # [g] allowable peak aero/chute g during EDL (tunable)

# touchdown vertical velocity requirement
MAX_SAFE_TOUCHDOWN_VVERT_MPS = 15.0  # [m/s]

# --- Thermal / ablation ---
NOSE_RADIUS_M              = 0.40
EMISSIVITY                 = 0.8
SIGMA                      = 5.670374419e-8
K_SUTTON                   = 1.1e-4
PICA_LATENT                = 2.5e6
ABLATION_THRESHOLD_Wm2     = 3.5e5
MAX_TEMP_K                 = 2500.0
Cp_STRUCT                  = 900.0

Q_JETTISON_MAX_PA          = 3000.0
QDOT_JETTISON_MAX_Wcm2     = 5.0
JETTISON_MIN_ALT_M         = 25_000.0

# --- Honeycomb crush core model (INPUT FROM HONEYCOMB SCRIPT)
#
# Paste here from hex_v1.2a.py console block:
#   [SIM CONSTANTS -> paste into EDL sim]
#
# Meaning of each:
#  HONEYCOMB_REL_DENSITY      = effective ρ*/ρ_s across both inner+rim zones
#  HONEYCOMB_BASE_MATERIAL_SIGY_MPa = base aluminum yield in MPa
#  HONEYCOMB_PLATEAU_FACTOR   = PF_eff (effective plateau coefficient)
#  HONEYCOMB_FOOTPRINT_M2     = actual contact area (pi R_total^2)
#  HONEYCOMB_STROKE_M         = crush stroke height

HONEYCOMB_REL_DENSITY      = 0.047828    # Density of crushed hex
HONEYCOMB_BASE_MATERIAL_SIGY_MPa = 120.0 # MPa
HONEYCOMB_PLATEAU_FACTOR   = 0.014249    # PF_eff
HONEYCOMB_FOOTPRINT_M2     = 0.502700    # m^2
HONEYCOMB_STROKE_M         = 0.050000    # m

##############################################
# ========= HELPER FUNCTIONS ===============
##############################################

def mars_gravity(alt_m):
    r = MARS_RADIUS + alt_m
    return MARS_MU / (r * r)

def mars_density(alt_m):
    if alt_m < 0.0:
        alt_m = 0.0
    scale = DUST_RHO_SCALE if DUST_STORM_ENABLE else 1.0
    return scale * MARS_RHO0 * math.exp(-alt_m / MARS_HSCALE)

def dyn_pressure(rho, vmag):
    return 0.5 * rho * vmag * vmag

def aero_drag_force(Cd, area, rho, vmag):
    return Cd * area * dyn_pressure(rho, vmag)

def est_mach(vmag):
    if MARS_A0 <= 0:
        return 0.0
    return vmag / MARS_A0

def convective_heat_flux(rho, vmag):
    # Sutton-Graves style qdot ~ k * sqrt(rho/Rn) * v^3
    r_eff = NOSE_RADIUS_M if NOSE_RADIUS_M > 0 else 0.1
    q = K_SUTTON * math.sqrt(max(rho,1e-9)/r_eff) * (vmag**3)
    return max(q, 0.0)

def ablation_rate(qdot_Wm2):
    if qdot_Wm2 < ABLATION_THRESHOLD_Wm2:
        return 0.0
    return qdot_Wm2 / PICA_LATENT  # kg/s/m^2

def init_main_chute_state():
    return {
        "deployed": False,
        "cutaway": False,
        "t_deploy": 0.0,
        "stage": "stowed",
        "inflation_frac": 0.0,
        "riser_force_N": 0.0,
        "riser_failed": False,
        "mach_at_deploy": None,
        "q_at_deploy": None
    }

def init_drogue_state():
    return {
        "active": False,
        "cut":   False
    }

def update_reefing(main_chute, q_pa, mach, alt_m, t):
    # deploy trigger
    if not main_chute["deployed"]:
        # corridor check in Mach–q space
        corridor_ok = (
            mach >= MACH_MIN and mach <= MACH_MAX and
            q_pa >= Q_MIN_PA and q_pa <= Q_MAX_PA
        )

        # hard safety: if we never got a nice corridor but we've fallen below some
        # altitude, just pop the chute anyway rather than lawn-darting.
        hard_trigger = (
            FORCE_MAIN_DEPLOY_BELOW_ALT_M is not None and
            alt_m <= FORCE_MAIN_DEPLOY_BELOW_ALT_M
        )

        if alt_m <= PARA_DEPLOY_ALT_M and (corridor_ok or hard_trigger):
            main_chute["deployed"] = True
            main_chute["t_deploy"] = t
            main_chute["stage"]   = "line_stretch"
            main_chute["inflation_frac"] = 0.0
            main_chute["cutaway"] = False
            main_chute["mach_at_deploy"] = mach
            main_chute["q_at_deploy"]    = q_pa

    if not main_chute["deployed"]:
        # keep loads 0 so we don't falsely trigger failures
        main_chute["riser_force_N"] = 0.0
        return main_chute

    tau = t - main_chute["t_deploy"]

    # stage progression
    if main_chute["stage"] == "line_stretch":
        if tau >= TAU_LINE_STRETCH_S:
            main_chute["stage"] = "reef1"

    elif main_chute["stage"] == "reef1":
        frac_target = REEF1_TARGET_F
        if tau >= TAU_LINE_STRETCH_S + TAU_REEF1_S and q_pa <= Q_REEF1_TO_2_MAX:
            main_chute["stage"] = "reef2"
        main_chute["inflation_frac"] = frac_target

    elif main_chute["stage"] == "reef2":
        frac_target = REEF2_TARGET_F
        if tau >= TAU_LINE_STRETCH_S + TAU_REEF1_S + TAU_REEF2_S and q_pa <= Q_REEF2_TO_FULL_MAX:
            main_chute["stage"] = "full"
        main_chute["inflation_frac"] = frac_target

    elif main_chute["stage"] == "full":
        post_full_tau = tau - (TAU_LINE_STRETCH_S + TAU_REEF1_S + TAU_REEF2_S)
        if tau >= TAU_LINE_STRETCH_S + TAU_REEF1_S + TAU_REEF2_S + TAU_FULL_S:
            main_chute["inflation_frac"] = FULL_TARGET_F
        else:
            frac = REEF2_TARGET_F + (FULL_TARGET_F-REEF2_TARGET_F)*min(max(post_full_tau/TAU_FULL_S,0.0),1.0)
            main_chute["inflation_frac"] = min(frac, FULL_TARGET_F)

    # chute load calc with snatch overshoot
    effective_area = PARA_AREA_M2 * main_chute["inflation_frac"] * (1.0 - PARA_SPILL_HOLE_FR)
    chute_drag_N   = PARA_CD * effective_area * q_pa
    chute_drag_N  *= SNATCH_OVERSHOOT

    main_chute["riser_force_N"] = chute_drag_N

    # overload check
    if chute_drag_N > MAX_RISER_FORCE_N * SURVIVABLE_RISER_FACTOR:
        main_chute["riser_failed"] = True
        if ALLOW_CUTAWAY:
            main_chute["cutaway"] = True
            main_chute["inflation_frac"] = 0.0

    return main_chute

def chute_drag(main_chute, rho, vvx, vvy):
    if (not main_chute["deployed"]) or main_chute["cutaway"] or main_chute["inflation_frac"] <= 1e-6:
        return (0.0, 0.0)
    vmag = math.hypot(vvx, vvy)
    if vmag < 1e-5:
        return (0.0,0.0)
    q_pa = dyn_pressure(rho, vmag)
    A_eff = PARA_AREA_M2 * main_chute["inflation_frac"] * (1.0 - PARA_SPILL_HOLE_FR)
    Fmag = PARA_CD * A_eff * q_pa
    Fx = -Fmag * (vvx/vmag)
    Fy = -Fmag * (vvy/vmag)
    return (Fx, Fy)

def drogue_drag(drogue, rho, vx, vy):
    if not drogue["active"]:
        return (0.0, 0.0)
    vmag = math.hypot(vx, vy)
    if vmag < 1e-5:
        return (0.0, 0.0)
    q_pa = dyn_pressure(rho, vmag)
    Fmag = DROGUE_CD * DROGUE_AREA_M2 * q_pa
    Fx   = -Fmag * (vx/vmag)
    Fy   = -Fmag * (vy/vmag)
    return (Fx, Fy)

def compute_required_deorbit_dv_and_prop(m0_total, isp_s):
    """
    Estimate DV to go from circular at START_ALT_CIRC_M down to a periapsis TARGET_PERIAPSIS_ALT_M.
    Then estimate prop used with a simple rocket equation.
    """
    r0 = MARS_RADIUS + START_ALT_CIRC_M
    rp_target = MARS_RADIUS + TARGET_PERIAPSIS_ALT_M
    v_circ = math.sqrt(MARS_MU / r0)

    dv_lo = 0.0
    dv_hi = 300.0
    for _ in range(60):
        dv_mid = 0.5*(dv_lo+dv_hi)
        v_post = v_circ - dv_mid
        eps = v_post**2/2.0 - MARS_MU/r0
        h   = r0 * v_post
        e   = math.sqrt(1.0 + 2.0*eps*(h**2)/(MARS_MU**2))
        rp  = (h**2)/(MARS_MU*(1.0+e))
        if rp <= rp_target:
            dv_hi = dv_mid
        else:
            dv_lo = dv_mid
    required_dv = 0.5*(dv_lo+dv_hi)

    # rocket equation
    g0 = 9.80665
    if isp_s <= 0:
        prop_used = 0.0
    else:
        m1 = m0_total / math.exp(required_dv/(isp_s*g0))
        prop_used = max(m0_total - m1, 0.0)
    return required_dv, prop_used

def honeycomb_model(lander_mass_kg, v_impact_mps, g_limit_g=None):
    """
    Compute crush response using EFFECTIVE honeycomb constants from the generator.

    Two bounding g's:
      g_stiffness = plateau_force / (m*g0)
      g_energy_limit = (KE / stroke) / (m*g0)

    The physically governing peak g is the LARGER of those two
    ("stiffness" if pad is too stiff, "stroke" if you bottom the stroke).

    Returns dict with:
      peak_g (alias of g_effective)
      g_stiffness
      g_energy_limit
      governing_mode
      required_plateau_stress_for_energy_Pa
      etc.
    """
    # current plateau stress
    plateau_stress_pa = (
        HONEYCOMB_PLATEAU_FACTOR *
        HONEYCOMB_REL_DENSITY *
        (HONEYCOMB_BASE_MATERIAL_SIGY_MPa * 1.0e6)
    )
    plateau_force_N   = plateau_stress_pa * HONEYCOMB_FOOTPRINT_M2

    KE_in_J = 0.5 * lander_mass_kg * (v_impact_mps**2)

    # energy capacity at *this* plateau force over full stroke
    energy_capacity_J = plateau_force_N * HONEYCOMB_STROKE_M
    absorbed_all      = (KE_in_J <= energy_capacity_J + 1e-9)

    g0 = 9.80665
    if lander_mass_kg > 0.0:
        g_stiffness = plateau_force_N / (lander_mass_kg * g0)
    else:
        g_stiffness = 0.0

    # stroke required if plateau_force is fixed:
    if plateau_force_N > 1e-9:
        required_stroke_m = KE_in_J / plateau_force_N
    else:
        required_stroke_m = None

    # minimum constant force needed *just* to absorb KE in available stroke
    if HONEYCOMB_STROKE_M > 1e-9:
        F_required_energy_N = KE_in_J / HONEYCOMB_STROKE_M
    else:
        F_required_energy_N = float('inf')

    # that force as stress
    if HONEYCOMB_FOOTPRINT_M2 > 1e-12:
        required_plateau_stress_for_energy_Pa = F_required_energy_N / HONEYCOMB_FOOTPRINT_M2
    else:
        required_plateau_stress_for_energy_Pa = float('inf')

    # convert F_required_energy_N into g's:
    if lander_mass_kg > 0.0:
        g_energy_limit = F_required_energy_N / (lander_mass_kg * g0)
    else:
        g_energy_limit = 0.0

    # governing
    if g_stiffness >= g_energy_limit:
        g_effective = g_stiffness
        governing_mode = "stiffness"
    else:
        g_effective = g_energy_limit
        governing_mode = "stroke"

    # allowable stress to meet limit_g
    required_plateau_stress_for_limit = None
    if g_limit_g is not None and lander_mass_kg > 0 and HONEYCOMB_FOOTPRINT_M2 > 0:
        F_allow = g_limit_g * lander_mass_kg * g0
        stress_allow = F_allow / HONEYCOMB_FOOTPRINT_M2
        required_plateau_stress_for_limit = stress_allow  # Pa

    return {
        "plateau_stress_Pa": plateau_stress_pa,
        "plateau_force_N": plateau_force_N,
        "energy_capacity_J": energy_capacity_J,
        "impact_ke_J": KE_in_J,
        "absorbed_all": absorbed_all,
        "required_stroke_m": required_stroke_m,

        "F_required_energy_N": F_required_energy_N,
        "required_plateau_stress_for_energy_Pa": required_plateau_stress_for_energy_Pa,
        "g_stiffness": g_stiffness,
        "g_energy_limit": g_energy_limit,
        "g_effective": g_effective,
        "governing_mode": governing_mode,

        "peak_g": g_effective,
        "required_plateau_stress_for_limit_Pa": required_plateau_stress_for_limit
    }

##############################################
# ========= PLOTTING HELPERS ================
##############################################

def _savefig_stub(stub):
    os.makedirs(SAVE_DIR, exist_ok=True)
    outpath = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}{stub}.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")

def make_and_optionally_save_plot(
    x, y,
    xlabel, ylabel,
    title,
    filename_stub,
    extra_lines=None,
    scatter_points=None
):
    """
    Generic time series / XY plot helper.
    """
    if not (SHOW_PLOTS or SAVE_PLOTS):
        return
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if extra_lines:
        for ln in extra_lines:
            style = ln.get("style", {})
            if ln.get("vertical", False):
                plt.axvline(ln["x"], **style)
                if "label" in ln:
                    plt.text(ln["x"], plt.ylim()[1]*0.9, ln["label"],
                             rotation=90, va="top", ha="right", fontsize=8)
            else:
                plt.axhline(ln["y"], **style)
                if "label" in ln:
                    plt.text(plt.xlim()[1]*0.7, ln["y"], ln["label"],
                             va="bottom", ha="left", fontsize=8)

    if scatter_points:
        for pt in scatter_points:
            style = pt.get("style", {})
            plt.scatter(pt["x"], pt["y"], **style)
            if "label" in pt:
                plt.text(pt["x"], pt["y"], " "+pt["label"], fontsize=8,
                         va="bottom", ha="left")

    if SAVE_PLOTS:
        _savefig_stub(filename_stub)

def make_altitude_phase_plot(
    alt_list,
    val_list,
    xlabel,
    ylabel,
    title,
    filename_stub,
    annotations=None,
    extra_lines=None
):
    """
    Plot val vs altitude. Y-axis is normal: 0 m at bottom, higher altitudes up.
    """
    if not (SHOW_PLOTS or SAVE_PLOTS):
        return
    plt.figure()
    plt.plot(val_list, alt_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if annotations:
        for ann in annotations:
            plt.scatter(ann["val"], ann["alt"], marker="x")
            plt.text(
                ann["val"], ann["alt"], " " + ann["text"],
                fontsize=8, va="bottom", ha="left"
            )

    if extra_lines:
        for ln in extra_lines:
            style = ln.get("style", {})
            if ln.get("horizontal_alt", False):
                plt.axhline(ln["alt"], **style)
                if "label" in ln:
                    plt.text(
                        plt.xlim()[1]*0.7, ln["alt"], ln["label"],
                        va="bottom", ha="left", fontsize=8
                    )
            elif ln.get("vertical_val", False):
                plt.axvline(ln["val"], **style)
                if "label" in ln:
                    plt.text(
                        ln["val"], plt.ylim()[1]*0.9, ln["label"],
                        rotation=90, va="top", ha="right", fontsize=8
                    )

    # NOTE: We no longer invert altitude; 0 m is at the bottom (more intuitive).
    # plt.gca().invert_yaxis()

    if SAVE_PLOTS:
        _savefig_stub(filename_stub)

##############################################
# ========= MAIN SIM FUNCTION ===============
##############################################

def run_edl_sim():
    events = []  # list[(t, event_code, detail_str)]
    telemetry_rows = []

    # tracking
    first_bus_exposure = None
    peak_g_entry_info  = {"g":0.0, "t":0.0, "alt":0.0}
    peak_bus_temp_after_exposure = None
    peak_riser_force = 0.0

    # aerodynamic staging snapshots
    aero_staging = {
        "AEROSHELL_JETTISON": None,
        "DROGUE_DEPLOY": None,
        "MAIN_CHUTE_DEPLOY": None
    }

    # where did we first cross the main chute deploy altitude?
    first_below_para_alt_snapshot = None

    # traces for plots
    alt_trace      = []
    vvert_trace    = []
    q_trace        = []
    riser_trace    = []
    riser_t        = []

    # For extra diagnostic plots
    downrange_trace = []
    mach_trace      = []
    shield_T_trace  = []
    bus_T_trace     = []
    mass_trace      = []
    ay_trace        = []  # vertical accel (including gravity)
    chute_frac_trace= []
    heatflux_trace  = []

    # low-alt velocity checkpoints
    v_at_100m = None
    v_at_50m  = None
    v_at_10m  = None

    # chute-only descent rate before retro
    pre_retro_vdesc_samples = []

    # compute mass blocks
    terminal_prop_initial = BUS_TERMINAL_PROP_KG if TERMINAL_RETRO_ENABLE else 0.0

    m_bus_block = (
        BUS_DRY_KG
        + terminal_prop_initial
        + MAIN_CHUTE_MASS_KG
        + HONEYCOMB_PAD_MASS_KG
    )

    m_aeroshell_block = AEROSHELL_MASS_KG + ABLATOR_MASS_KG
    m_deorbit_block   = DEORBIT_PACK_DRY_KG + DEORBIT_PROP_MASS_KG
    m0_total          = m_bus_block + m_aeroshell_block + m_deorbit_block

    # Deorbit DV + prop usage (assume cold gas ~57s Isp)
    required_dv_mps, prop_used_kg = compute_required_deorbit_dv_and_prop(
        m0_total,
        isp_s=57.0
    )

    # Mass at EI: after deorbit prop is burned and pack is dropped
    mass_at_EI = m_bus_block + m_aeroshell_block

    # Ballistic coefficients (β = m / (Cd * A))
    beta_EI_kg_m2 = mass_at_EI / (AEROSHELL_CD * AEROSHELL_AREA_M2)
    landed_mass_nominal = (
        BUS_DRY_KG + terminal_prop_initial + MAIN_CHUTE_MASS_KG + HONEYCOMB_PAD_MASS_KG
    )
    beta_lander_kg_m2 = landed_mass_nominal / (PROBE_CD * PROBE_AREA_M2)

    # Orbital speed at EI for that ellipse:
    r_ei   = MARS_RADIUS + ENTRY_INTERFACE_ALT_M
    r0     = MARS_RADIUS + START_ALT_CIRC_M
    rp     = MARS_RADIUS + TARGET_PERIAPSIS_ALT_M
    a_semi = 0.5*(r0+rp)
    v_ei   = math.sqrt(MARS_MU*(2.0/r_ei - 1.0/a_semi))

    # Initial conditions at EI
    t   = 0.0
    x   = 0.0
    y   = ENTRY_INTERFACE_ALT_M
    vx  = v_ei
    vy  = 0.0

    aeroshell_attached = True
    ablator_mass       = ABLATOR_MASS_KG
    shield_temp_K      = 300.0
    bus_temp_K         = 300.0

    drogue             = init_drogue_state()
    main_chute         = init_main_chute_state()

    chute_deploy_time  = None
    aerojet_jettison_t = None
    touchdown_time     = None

    t_terminal_fire    = None
    used_terminal_prop = False
    tps_burnthrough    = False
    chute_failed       = False
    landing_destroyed  = False

    peak_heat_Wcm2     = 0.0
    peak_q_pa          = 0.0
    max_q_pa_time_alt  = (0.0,0.0)

    # integrated convective heat load per unit area
    cumulative_heat_Jm2 = 0.0

    # SIM LOOP
    while t < MAX_TIME_S:

        rho   = mars_density(y)
        vmag  = math.hypot(vx, vy)
        mach  = est_mach(vmag)
        q_pa  = dyn_pressure(rho, vmag)
        g_mag = mars_gravity(y)

        # record first time we pass below main chute deploy altitude
        if first_below_para_alt_snapshot is None and y <= PARA_DEPLOY_ALT_M:
            first_below_para_alt_snapshot = {
                "t": t,
                "alt_m": y,
                "mach": mach,
                "q_pa": q_pa
            }

        # convective heating
        qdot_Wm2   = convective_heat_flux(rho, vmag)
        qdot_Wcm2  = qdot_Wm2/1e4
        if qdot_Wcm2 > peak_heat_Wcm2:
            peak_heat_Wcm2 = qdot_Wcm2

        # integrate heat load per m^2 over time
        cumulative_heat_Jm2 += qdot_Wm2 * DT

        # ablation / TPS heating
        if aeroshell_attached and ablator_mass > 0.0:
            mdot_per_m2 = ablation_rate(qdot_Wm2)
            mdot_total  = mdot_per_m2 * AEROSHELL_AREA_M2
            dmass       = mdot_total * DT
            dmass       = min(dmass, ablator_mass)
            ablator_mass -= dmass

            shield_temp_K += (qdot_Wm2 * DT)/(Cp_STRUCT * (AEROSHELL_MASS_KG+1e-6))
            shield_temp_K  = min(shield_temp_K, MAX_TEMP_K)
        else:
            bus_temp_K += (qdot_Wm2 * DT)/(Cp_STRUCT * (BUS_DRY_KG+1e-6))
            bus_temp_K  = min(bus_temp_K, MAX_TEMP_K)

        # TPS burnthrough check
        if aeroshell_attached and (ablator_mass <= 1e-6) and (qdot_Wm2 >= ABLATION_THRESHOLD_Wm2):
            tps_burnthrough = True
            events.append((t, "LOSS",
                           f"TPS burn-through at alt={y:.1f} m, q={q_pa:.1f} Pa, heat={qdot_Wcm2:.2f} W/cm^2"))
            break

        # Drogue deploy
        if DROGUE_ENABLE and aeroshell_attached and (not drogue["active"]):
            if mach < DROGUE_DEPLOY_MACH:
                drogue["active"] = True
                events.append((t,"DROGUE_DEPLOY",
                               f"t={t:.1f}s alt={y:.1f} m M={mach:.2f} q={q_pa:.1f} Pa"))
                if aero_staging["DROGUE_DEPLOY"] is None:
                    aero_staging["DROGUE_DEPLOY"] = {
                        "t": t,
                        "alt_m": y,
                        "mach": mach,
                        "q_pa": q_pa,
                        "heat_Wcm2": qdot_Wcm2,
                        "shield_temp_K": shield_temp_K,
                        "bus_temp_K": bus_temp_K
                    }

        # Main chute reefing / load tracking
        main_chute = update_reefing(main_chute, q_pa, mach, y, t)

        if main_chute["deployed"] and chute_deploy_time is None:
            chute_deploy_time = t
            events.append((t,"MAIN_CHUTE_DEPLOY",
                           f"t={t:.1f}s alt={y:.1f} m M={mach:.2f} q={q_pa:.1f} Pa"))
            aero_staging["MAIN_CHUTE_DEPLOY"] = {
                "t": t,
                "alt_m": y,
                "mach": mach,
                "q_pa": q_pa,
                "heat_Wcm2": qdot_Wcm2,
                "shield_temp_K": shield_temp_K,
                "bus_temp_K": bus_temp_K
            }

        # peak riser load tracking
        if main_chute["riser_force_N"] > peak_riser_force:
            peak_riser_force = main_chute["riser_force_N"]

        # Aeroshell jettison check
        can_jettison = (
            aeroshell_attached and
            (y <= JETTISON_MIN_ALT_M) and
            (q_pa <= Q_JETTISON_MAX_PA) and
            (qdot_Wcm2 <= QDOT_JETTISON_MAX_Wcm2)
        )
        if can_jettison:
            aeroshell_attached = False
            aerojet_jettison_t = t
            if DROGUE_CUT_AT_JETTISON:
                drogue["active"] = False
                drogue["cut"]    = True
            events.append((t,"AEROSHELL_JETTISON",
                           f"t={t:.1f}s alt={y:.1f} m q={q_pa:.1f} Pa heat={qdot_Wcm2:.2f} W/cm^2"))

            if aero_staging["AEROSHELL_JETTISON"] is None:
                aero_staging["AEROSHELL_JETTISON"] = {
                    "t": t,
                    "alt_m": y,
                    "mach": mach,
                    "q_pa": q_pa,
                    "heat_Wcm2": qdot_Wcm2,
                    "shield_temp_K": shield_temp_K,
                    "bus_temp_K": bus_temp_K
                }

            # record bus first exposure snapshot
            if first_bus_exposure is None:
                first_bus_exposure = {
                    "t": t,
                    "alt_m": y,
                    "q_pa": q_pa,
                    "heat_Wcm2": qdot_Wcm2,
                    "shield_temp_K": shield_temp_K,
                    "bus_temp_K": bus_temp_K
                }
                peak_bus_temp_after_exposure = bus_temp_K

        # update post-exposure peak bus temp
        if first_bus_exposure is not None:
            if (peak_bus_temp_after_exposure is None) or (bus_temp_K > peak_bus_temp_after_exposure):
                peak_bus_temp_after_exposure = bus_temp_K

        # chute fail flag
        if main_chute["riser_failed"]:
            chute_failed = True
            if ALLOW_CUTAWAY and (not main_chute["cutaway"]):
                main_chute["cutaway"] = True

        # Current mass state
        current_terminal_prop = terminal_prop_initial
        if used_terminal_prop:
            current_terminal_prop = max(terminal_prop_initial - TERMINAL_PROP_USED_ON_FIRE, 0.0)

        if aeroshell_attached:
            current_mass = (BUS_DRY_KG + current_terminal_prop +
                            MAIN_CHUTE_MASS_KG + HONEYCOMB_PAD_MASS_KG +
                            AEROSHELL_MASS_KG + max(ablator_mass,0.0))
        else:
            current_mass = (BUS_DRY_KG + current_terminal_prop +
                            MAIN_CHUTE_MASS_KG + HONEYCOMB_PAD_MASS_KG)

        # Terminal retro trigger
        fire_now = False
        if TERMINAL_RETRO_ENABLE and (not used_terminal_prop) and current_terminal_prop > 0.0:
            can_fire = (G12_COUNT > 0) or (terminal_prop_initial > 0.0)
            if can_fire:
                if RETRO_MODE == "at_altitude":
                    if y <= RETRO_TRIGGER_ALT_M:
                        fire_now = True
                elif RETRO_MODE == "after_chute_full":
                    if main_chute["inflation_frac"] >= 0.99:
                        fire_now = True

        # collect chute-only descent stats before retro
        if (not used_terminal_prop) and main_chute["deployed"] and (not main_chute["cutaway"]):
            pre_retro_vdesc_samples.append(-vy)  # downward rate magnitude

        if fire_now:
            vy += TERMINAL_REVERSE_DV_MPS
            used_terminal_prop  = True
            t_terminal_fire     = t
            events.append((t,"TERMINAL_RETRO",
                           f"t={t:.1f}s alt={y:.1f} m dv={TERMINAL_REVERSE_DV_MPS:.2f} m/s"))

            # recompute mass after prop dump
            current_terminal_prop = max(terminal_prop_initial - TERMINAL_PROP_USED_ON_FIRE, 0.0)
            if aeroshell_attached:
                current_mass = (BUS_DRY_KG + current_terminal_prop +
                                MAIN_CHUTE_MASS_KG + HONEYCOMB_PAD_MASS_KG +
                                AEROSHELL_MASS_KG + max(ablator_mass,0.0))
            else:
                current_mass = (BUS_DRY_KG + current_terminal_prop +
                                MAIN_CHUTE_MASS_KG + HONEYCOMB_PAD_MASS_KG)

        # Forces
        Fx_total = 0.0
        Fy_total = -current_mass * g_mag  # gravity down

        # body drag
        if aeroshell_attached:
            Cd   = AEROSHELL_CD
            Aref = AEROSHELL_AREA_M2
        else:
            Cd   = PROBE_CD
            Aref = PROBE_AREA_M2
        F_drag_mag = aero_drag_force(Cd, Aref, rho, vmag)
        if vmag > 1e-6:
            Fx_total += -F_drag_mag*(vx/vmag)
            Fy_total += -F_drag_mag*(vy/vmag)

        # drogue drag
        dFx, dFy = drogue_drag(drogue, rho, vx, vy)
        Fx_total += dFx
        Fy_total += dFy

        # main chute drag
        cFx, cFy = chute_drag(main_chute, rho, vx, vy)
        Fx_total += cFx
        Fy_total += cFy

        # Accelerations
        ax = Fx_total / current_mass
        ay = Fy_total / current_mass

        # total g magnitude (what you already had)
        g_load = math.hypot(ax, ay) / 9.80665

        # vertical g (down is negative ay; take magnitude)
        g_vert = abs(ay) / 9.80665

        # excess vertical g beyond static Mars gravity
        g_mars_local = mars_gravity(y) / 9.80665
        g_no_grav_vert = max(g_vert - g_mars_local, 0.0)

        if g_load > peak_g_entry_info["g"]:
            peak_g_entry_info = {"g":g_load, "t":t, "alt":y}

        if q_pa > peak_q_pa:
            peak_q_pa = q_pa
            max_q_pa_time_alt = (t, y)

        # --- Integrate motion with simple ground-contact event detection ---
        vx_old, vy_old = vx, vy
        x_old,  y_old  = x, y
        t_old          = t

        # same semi-implicit Euler scheme as before: update v then x,y
        vx = vx_old + ax * DT
        vy = vy_old + ay * DT
        x  = x_old  + vx * DT
        y  = y_old  + vy * DT
        t  = t_old  + DT

        # If we crossed y=0 within this step, linearly interpolate to contact
        if y <= 0.0 and y_old > 0.0:
            frac = y_old / (y_old - y)  # 0 <= frac <= 1
            t = t_old + frac * DT
            vx = vx_old + (vx - vx_old) * frac
            vy = vy_old + (vy - vy_old) * frac
            x  = x_old + (x - x_old) * frac
            y  = 0.0  # clamp exactly to ground

        # checkpoint velocities close to ground altitude
        if v_at_100m is None and y <= 100.0:
            v_at_100m = vy
        if v_at_50m  is None and y <= 50.0:
            v_at_50m  = vy
        if v_at_10m  is None and y <= 10.0:
            v_at_10m  = vy

        # EDL phase classification (for telemetry / plotting)
        if aeroshell_attached:
            if mach >= 3.0:
                phase = "hypersonic_body"
            elif mach >= 1.0:
                phase = "supersonic_body"
            else:
                phase = "subsonic_body"
        elif main_chute["deployed"] and not main_chute["cutaway"]:
            phase = "main_chute"
        elif drogue["active"]:
            phase = "drogue"
        elif TERMINAL_RETRO_ENABLE and used_terminal_prop:
            phase = "terminal_retro"
        else:
            phase = "freefall"

        # record telemetry row
        telemetry_rows.append({
            "t_s": t,
            "alt_m": y,
            "vx_mps": vx,
            "vy_mps": vy,
            "vmag_mps": math.hypot(vx, vy),
            "mach": mach,
            "rho": rho,
            "q_pa": q_pa,
            "heat_Wcm2": qdot_Wcm2,
            "heat_load_Jm2": cumulative_heat_Jm2,
            "shield_temp_K": shield_temp_K,
            "bus_temp_K": bus_temp_K,
            "ablator_mass_kg": ablator_mass,
            "mass_kg": current_mass,
            "chute_frac": main_chute["inflation_frac"],
            "drogue_active": drogue["active"],
            "g_load": g_load,
            "g_vert": g_vert,
            "g_no_grav_vert": g_no_grav_vert,
            "riser_force_N": main_chute["riser_force_N"],
            "ax_mps2": ax,
            "ay_mps2": ay,
            "phase": phase,
        })

        # traces for phase plots
        alt_trace.append(y)
        vvert_trace.append(vy)
        q_trace.append(q_pa)
        riser_trace.append(main_chute["riser_force_N"])
        riser_t.append(t)

        # new traces
        downrange_trace.append(x)
        mach_trace.append(mach)
        shield_T_trace.append(shield_temp_K)
        bus_T_trace.append(bus_temp_K)
        mass_trace.append(current_mass)
        ay_trace.append(ay)
        chute_frac_trace.append(main_chute["inflation_frac"])
        heatflux_trace.append(qdot_Wcm2)

        # touchdown?
        if y <= 0.0:
            touchdown_time = t
            break

    # === POST-SIM ANALYSIS ===

    if pre_retro_vdesc_samples:
        avg_pre_retro_descent_mps = sum(pre_retro_vdesc_samples)/len(pre_retro_vdesc_samples)
    else:
        avg_pre_retro_descent_mps = None

    if touchdown_time is not None and not tps_burnthrough:
        # final velocities at touchdown
        v_vert_final  = vy
        v_horiz_final = vx
        v_res         = math.hypot(v_vert_final, v_horiz_final)

        # prop left after burn
        if used_terminal_prop:
            final_terminal_prop_mass = max(terminal_prop_initial - TERMINAL_PROP_USED_ON_FIRE, 0.0)
        else:
            final_terminal_prop_mass = terminal_prop_initial

        landed_mass_kg = BUS_DRY_KG + final_terminal_prop_mass + MAIN_CHUTE_MASS_KG + HONEYCOMB_PAD_MASS_KG

        crush = honeycomb_model(
            lander_mass_kg=landed_mass_kg,
            v_impact_mps=abs(v_vert_final),
            g_limit_g=SURVIVABLE_PEAK_G_LIMIT
        )

        if crush["peak_g"] > SURVIVABLE_PEAK_G_LIMIT:
            landing_destroyed = True

        # chute corridor checks
        mach_at_deploy = main_chute["mach_at_deploy"]
        q_at_deploy    = main_chute["q_at_deploy"]
        if mach_at_deploy is not None and q_at_deploy is not None:
            chute_corridor_ok = (
                (mach_at_deploy >= MACH_MIN) and (mach_at_deploy <= MACH_MAX) and
                (q_at_deploy    >= Q_MIN_PA ) and (q_at_deploy    <= Q_MAX_PA )
            )
            mach_margin_low  = mach_at_deploy - MACH_MIN
            mach_margin_high = MACH_MAX - mach_at_deploy
            q_margin_low     = q_at_deploy - Q_MIN_PA
            q_margin_high    = Q_MAX_PA - q_at_deploy
        else:
            chute_corridor_ok = False
            mach_margin_low = mach_margin_high = None
            q_margin_low   = q_margin_high   = None

        # requirements
        req_touchdown_vertical_ok = (abs(v_vert_final) <= MAX_SAFE_TOUCHDOWN_VVERT_MPS)
        req_payload_g_ok          = (crush["peak_g"]   <= SURVIVABLE_PEAK_G_LIMIT)
        req_chute_ok              = (not chute_failed)
        req_tps_ok                = (not tps_burnthrough)
        req_entry_g_ok            = (peak_g_entry_info["g"] <= SURVIVABLE_ENTRY_PEAK_G_LIMIT)
        req_all_ok = (
            req_touchdown_vertical_ok and
            req_payload_g_ok and
            req_chute_ok and
            req_tps_ok and
            req_entry_g_ok
        )

        # phase durations
        dur_ei_to_jett   = aerojet_jettison_t - 0.0 if aerojet_jettison_t is not None else None
        dur_jett_to_main = (chute_deploy_time - aerojet_jettison_t) if (aerojet_jettison_t and chute_deploy_time) else None
        dur_main_to_retro= (t_terminal_fire - chute_deploy_time) if (t_terminal_fire and chute_deploy_time) else None
        dur_retro_to_td  = (touchdown_time - t_terminal_fire) if (t_terminal_fire and touchdown_time) else None
        total_ei_to_td   = touchdown_time - 0.0

        # summary bundle
        summary = {
            # mass/orbit
            "init_mass_total_kg": m0_total,
            "deorbit_dv_mps": required_dv_mps,
            "deorbit_prop_used_kg": prop_used_kg,
            "mass_at_EI_kg": mass_at_EI,
            "ballistic_coeff_EI_kg_m2": beta_EI_kg_m2,
            "ballistic_coeff_lander_kg_m2": beta_lander_kg_m2,

            # EI conditions
            "t_EI_s": 0.0,
            "EI_alt_m": ENTRY_INTERFACE_ALT_M,
            "EI_speed_mps": v_ei,

            # env config
            "dust_storm": DUST_STORM_ENABLE,
            "dust_rho_scale": (DUST_RHO_SCALE if DUST_STORM_ENABLE else 1.0),

            # aero/thermal peaks
            "peak_heat_Wcm2": peak_heat_Wcm2,
            "peak_dynamic_q_Pa": peak_q_pa,
            "peak_dynamic_q_when": {
                "t_s": max_q_pa_time_alt[0],
                "alt_m": max_q_pa_time_alt[1]
            },
            "total_heat_load_Jm2": cumulative_heat_Jm2,

            # G-load (aero / chute portion, not impact)
            "peak_g_load_entry": peak_g_entry_info["g"],
            "peak_g_load_entry_t_s": peak_g_entry_info["t"],
            "peak_g_load_entry_alt_m": peak_g_entry_info["alt"],

            # bus thermal exposure
            "bus_first_exposed": first_bus_exposure,
            "peak_bus_temp_after_exposure_K": peak_bus_temp_after_exposure,

            # chute info
            "main_chute_deployed_t": chute_deploy_time,
            "main_chute_failed": chute_failed,
            "main_chute_cutaway": main_chute["cutaway"],
            "aeroshell_jettison_t": aerojet_jettison_t,
            "terminal_retro_fire_t": t_terminal_fire,
            "terminal_retro_enabled": TERMINAL_RETRO_ENABLE,
            "peak_riser_force_N": peak_riser_force,
            "riser_force_limit_N": MAX_RISER_FORCE_N * SURVIVABLE_RISER_FACTOR,
            "chute_corridor_ok": chute_corridor_ok,
            "mach_at_deploy": mach_at_deploy,
            "q_at_deploy_Pa": q_at_deploy,
            "mach_margin_low": mach_margin_low,
            "mach_margin_high": mach_margin_high,
            "q_margin_low_Pa": q_margin_low,
            "q_margin_high_Pa": q_margin_high,

            # touchdown kinematics
            "touchdown_time_s": touchdown_time,
            "touchdown_vvert_mps": v_vert_final,
            "touchdown_vhoriz_mps": v_horiz_final,
            "touchdown_vres_mps": v_res,
            "v_at_100m_mps": v_at_100m,
            "v_at_50m_mps":  v_at_50m,
            "v_at_10m_mps":  v_at_10m,
            "avg_pre_retro_descent_mps": avg_pre_retro_descent_mps,
            "landed_mass_kg": landed_mass_kg,
            "downrange_m": x,

            # thermal & ablator post-landing
            "shield_temp_K_end": shield_temp_K,
            "bus_temp_K_end": bus_temp_K,
            "ablator_remaining_kg": max(ablator_mass,0.0),

            # honeycomb data (expanded)
            "impact_peak_g_payload": crush["peak_g"],
            "impact_g_mode": crush["governing_mode"],
            "impact_g_stiffness": crush["g_stiffness"],
            "impact_g_energy_limit": crush["g_energy_limit"],
            "crush_plateau_stress_Pa": crush["plateau_stress_Pa"],
            "crush_plateau_force_N": crush["plateau_force_N"],
            "crush_energy_capacity_J": crush["energy_capacity_J"],
            "impact_ke_J": crush["impact_ke_J"],
            "crush_absorbed_all": crush["absorbed_all"],
            "required_plateau_stress_for_limit_Pa": crush["required_plateau_stress_for_limit_Pa"],
            "required_plateau_stress_for_energy_Pa": crush["required_plateau_stress_for_energy_Pa"],
            "required_stroke_m": crush["required_stroke_m"],
            "F_required_energy_N": crush["F_required_energy_N"],
            "landing_destroyed": landing_destroyed,

            # pass/fail requirements
            "req_touchdown_vertical_ok": req_touchdown_vertical_ok,
            "req_payload_g_ok": req_payload_g_ok,
            "req_chute_ok": req_chute_ok,
            "req_tps_ok": req_tps_ok,
            "req_entry_g_ok": req_entry_g_ok,
            "req_all_ok": req_all_ok,

            # burnthrough etc.
            "tps_burnthrough": tps_burnthrough,

            # timing / phases
            "dur_ei_to_jett_s": dur_ei_to_jett,
            "dur_jett_to_main_s": dur_jett_to_main,
            "dur_main_to_retro_s": dur_main_to_retro,
            "dur_retro_to_td_s": dur_retro_to_td,
            "total_ei_to_td_s": total_ei_to_td
        }

    else:
        # either no touchdown or died before ground
        v_vert_final   = vy
        v_horiz_final  = vx
        v_res          = math.hypot(v_vert_final, v_horiz_final)

        crush = {
            "peak_g": None,
            "required_plateau_stress_for_limit_Pa": None,
            "required_stroke_m": None,
            "g_stiffness": None,
            "g_energy_limit": None,
            "governing_mode": None,
            "F_required_energy_N": None,
            "required_plateau_stress_for_energy_Pa": None
        }

        mach_at_deploy = main_chute["mach_at_deploy"]
        q_at_deploy    = main_chute["q_at_deploy"]
        chute_corridor_ok = (
            mach_at_deploy is not None and
            q_at_deploy is not None and
            (mach_at_deploy >= MACH_MIN) and (mach_at_deploy <= MACH_MAX) and
            (q_at_deploy    >= Q_MIN_PA ) and (q_at_deploy    <= Q_MAX_PA )
        )
        if mach_at_deploy is not None:
            mach_margin_low  = mach_at_deploy - MACH_MIN
            mach_margin_high = MACH_MAX - mach_at_deploy
        else:
            mach_margin_low  = None
            mach_margin_high = None
        if q_at_deploy is not None:
            q_margin_low     = q_at_deploy - Q_MIN_PA
            q_margin_high    = Q_MAX_PA - q_at_deploy
        else:
            q_margin_low = None
            q_margin_high= None

        req_touchdown_vertical_ok = (touchdown_time is not None and abs(v_vert_final) <= MAX_SAFE_TOUCHDOWN_VVERT_MPS)
        req_payload_g_ok          = (crush["peak_g"] is not None and crush["peak_g"] <= SURVIVABLE_PEAK_G_LIMIT)
        req_chute_ok              = (not chute_failed)
        req_tps_ok                = (not tps_burnthrough)
        req_entry_g_ok            = (peak_g_entry_info["g"] <= SURVIVABLE_ENTRY_PEAK_G_LIMIT)
        req_all_ok = (
            req_touchdown_vertical_ok and
            req_payload_g_ok and
            req_chute_ok and
            req_tps_ok and
            req_entry_g_ok
        )

        summary = {
            "init_mass_total_kg": m0_total,
            "deorbit_dv_mps": required_dv_mps,
            "deorbit_prop_used_kg": prop_used_kg,
            "mass_at_EI_kg": mass_at_EI,
            "ballistic_coeff_EI_kg_m2": beta_EI_kg_m2,
            "ballistic_coeff_lander_kg_m2": beta_lander_kg_m2,

            "t_EI_s": 0.0,
            "EI_alt_m": ENTRY_INTERFACE_ALT_M,
            "EI_speed_mps": v_ei,

            "dust_storm": DUST_STORM_ENABLE,
            "dust_rho_scale": (DUST_RHO_SCALE if DUST_STORM_ENABLE else 1.0),

            "peak_heat_Wcm2": peak_heat_Wcm2,
            "peak_dynamic_q_Pa": peak_q_pa,
            "peak_dynamic_q_when": {
                "t_s": max_q_pa_time_alt[0],
                "alt_m": max_q_pa_time_alt[1]
            },
            "total_heat_load_Jm2": cumulative_heat_Jm2,

            "peak_g_load_entry": peak_g_entry_info["g"],
            "peak_g_load_entry_t_s": peak_g_entry_info["t"],
            "peak_g_load_entry_alt_m": peak_g_entry_info["alt"],

            "bus_first_exposed": first_bus_exposure,
            "peak_bus_temp_after_exposure_K": peak_bus_temp_after_exposure,

            "main_chute_deployed_t": chute_deploy_time,
            "main_chute_failed": chute_failed,
            "main_chute_cutaway": main_chute["cutaway"],
            "aeroshell_jettison_t": aerojet_jettison_t,
            "terminal_retro_fire_t": t_terminal_fire,
            "terminal_retro_enabled": TERMINAL_RETRO_ENABLE,

            "peak_riser_force_N": peak_riser_force,
            "riser_force_limit_N": MAX_RISER_FORCE_N * SURVIVABLE_RISER_FACTOR,
            "chute_corridor_ok": chute_corridor_ok,
            "mach_at_deploy": mach_at_deploy,
            "q_at_deploy_Pa": q_at_deploy,
            "mach_margin_low": mach_margin_low,
            "mach_margin_high": mach_margin_high,
            "q_margin_low_Pa": q_margin_low,
            "q_margin_high_Pa": q_margin_high,

            "touchdown_time_s": touchdown_time,
            "touchdown_vvert_mps": v_vert_final,
            "touchdown_vhoriz_mps": v_horiz_final,
            "touchdown_vres_mps": v_res,
            "v_at_100m_mps": v_at_100m,
            "v_at_50m_mps":  v_at_50m,
            "v_at_10m_mps":  v_at_10m,
            "avg_pre_retro_descent_mps": avg_pre_retro_descent_mps,
            "landed_mass_kg": None,
            "downrange_m": x,

            "shield_temp_K_end": shield_temp_K,
            "bus_temp_K_end": bus_temp_K,
            "ablator_remaining_kg": max(ablator_mass,0.0),

            "impact_peak_g_payload": crush["peak_g"],
            "impact_g_mode": crush["governing_mode"],
            "impact_g_stiffness": crush["g_stiffness"],
            "impact_g_energy_limit": crush["g_energy_limit"],
            "required_plateau_stress_for_limit_Pa": crush["required_plateau_stress_for_limit_Pa"],
            "required_plateau_stress_for_energy_Pa": crush["required_plateau_stress_for_energy_Pa"],
            "required_stroke_m": crush["required_stroke_m"],
            "F_required_energy_N": crush["F_required_energy_N"],

            "landing_destroyed": None,

            "req_touchdown_vertical_ok": req_touchdown_vertical_ok,
            "req_payload_g_ok": req_payload_g_ok,
            "req_chute_ok": req_chute_ok,
            "req_tps_ok": req_tps_ok,
            "req_entry_g_ok": req_entry_g_ok,
            "req_all_ok": req_all_ok,

            "tps_burnthrough": tps_burnthrough,

            "dur_ei_to_jett_s": (aerojet_jettison_t - 0.0) if aerojet_jettison_t else None,
            "dur_jett_to_main_s": (chute_deploy_time - aerojet_jettison_t) if (aerojet_jettison_t and chute_deploy_time) else None,
            "dur_main_to_retro_s": (t_terminal_fire - chute_deploy_time) if (t_terminal_fire and chute_deploy_time) else None,
            "dur_retro_to_td_s": (touchdown_time - t_terminal_fire) if (t_terminal_fire and touchdown_time) else None,
            "total_ei_to_td_s": (touchdown_time - 0.0) if touchdown_time else None
        }

    # === WRITE OUTPUT FILES ===
    if WRITE_TELEMETRY_CSV and telemetry_rows:
        with open(TELEM_CSV_PATH, "w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(telemetry_rows[0].keys()))
            w.writeheader()
            for row in telemetry_rows:
                w.writerow(row)

    if WRITE_EVENTS_CSV:
        with open(EVENTS_CSV_PATH, "w", newline="") as fe:
            we = csv.writer(fe)
            we.writerow(["t_s", "event", "details"])
            we.writerow([0.0, "INIT",
                         f"Circular orbit {START_ALT_CIRC_M/1000:.1f} km alt, mass={m0_total:.2f} kg"])
            we.writerow([0.0, "DEORBIT",
                         f"Req DV={required_dv_mps:.2f} m/s, prop_used={prop_used_kg:.3f} kg, mass_at_EI={mass_at_EI:.2f} kg"])
            for (te, ev, det) in events:
                we.writerow([f"{te:.2f}", ev, det])

    if WRITE_SUMMARY_JSON:
        with open(SUMMARY_JSON_PATH, "w") as fj:
            json.dump(summary, fj, indent=2)

    # === PLOTS ===
    if telemetry_rows:
        ts   = [r["t_s"] for r in telemetry_rows]
        alt  = [r["alt_m"] for r in telemetry_rows]
        vel  = [r["vmag_mps"] for r in telemetry_rows]
        vvert= [r["vy_mps"] for r in telemetry_rows]
        gs   = [r["g_load"] for r in telemetry_rows]
        gvert= [r["g_vert"] for r in telemetry_rows]
        heat = [r["heat_Wcm2"] for r in telemetry_rows]
        qdyn = [r["q_pa"] for r in telemetry_rows]
        riser_force_series = [r["riser_force_N"] for r in telemetry_rows]
        mach_s = [r["mach"] for r in telemetry_rows]
        shieldT= [r["shield_temp_K"] for r in telemetry_rows]
        busT   = [r["bus_temp_K"] for r in telemetry_rows]
        mass_s = [r["mass_kg"] for r in telemetry_rows]
        ay_s   = [r["ay_mps2"] for r in telemetry_rows]
        chuteF = [r["chute_frac"] for r in telemetry_rows]
        heatload = [r["heat_load_Jm2"] for r in telemetry_rows]

        # 1. Time series basics
        make_and_optionally_save_plot(
            ts, alt,
            "Time [s]", "Altitude [m]",
            "Altitude vs Time",
            "altitude_vs_time"
        )

        make_and_optionally_save_plot(
            ts, vel,
            "Time [s]", "Speed [m/s]",
            "Speed vs Time",
            "speed_vs_time"
        )

        make_and_optionally_save_plot(
            ts, vvert,
            "Time [s]", "Vertical Velocity [m/s] (down is negative)",
            "Vertical Velocity vs Time",
            "vvert_vs_time"
        )

        make_and_optionally_save_plot(
            ts, gs,
            "Time [s]", "Total g-load [g]",
            "G-load vs Time",
            "gload_vs_time",
            scatter_points=[{
                "x": peak_g_entry_info["t"],
                "y": peak_g_entry_info["g"],
                "label":"peak aero g",
                "style":{"marker":"o","color":"red","zorder":5}
            }]
        )

        # NEW: vertical g vs time
        make_and_optionally_save_plot(
            ts, gvert,
            "Time [s]", "Vertical g-load [g]",
            "Vertical G-load vs Time",
            "gvert_vs_time"
        )

        make_and_optionally_save_plot(
            ts, heat,
            "Time [s]", "Heat Flux [W/cm^2]",
            "Heat Flux vs Time",
            "heatflux_vs_time"
        )

        # NEW: cumulative heat load vs time
        make_and_optionally_save_plot(
            ts, heatload,
            "Time [s]", "Cumulative Heat Load [J/m²]",
            "Integrated Heat Load vs Time",
            "heatload_vs_time"
        )

        make_and_optionally_save_plot(
            ts, qdyn,
            "Time [s]", "Dynamic Pressure [Pa]",
            "Dynamic Pressure vs Time",
            "q_vs_time",
            extra_lines=[{
                "y": Q_JETTISON_MAX_PA,
                "label": "Q jettison max",
                "style":{"linestyle":"--","color":"orange"}
            }]
        )

        # Riser force
        extra_lines_riser = [{
            "y": MAX_RISER_FORCE_N * SURVIVABLE_RISER_FACTOR,
            "label": "Riser limit",
            "style": {"linestyle":"--","color":"red"}
        }]
        scatter_pts_riser = []
        if chute_deploy_time is not None:
            idx_cd = min(range(len(ts)), key=lambda k: abs(ts[k]-chute_deploy_time))
            scatter_pts_riser.append({
                "x": ts[idx_cd],
                "y": riser_force_series[idx_cd],
                "label":"chute deploy",
                "style":{"marker":"x","color":"black","zorder":5}
            })

        make_and_optionally_save_plot(
            ts, riser_force_series,
            "Time [s]", "Riser Force [N]",
            "Main Chute Riser Force vs Time",
            "riser_force_vs_time",
            extra_lines=extra_lines_riser,
            scatter_points=scatter_pts_riser
        )

        # NEW: shield/bus temp vs time
        if SHOW_PLOTS or SAVE_PLOTS:
            plt.figure()
            plt.plot(ts, shieldT, label="shield")
            plt.plot(ts, busT,    label="bus")
            plt.xlabel("Time [s]")
            plt.ylabel("Temp [K]")
            plt.title("Component Temperatures vs Time")
            plt.grid(True)
            plt.axhline(MAX_TEMP_K, linestyle=":", color="gray")
            plt.legend()
            if SAVE_PLOTS:
                _savefig_stub("temps_vs_time")

        # NEW: chute inflation vs time
        make_and_optionally_save_plot(
            ts, chuteF,
            "Time [s]", "Main Chute Inflation Fraction [-]",
            "Chute Inflation vs Time",
            "chute_frac_vs_time"
        )

        # NEW: vehicle mass vs time
        make_and_optionally_save_plot(
            ts, mass_s,
            "Time [s]", "Mass [kg]",
            "Vehicle Mass vs Time",
            "mass_vs_time"
        )

        # NEW: vertical accel vs time
        make_and_optionally_save_plot(
            ts, ay_s,
            "Time [s]", "Vertical Accel ay [m/s²] (down negative)",
            "Vertical Acceleration vs Time",
            "ay_vs_time"
        )

        # 2. Phase plots vs altitude
        annos_vy = []
        if chute_deploy_time is not None:
            idx_cd = min(range(len(ts)), key=lambda k: abs(ts[k]-chute_deploy_time))
            annos_vy.append({
                "alt": alt[idx_cd],
                "val": vvert[idx_cd],
                "text":"MAIN CHUTE"
            })
        if aerojet_jettison_t is not None:
            idx_j = min(range(len(ts)), key=lambda k: abs(ts[k]-aerojet_jettison_t))
            annos_vy.append({
                "alt": alt[idx_j],
                "val": vvert[idx_j],
                "text":"AEROSHELL OFF"
            })
        if t_terminal_fire is not None:
            idx_rt = min(range(len(ts)), key=lambda k: abs(ts[k]-t_terminal_fire))
            annos_vy.append({
                "alt": alt[idx_rt],
                "val": vvert[idx_rt],
                "text":"TERMINAL RETRO"
            })
        if touchdown_time is not None:
            idx_td = min(range(len(ts)), key=lambda k: abs(ts[k]-touchdown_time))
            annos_vy.append({
                "alt": alt[idx_td],
                "val": vvert[idx_td],
                "text":"TOUCHDOWN"
            })

        make_altitude_phase_plot(
            alt_list=alt_trace,
            val_list=vvert_trace,
            xlabel="Vertical Velocity [m/s] (down is negative)",
            ylabel="Altitude [m]",
            title="Descent Corridor (Vy vs Alt)",
            filename_stub="vy_vs_alt",
            annotations=annos_vy
        )

        annos_q = []
        if chute_deploy_time is not None:
            idx_cd = min(range(len(ts)), key=lambda k: abs(ts[k]-chute_deploy_time))
            annos_q.append({
                "alt": alt[idx_cd],
                "val": qdyn[idx_cd],
                "text":"MAIN CHUTE"
            })
        if aerojet_jettison_t is not None:
            idx_j = min(range(len(ts)), key=lambda k: abs(ts[k]-aerojet_jettison_t))
            annos_q.append({
                "alt": alt[idx_j],
                "val": qdyn[idx_j],
                "text":"AEROSHELL OFF"
            })

        make_altitude_phase_plot(
            alt_list=alt_trace,
            val_list=qdyn,
            xlabel="Dynamic Pressure q [Pa]",
            ylabel="Altitude [m]",
            title="Dynamic Pressure vs Altitude",
            filename_stub="q_vs_alt",
            annotations=annos_q,
            extra_lines=[
                {
                    "horizontal_alt": True,
                    "alt": JETTISON_MIN_ALT_M,
                    "label": "JETTISON_MIN_ALT",
                    "style":{"linestyle":":","color":"gray"}
                },
                {
                    "vertical_val": True,
                    "val": Q_JETTISON_MAX_PA,
                    "label": "Q_JETTISON_MAX",
                    "style":{"linestyle":"--","color":"orange"}
                },
            ]
        )

        # NEW: Mach vs Alt
        make_altitude_phase_plot(
            alt_list=alt_trace,
            val_list=mach_s,
            xlabel="Mach [-]",
            ylabel="Altitude [m]",
            title="Mach vs Altitude",
            filename_stub="mach_vs_alt"
        )

        # NEW: Heat Flux vs Altitude
        make_altitude_phase_plot(
            alt_list=alt_trace,
            val_list=heatflux_trace,
            xlabel="Heat Flux [W/cm^2]",
            ylabel="Altitude [m]",
            title="Heat Flux vs Altitude",
            filename_stub="heatflux_vs_alt"
        )

        # NEW: G-load vs Altitude
        make_altitude_phase_plot(
            alt_list=alt_trace,
            val_list=gs,
            xlabel="Total g-load [g]",
            ylabel="Altitude [m]",
            title="G-load vs Altitude",
            filename_stub="g_vs_alt",
            annotations=[{
                "alt": peak_g_entry_info["alt"],
                "val": peak_g_entry_info["g"],
                "text":"peak aero g"
            }]
        )

        # NEW: ground track-ish (alt vs downrange)
        if SHOW_PLOTS or SAVE_PLOTS:
            plt.figure()
            plt.plot([d/1000.0 for d in downrange_trace], alt_trace)
            plt.xlabel("Downrange [km]")
            plt.ylabel("Altitude [m]")
            plt.title("Altitude vs Downrange")
            plt.grid(True)
            # We no longer invert altitude: 0 m is at the bottom.
            # plt.gca().invert_yaxis()
            if SAVE_PLOTS:
                _savefig_stub("alt_vs_downrange")

        # NEW: terminal corridor: Vy vs q below 500 m
        lowalt_idx = [i for i,a in enumerate(alt_trace) if a <= 500.0]
        if lowalt_idx:
            sel_vy = [vvert_trace[i] for i in lowalt_idx]
            sel_q  = [q_trace[i]    for i in lowalt_idx]

            if SHOW_PLOTS or SAVE_PLOTS:
                plt.figure()
                plt.scatter(sel_q, sel_vy, s=8)
                plt.xlabel("Dynamic Pressure q [Pa]")
                plt.ylabel("Vertical Velocity [m/s] (down negative)")
                plt.title("Terminal Corridor (Vy vs q, <500 m AGL)")
                plt.grid(True)

                # annotate retro-fire and touchdown if in this set
                if t_terminal_fire is not None:
                    idx_rt = min(lowalt_idx, key=lambda k: abs(ts[k]-t_terminal_fire))
                    plt.scatter(q_trace[idx_rt], vvert_trace[idx_rt],
                                marker="x", color="black")
                    plt.text(q_trace[idx_rt], vvert_trace[idx_rt],
                             " retro burn", fontsize=8,
                             va="bottom", ha="left")
                if touchdown_time is not None:
                    idx_td = min(lowalt_idx, key=lambda k: abs(ts[k]-touchdown_time))
                    plt.scatter(q_trace[idx_td], vvert_trace[idx_td],
                                marker="o", color="red")
                    plt.text(q_trace[idx_td], vvert_trace[idx_td],
                             " touchdown", fontsize=8,
                             va="bottom", ha="left")

                if SAVE_PLOTS:
                    _savefig_stub("terminal_corridor_vy_vs_q")

        # finally either show or close figs
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close('all')

    # === CONSOLE REPORT ===

    print(f"[INIT] Circular orbit {START_ALT_CIRC_M/1000:.1f} km alt, mass={m0_total:.2f} kg")
    print(f"[DEORBIT] solving required Δv to reach periapsis {TARGET_PERIAPSIS_ALT_M/1000:.1f} km ...")
    print(f"          required DV ~ {required_dv_mps:.2f} m/s")
    print(f"          deorbit prop mass ~ {prop_used_kg:.3f} kg (Isp=57.0s)")
    print(f"          EI stack mass after drop ~ {mass_at_EI:.2f} kg")
    print(f"          β_EI (aeroshell): {beta_EI_kg_m2:.1f} kg/m²")
    print(f"          β_lander (post-jettison body): {beta_lander_kg_m2:.1f} kg/m²")

    if tps_burnthrough:
        # we died thermally before touchdown
        print(f"[LOSS] TPS burn-through at t≈{t:.1f}s alt≈{y:.1f} m.")
        print("")
        print("--- Reentry Peak Loads ---")
        print(f"Peak heat flux:   {peak_heat_Wcm2:.2f} W/cm^2")
        print(f"Total heat load:  {cumulative_heat_Jm2:.1f} J/m²")
        print(f"Max dynamic q:    {peak_q_pa:.0f} Pa at t={max_q_pa_time_alt[0]:.1f}s alt={max_q_pa_time_alt[1]:.0f} m")
        print(f"Peak aero g-load: {peak_g_entry_info['g']:.2f} g @ t={peak_g_entry_info['t']:.1f}s alt={peak_g_entry_info['alt']:.1f} m")
        if first_bus_exposure:
            print("")
            print("[BUS FIRST EXPOSED]")
            print(f" t={first_bus_exposure['t']:.1f}s alt={first_bus_exposure['alt_m']:.1f} m")
            print(f" q={first_bus_exposure['q_pa']:.1f} Pa heat={first_bus_exposure['heat_Wcm2']:.2f} W/cm^2")
            print(f" temps: shield={first_bus_exposure['shield_temp_K']:.1f} K bus={first_bus_exposure['bus_temp_K']:.1f} K")
            if peak_bus_temp_after_exposure is not None:
                print(f" peak bus temp after exposure: {peak_bus_temp_after_exposure:.1f} K")
        print("")
        print("=== EVENT TIMELINE ===")
        for (te, ev, det) in events:
            print(f"t={te:.1f}s : {ev} :: {det}")
        print("")
        print("Vehicle did not survive EDL (thermal).")
        print("")
        print("=== FILE OUTPUTS ===")
        print(f"{TELEM_CSV_PATH}")
        print(f"{EVENTS_CSV_PATH}")
        print(f"{SUMMARY_JSON_PATH}")
        if SAVE_PLOTS:
            # list of all .pngs we generated
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"altitude_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"speed_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"vvert_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"gload_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"gvert_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatflux_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatload_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"q_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"riser_force_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"chute_frac_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"mass_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"ay_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"temps_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"vy_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"q_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"mach_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatflux_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"g_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"alt_vs_downrange.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"terminal_corridor_vy_vs_q.png"))
        print("Done.")
        return

    if touchdown_time is None:
        # we never actually got to y<=0 (or we bailed earlier but not thermal death)
        print("[WARN] DID NOT REACH SURFACE in sim window or was lost.")
        print("")
        print(f"[INIT] circular orbit {START_ALT_CIRC_M/1000:.0f} km around Mars")
        print(f"       mass(preburn)={m0_total:.2f} kg, req_dv={required_dv_mps:.2f} m/s")
        print(f"[DEORBIT] EI reached at t=0.0s then sim ended t={t:.1f}s alt={y:.1f} m")
        print("")
        print("--- Reentry Peak Loads ---")
        print(f"Peak heat flux:   {peak_heat_Wcm2:.2f} W/cm^2")
        print(f"Total heat load:  {cumulative_heat_Jm2:.1f} J/m²")
        print(f"Max dynamic q:    {peak_q_pa:.0f} Pa at t={max_q_pa_time_alt[0]:.1f}s alt={max_q_pa_time_alt[1]:.0f} m")
        print(f"Peak aero g-load: {peak_g_entry_info['g']:.2f} g @ t={peak_g_entry_info['t']:.1f}s alt={peak_g_entry_info['alt']:.1f} m")
        if first_bus_exposure:
            print("")
            print("[BUS FIRST EXPOSED]")
            print(f" t={first_bus_exposure['t']:.1f}s alt={first_bus_exposure['alt_m']:.1f} m")
            print(f" q={first_bus_exposure['q_pa']:.1f} Pa heat={first_bus_exposure['heat_Wcm2']:.2f} W/cm^2")
            print(f" temps: shield={first_bus_exposure['shield_temp_K']:.1f} K bus={first_bus_exposure['bus_temp_K']:.1f} K")
            if peak_bus_temp_after_exposure is not None:
                print(f" peak bus temp after exposure: {peak_bus_temp_after_exposure:.1f} K")
        print("")
        print("[CHUTE] Final chute status:")
        print(f"        deployed={main_chute['deployed']} cutaway={main_chute['cutaway']}")
        print(f"        peak_riser_force={peak_riser_force:.1f} N "
              f"(limit {MAX_RISER_FORCE_N*SURVIVABLE_RISER_FACTOR:.1f} N)")
        print(f"        riser_failed_anytime={chute_failed} cutaway={main_chute['cutaway']}")
        if main_chute["mach_at_deploy"] is not None and main_chute["q_at_deploy"] is not None:
            print(f"        deploy corridor: M={main_chute['mach_at_deploy']:.2f} "
                  f"[{MACH_MIN:.2f}..{MACH_MAX:.2f}], q={main_chute['q_at_deploy']:.1f} Pa "
                  f"[{Q_MIN_PA:.0f}..{Q_MAX_PA:.0f}]")
        else:
            print("        main chute never deployed.")
        print("")
        if avg_pre_retro_descent_mps is not None:
            print(f"[DESCENT RATE] Avg under-chute (pre-retro) descent rate: {avg_pre_retro_descent_mps:.2f} m/s downward")
        print(f"[LOW ALT] Vy@100m={v_at_100m:.2f} m/s Vy@50m={v_at_50m:.2f} m/s Vy@10m={v_at_10m:.2f} m/s")
        print(f"[RANGE] Horizontal downrange from EI: {x/1000.0:.2f} km")
        print("")
        print("=== PHASE DURATIONS ===")
        if aerojet_jettison_t is not None:
            print(f" EI→Jettison: {aerojet_jettison_t-0.0:.1f} s")
        if (aerojet_jettison_t and chute_deploy_time):
            print(f" Jettison→Main Chute: {chute_deploy_time-aerojet_jettison_t:.1f} s")
        if (chute_deploy_time and t_terminal_fire):
            print(f" Main Chute→Terminal Retro: {t_terminal_fire-chute_deploy_time:.1f} s")
        if (t_terminal_fire and touchdown_time):
            print(f" Terminal Retro→Touchdown: {touchdown_time-t_terminal_fire:.1f} s")
        print("")
        print("=== EVENT TIMELINE ===")
        for (te, ev, det) in events:
            print(f"t={te:.1f}s : {ev} :: {det}")
        print("")
        print("=== FILE OUTPUTS ===")
        print(f"{TELEM_CSV_PATH}")
        print(f"{EVENTS_CSV_PATH}")
        print(f"{SUMMARY_JSON_PATH}")
        if SAVE_PLOTS:
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"altitude_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"speed_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"vvert_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"gload_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"gvert_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatflux_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatload_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"q_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"riser_force_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"chute_frac_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"mass_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"ay_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"temps_vs_time.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"vy_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"q_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"mach_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatflux_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"g_vs_alt.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"alt_vs_downrange.png"))
            print(os.path.join(SAVE_DIR, SAVE_PREFIX+"terminal_corridor_vy_vs_q.png"))
        print("Done.")
        return

    # touched down + survived thermal (though maybe not structurally)

    crush_peak_g  = summary["impact_peak_g_payload"]
    crush_abs_all = summary.get("crush_absorbed_all", None)
    req_stress_for_limit = summary.get("required_plateau_stress_for_limit_Pa", None)

    # simple touchdown energetics
    landed_mass = summary["landed_mass_kg"]
    vdown       = abs(summary["touchdown_vvert_mps"])
    impact_ke   = 0.5 * landed_mass * (vdown**2) if landed_mass is not None else None
    specific_ke = (impact_ke / landed_mass) if (landed_mass and impact_ke is not None) else None
    momentum    = landed_mass * vdown if landed_mass is not None else None

    print(f"[EI] Entry interface reached at t=0.0s, alt~{ENTRY_INTERFACE_ALT_M/1000:.1f} km, V~{summary['EI_speed_mps']:.1f} m/s")

    if aerojet_jettison_t is not None:
        print(f"[EDL] t={aerojet_jettison_t:.1f}s AEROSHELL JETTISON")
    if chute_deploy_time is not None:
        print(f"[EDL] t={chute_deploy_time:.1f}s MAIN CHUTE DEPLOY (reefed)")
    if t_terminal_fire is not None:
        print(f"[EDL] t={t_terminal_fire:.1f}s TERMINAL RETRO FIRE dv={TERMINAL_REVERSE_DV_MPS:.1f} m/s "
              f"(enabled={TERMINAL_RETRO_ENABLE})")
    print(f"[EDL] t={touchdown_time:.1f}s TOUCHDOWN detected: vx={summary['touchdown_vhoriz_mps']:.2f} m/s vy={summary['touchdown_vvert_mps']:.2f} m/s")

    print("")
    print(f"[INIT] circular orbit {START_ALT_CIRC_M/1000:.0f} km around Mars")
    print(f"       mass(preburn)={m0_total:.2f} kg, req_dv={required_dv_mps:.2f} m/s")
    print(f"[DEORBIT] EI reached at t=0.0s, touchdown at t={touchdown_time:.1f}s")
    print("")
    print("--- Reentry Peak Loads ---")
    print(f"Peak heat flux:   {peak_heat_Wcm2:.2f} W/cm^2")
    print(f"Total heat load:  {cumulative_heat_Jm2:.1f} J/m²")
    print(f"Max dynamic q:    {peak_q_pa:.0f} Pa at t={max_q_pa_time_alt[0]:.1f}s alt={max_q_pa_time_alt[1]:.0f} m")
    print(f"Peak aero g-load: {peak_g_entry_info['g']:.2f} g @ t={peak_g_entry_info['t']:.1f}s alt={peak_g_entry_info['alt']:.1f} m")
    print("")
    if first_bus_exposure:
        print("[BUS FIRST EXPOSED]")
        print(f" t={first_bus_exposure['t']:.1f}s alt={first_bus_exposure['alt_m']:.1f} m")
        print(f" q={first_bus_exposure['q_pa']:.1f} Pa heat={first_bus_exposure['heat_Wcm2']:.2f} W/cm^2")
        print(f" temps: shield={first_bus_exposure['shield_temp_K']:.1f} K bus={first_bus_exposure['bus_temp_K']:.1f} K")
    if peak_bus_temp_after_exposure is not None:
        print(f" peak bus temp after exposure: {peak_bus_temp_after_exposure:.1f} K")
    print("")

    # Aerodynamic staging snapshots
    print("[AERODYNAMIC STAGING SNAPSHOTS]")
    def _print_stage(label, snap):
        if snap is None:
            print(f" {label}: never occurred")
        else:
            print(
                f" {label}: t={snap['t']:.1f}s alt={snap['alt_m']:.1f} m "
                f"M={snap['mach']:.2f} q={snap['q_pa']:.1f} Pa "
                f"heat={snap['heat_Wcm2']:.2f} W/cm^2 "
                f"shieldT={snap['shield_temp_K']:.1f} K busT={snap['bus_temp_K']:.1f} K"
            )
    _print_stage("AEROSHELL_JETTISON", aero_staging["AEROSHELL_JETTISON"])
    _print_stage("DROGUE_DEPLOY",      aero_staging["DROGUE_DEPLOY"])
    _print_stage("MAIN_CHUTE_DEPLOY",  aero_staging["MAIN_CHUTE_DEPLOY"])
    print("")

    print("[CHUTE] Final chute status:")
    print(f"        deployed={main_chute['deployed']} cutaway={main_chute['cutaway']}")
    print(f"        peak_riser_force={peak_riser_force:.1f} N "
          f"(limit {MAX_RISER_FORCE_N*SURVIVABLE_RISER_FACTOR:.1f} N)")
    print(f"        riser_failed_anytime={chute_failed} cutaway={main_chute['cutaway']}")
    if main_chute["mach_at_deploy"] is not None and main_chute["q_at_deploy"] is not None:
        print(f"        deploy corridor: M={main_chute['mach_at_deploy']:.2f} "
              f"[{MACH_MIN:.2f}..{MACH_MAX:.2f}], q={main_chute['q_at_deploy']:.1f} Pa "
              f"[{Q_MIN_PA:.0f}..{Q_MAX_PA:.0f}]")
        if summary["mach_margin_low"] is not None:
            print(f"        corridor margins: "
                  f"M_low_margin={summary['mach_margin_low']:.2f}, "
                  f"M_high_margin={summary['mach_margin_high']:.2f}, "
                  f"q_low_margin={summary['q_margin_low_Pa']:.1f} Pa, "
                  f"q_high_margin={summary['q_margin_high_Pa']:.1f} Pa")
    else:
        print("        main chute never deployed.")
        if first_below_para_alt_snapshot is not None:
            snap = first_below_para_alt_snapshot
            print("        first alt <= deploy threshold:")
            print(f"          t={snap['t']:.1f}s alt={snap['alt_m']:.1f} m "
                  f"M={snap['mach']:.2f} q={snap['q_pa']:.1f} Pa "
                  f"(corridor M:[{MACH_MIN:.2f}..{MACH_MAX:.2f}] "
                  f"q:[{Q_MIN_PA:.0f}..{Q_MAX_PA:.0f}] Pa)")
            reasons = []
            if snap['mach'] < MACH_MIN:
                reasons.append("Mach below minimum")
            elif snap['mach'] > MACH_MAX:
                reasons.append("Mach above maximum")
            if snap['q_pa'] < Q_MIN_PA:
                reasons.append("q below minimum")
            elif snap['q_pa'] > Q_MAX_PA:
                reasons.append("q above maximum")
            if reasons:
                print("          corridor violations at that point: " + ", ".join(reasons))
    print("")

    print("=== FINAL LANDING / IMPACT ===")
    print(f"Touchdown time: {touchdown_time:.1f} s")
    print(f"Touchdown vertical vel: {summary['touchdown_vvert_mps']:.2f} m/s")
    print(f"Touchdown horizontal vel: {summary['touchdown_vhoriz_mps']:.2f} m/s")
    print(f"Resultant touchdown speed: {summary['touchdown_vres_mps']:.2f} m/s")
    print(f"Vy@100m={v_at_100m:.2f} m/s Vy@50m={v_at_50m:.2f} m/s Vy@10m={v_at_10m:.2f} m/s")
    if avg_pre_retro_descent_mps is not None:
        print(f"Avg under-chute (pre-retro) descent rate: {avg_pre_retro_descent_mps:.2f} m/s downward")
    print(f"Mass at touchdown (lander+chute+pad+remaining prop): {summary['landed_mass_kg']:.2f} kg")
    print(f"Downrange from EI: {summary['downrange_m']/1000.0:.2f} km")
    print(f"Shield temp end: {summary['shield_temp_K_end']:.1f} K")
    print(f"Bus temp end:    {summary['bus_temp_K_end']:.1f} K")
    print(f"Ablator remaining: {summary['ablator_remaining_kg']:.3f} kg")
    print("")
    print("[IMPACT ENERGETICS]")
    if impact_ke is not None:
        print(f"Impact KE:      {impact_ke:.1f} J total")
    if specific_ke is not None:
        print(f"Specific KE:    {specific_ke:.1f} J/kg")
    if momentum is not None:
        print(f"Downward mom.:  {momentum:.2f} N·s")
    print("")
    print("[HONEYCOMB PAD MODEL]")
    print(f"stroke_available: {HONEYCOMB_STROKE_M:.3f} m over footprint {HONEYCOMB_FOOTPRINT_M2:.3f} m^2")
    print(f"rel_density ρ*/ρs: {HONEYCOMB_REL_DENSITY:.5f}")
    print(f"base material σy:  {HONEYCOMB_BASE_MATERIAL_SIGY_MPa:.1f} MPa (input)")
    print(f"PF_eff (blend):    {HONEYCOMB_PLATEAU_FACTOR:.5f} (-)  [from honeycomb generator]")
    print("")
    print(f"plateau_stress:    {summary['crush_plateau_stress_Pa']/1e6:.3f} MPa")
    print(f"plateau_force:     {summary['crush_plateau_force_N']:.1f} N")
    print(f"energy_capacity:   {summary['crush_energy_capacity_J']:.1f} J")
    print(f"impact_KE:         {summary['impact_ke_J']:.1f} J")
    print(f"absorbed_all?:     {crush_abs_all}")
    print(f"required stroke to fully absorb KE: {summary['required_stroke_m']:.4f} m "
          f"(available {HONEYCOMB_STROKE_M:.4f} m)")
    print("")
    print(f"F_required_energy: {summary['F_required_energy_N']:.1f} N "
          f"(=> stress {summary['required_plateau_stress_for_energy_Pa']/1e6:.3f} MPa "
          f"to JUST stop in {HONEYCOMB_STROKE_M:.3f} m)")
    print("")
    print(f"g_stiffness:    {summary['impact_g_stiffness']:.1f} g  (from actual plateau force)")
    print(f"g_energy_limit: {summary['impact_g_energy_limit']:.1f} g  (ideal if you ride full stroke)")
    print(f"governing_mode: {summary['impact_g_mode']}  -> peak_landing_g = {summary['impact_peak_g_payload']:.1f} g "
          f"(limit {SURVIVABLE_PEAK_G_LIMIT:.1f} g)")
    if req_stress_for_limit is not None:
        print(f"To stay under {SURVIVABLE_PEAK_G_LIMIT:.1f} g with mass={summary['landed_mass_kg']:.2f} kg "
              f"and foot={HONEYCOMB_FOOTPRINT_M2:.2f} m^2:")
        print(f"  required plateau stress ≲ {req_stress_for_limit/1e6:.3f} MPa")
    print("")
    print("[HONEYCOMB INTERPRETATION]")
    print("If governing_mode == 'stiffness': pad is too stiff, you spike g early. Softer core / lower PF_eff / lower rel_density / bigger footprint will drop g.")
    print("If governing_mode == 'stroke': you're using all available crush distance. More stroke (taller core) is how to go lower in g.")
    print("Bigger footprint always helps because same force is spread over more area -> less stress -> less g.")
    print("")
    print("=== SURVIVABILITY ASSESSMENT ===")
    print(f"CHUTE STRUCT: {'FAIL' if chute_failed else 'PASS'} "
          f"(peak {peak_riser_force:.0f} N / limit {MAX_RISER_FORCE_N*SURVIVABLE_RISER_FACTOR:.0f} N)")
    # chute corridor print (handle never-deployed case)
    if main_chute["mach_at_deploy"] is None or main_chute["q_at_deploy"] is None:
        corridor_str = "main chute never deployed"
    else:
        corridor_str = (f"M={main_chute['mach_at_deploy']:.2f} "
                        f"q={main_chute['q_at_deploy']:.1f} Pa")
    print(f"CHUTE CORRIDOR: {'PASS' if summary['chute_corridor_ok'] else 'FAIL'} ({corridor_str})")
    print(f"TERMINAL DESCENT: {'PASS' if summary['req_touchdown_vertical_ok'] else 'FAIL'} "
          f"({abs(summary['touchdown_vvert_mps']):.1f} m/s vs {MAX_SAFE_TOUCHDOWN_VVERT_MPS:.1f} m/s allowed)")
    print(f"PAYLOAD G: {'PASS' if summary['req_payload_g_ok'] else 'FAIL'} "
          f"({summary['impact_peak_g_payload']:.1f} g vs {SURVIVABLE_PEAK_G_LIMIT:.1f} g limit)")
    print(f"ENTRY G: {'PASS' if summary['req_entry_g_ok'] else 'FAIL'} "
          f"({peak_g_entry_info['g']:.1f} g vs {SURVIVABLE_ENTRY_PEAK_G_LIMIT:.1f} g limit)")
    print(f"TPS THERMAL: {'PASS' if summary['req_tps_ok'] else 'FAIL'}")
    print(f"OVERALL EDL RESULT: {'PASS' if summary['req_all_ok'] else 'FAIL'}")
    print("")

    # FAILURE TAGS for quick scanning
    failure_tags = []
    if not summary["req_touchdown_vertical_ok"]:
        failure_tags.append("touchdown_vertical_speed_out_of_limit")
    if not summary["req_payload_g_ok"]:
        failure_tags.append("payload_g_out_of_limit")
    if (summary["impact_peak_g_payload"] is not None and
        summary["impact_peak_g_payload"] > SURVIVABLE_PEAK_G_LIMIT):
        failure_tags.append("landing_crush_g_exceeded_limit")
    if not main_chute["deployed"]:
        failure_tags.append("main_chute_never_deployed")
    if tps_burnthrough:
        failure_tags.append("tps_burnthrough")
    if chute_failed:
        failure_tags.append("chute_structural_failure")

    if failure_tags:
        print("FAILURE TAGS: " + ", ".join(failure_tags))
    else:
        print("FAILURE TAGS: (none)")
    print("")

    print("=== G-LOAD SUMMARY ===")
    print(f"Peak g during atmospheric EDL (aero/chute): {peak_g_entry_info['g']:.1f} g "
          f"@ t={peak_g_entry_info['t']:.1f}s alt={peak_g_entry_info['alt']:.0f} m")
    print(f"Peak g at impact transmitted through honeycomb (payload g): {summary['impact_peak_g_payload']:.1f} g @ touchdown")
    if req_stress_for_limit is not None:
        print(f"G-safe honeycomb target stress (~{SURVIVABLE_PEAK_G_LIMIT:.0f} g limit): {req_stress_for_limit/1e6:.3f} MPa")
    print("")
    print("=== PHASE DURATIONS ===")
    if summary["dur_ei_to_jett_s"] is not None:
        frac = summary["dur_ei_to_jett_s"]/summary["total_ei_to_td_s"]*100.0 if summary["total_ei_to_td_s"] else None
        print(f" EI→Jettison: {summary['dur_ei_to_jett_s']:.1f} s"
              + (f" ({frac:.1f}% of total)" if frac is not None else ""))
    if summary["dur_jett_to_main_s"] is not None:
        frac = summary["dur_jett_to_main_s"]/summary["total_ei_to_td_s"]*100.0 if summary["total_ei_to_td_s"] else None
        print(f" Jettison→Main Chute: {summary['dur_jett_to_main_s']:.1f} s"
              + (f" ({frac:.1f}% of total)" if frac is not None else ""))
    if summary["dur_main_to_retro_s"] is not None:
        frac = summary["dur_main_to_retro_s"]/summary["total_ei_to_td_s"]*100.0 if summary["total_ei_to_td_s"] else None
        print(f" Main Chute→Terminal Retro: {summary['dur_main_to_retro_s']:.1f} s"
              + (f" ({frac:.1f}% of total)" if frac is not None else ""))
    if summary["dur_retro_to_td_s"] is not None:
        frac = summary["dur_retro_to_td_s"]/summary["total_ei_to_td_s"]*100.0 if summary["total_ei_to_td_s"] else None
        print(f" Terminal Retro→Touchdown: {summary['dur_retro_to_td_s']:.1f} s"
              + (f" ({frac:.1f}% of total)" if frac is not None else ""))
    print("")
    print("=== EVENT TIMELINE ===")
    print(f"t=0.0s : INIT :: Circular orbit {START_ALT_CIRC_M/1000:.1f} km alt, mass={m0_total:.2f} kg")
    print(f"t=0.0s : DEORBIT :: Req DV={required_dv_mps:.2f} m/s, prop_used={prop_used_kg:.3f} kg, mass_at_EI={mass_at_EI:.2f} kg")
    for (te, ev, det) in events:
        print(f"t={te:.1f}s : {ev} :: {det}")
    print("")
    print("=== FILE OUTPUTS ===")
    print(f"{TELEM_CSV_PATH}")
    print(f"{EVENTS_CSV_PATH}")
    print(f"{SUMMARY_JSON_PATH}")
    if SAVE_PLOTS:
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"altitude_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"speed_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"vvert_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"gload_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"gvert_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatflux_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatload_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"q_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"riser_force_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"chute_frac_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"mass_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"ay_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"temps_vs_time.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"vy_vs_alt.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"q_vs_alt.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"mach_vs_alt.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"heatflux_vs_alt.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"g_vs_alt.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"alt_vs_downrange.png"))
        print(os.path.join(SAVE_DIR, SAVE_PREFIX+"terminal_corridor_vy_vs_q.png"))
    print("Done.")

##############################################
# ================== RUN =====================
##############################################

if __name__ == "__main__":
    run_edl_sim()
