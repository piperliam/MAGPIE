
# MAGPIE Probe Honeycomb Crush SCAD Script
# 2025 Liam Piper

# This is to create a crush core part for Project MAGPIE (plus ouputs to deteremine how much energy the system can take 

import math
from textwrap import dedent

# =========================
# ===== CONFIG  ======
# =========================

# Physical geometry (meters)
R_TOTAL_M         = 0.400         # [m] outer radius of entire pad (e.g. 0.40 => 0.8 m dia)
STROKE_H_M        = 0.050         # [m] crush stroke height / core height in Z
RIM_AREA_FRAC     = 0.05          # [-] area fraction reserved for rim ring (0..1)

# Honeycomb cell geometry "base" (inner region baseline)
CELL_FLAT_INNER_M = 0.010         # [m] flat-to-flat size of hex cell in inner zone
PF_INNER_NOM      = 0.005         # [-] inner plateau factor target (soft)
PF_RIM_NOM        = 0.050         # [-] rim plateau factor target (stiff)

# Material props
RHO_MAT           = 2700.0        # [kg/m^3] approx aluminum
SIGY_BASE_MPA     = 120.0         # [MPa] base material yield

# Meshing overscan (tiles beyond boundary to ensure full trim)
NX_EXTRA = 2
NY_EXTRA = 2

# Output file
SCAD_FILENAME     = "crush_core.scad"


# =========================
# === GEOMETRY HELPERS ====
# =========================

def inner_radius_from_area_frac(r_total, rim_area_frac):
    """
    We want the rim ring (between r_inner and r_total) to be RIM_AREA_FRAC of total area.
    Area_total = pi * r_total^2
    Area_inner = (1 - rim_area_frac)*Area_total
    r_inner    = sqrt(Area_inner / pi)
    """
    area_total = math.pi * r_total**2
    area_inner = (1.0 - rim_area_frac) * area_total
    r_inner    = math.sqrt(area_inner / math.pi)
    return r_inner, area_total, area_inner

def hex_cell_metrics(cell_flat):
    """
    - cell_flat is flat-to-flat distance
    """
    apothem = cell_flat / 2.0              # distance center->flat
    circ_r  = apothem / math.cos(math.radians(30.0))  # distance center->vertex
    hex_h   = 2.0 * apothem                # vertical height flat-to-flat
    pitch_x = 1.5 * circ_r
    pitch_y = hex_h
    # circumference and area of the hex cell polygon
    area_cell  = (3.0 * math.sqrt(3.0) / 2.0) * (circ_r**2)
    wall_perim = 6.0 * circ_r
    return {
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "wall_perim": wall_perim,
        "area_cell": area_cell,
        "circ_r": circ_r,
        "apothem": apothem,
        "hex_h": hex_h
    }

def estimate_zone_mass_and_rel_density(
    r_zone_inner_m,
    r_zone_outer_m,
    stroke_h_m,
    cell_flat_m,
    wall_thk_m,
    rho_mat,
):
    """
    Approximate honeycomb mass for a circular annulus (r from r_zone_inner -> r_zone_outer)
    assuming uniform hex pitch and uniform wall_thk in that annulus.

    tile bounding box of width 2*r_zone_outer, then count cells
    whose centers fall in that annulus radius band.

    (this was the best method without ungodly calc times) - hey did you know that calc is short for calculator 

    Relative density p*/p_s ~ (solid volume in zone) / (zone volume if solid). (pretend those p's are rhos )
    """
    cm = hex_cell_metrics(cell_flat_m)
    pitch_x    = cm["pitch_x"]
    pitch_y    = cm["pitch_y"]
    wall_perim = cm["wall_perim"]

    # bounding box extents for tiling
    bbox_L = 2.0 * r_zone_outer_m
    bbox_W = 2.0 * r_zone_outer_m

    # number of cells needed in x,y with overscan
    nx = math.ceil(bbox_L / pitch_x) + NX_EXTRA
    ny = math.ceil(bbox_W / pitch_y) + NY_EXTRA

    # step offsets to roughly center the honeycomb at (0,0)
    origin_x = -0.5 * nx * pitch_x
    origin_y = -0.5 * ny * pitch_y

    total_wall_vol_m3 = 0.0
    n_cells_used = 0

    for ix in range(nx):
        for iy in range(ny):
            # hex stagger
            y_off = iy * pitch_y + (0.5 * pitch_y if (ix % 2 == 1) else 0.0)
            x_off = ix * pitch_x
            cx = origin_x + x_off
            cy = origin_y + y_off

            r_c = math.hypot(cx, cy)

            # keep if center is inside outer radius
            # and >= inner radius
            if (r_c <= r_zone_outer_m) and (r_c >= r_zone_inner_m):
                # volume of walls approximated as perimeter * wall_thk * stroke_h
                vol_one_cell = wall_perim * wall_thk_m * stroke_h_m
                total_wall_vol_m3 += vol_one_cell
                n_cells_used += 1

    # geometric volume if that ring were fully solid:
    area_zone = math.pi * (r_zone_outer_m**2 - r_zone_inner_m**2)
    vol_zone_solid = area_zone * stroke_h_m

    rel_density = (total_wall_vol_m3 / vol_zone_solid) if vol_zone_solid > 0 else 0.0
    mass_kg     = total_wall_vol_m3 * rho_mat

    return {
        "mass_kg": mass_kg,
        "rel_density": rel_density,
        "area_zone_m2": area_zone,
        "vol_walls_m3": total_wall_vol_m3,
        "n_cells_used": n_cells_used,
        "pitch_x": cm["pitch_x"],
        "pitch_y": cm["pitch_y"],
    }

def plateau_stress(pa_pf, rel_density, sigy_base_mpa):
    """
    plateau_stress* ~ PF * rel_density * SIGY
    where SIGY is base yield strength in Pa.
    """
    sigy_pa = sigy_base_mpa * 1.0e6
    return pa_pf * rel_density * sigy_pa


# =========================
# === WALL THICK MODEL ====
# =========================

def wall_thickness_from_pf(pf, cell_flat_m):
    """
    Very simple mapping:
    - wall_thk scales linearly with desired plateau factor
    - We assume PF=0.050 ⇒ ~0.50 mm wall
      and PF=0.005 ⇒ ~0.05 mm wall
    We'll clamp to >=0.1 mm for printability unless we override.

     mapping should be tuned based on real crush coumpuations - but this is good enough.
    """
    # reference:
    PF_REF_HARD = 0.050
    THK_REF_HARD_M = 0.0005  # 0.50 mm
    PF_REF_SOFT = 0.005
    THK_REF_SOFT_M = 0.00005 # 0.05 mm

    # linear interpolation in PF space:
    if pf <= PF_REF_SOFT:
        base_thk = THK_REF_SOFT_M
    elif pf >= PF_REF_HARD:
        base_thk = THK_REF_HARD_M
    else:
        # interpolate
        alpha = ((pf - PF_REF_SOFT) / (PF_REF_HARD - PF_REF_SOFT))
        base_thk = THK_REF_SOFT_M + alpha*(THK_REF_HARD_M - THK_REF_SOFT_M)

    # sanity: don't let wall go under ~0.1 mm unless you KNOW it's SLM-grade metal printing (which it won't be)
    base_thk = max(base_thk, 0.0001)

    # could also scale with cell size, e.g. thicker if huge cells
    # but we'll keep it 1:1 for now
    return base_thk


# =========================
# === SCAD WRITER ==========
# =========================

# so epic we love the SCADsss

def emit_scad(
    fname,
    stroke_h_m,
    r_total_m,
    r_inner_m,
    cell_flat_inner_m,
    wall_thk_inner_m,
    wall_thk_rim_m
):
    """
    Send OpenSCAD with the following:
    - hex wall module (main crush)
    - tiler for inner and rim zones separately w/ their wall_thk
    - cropping by cylinder radii (makes da shape)
    - global scale([1000,1000,1000]) so output comes in as mm (had some problems with output not scalling properly)
    """

    scad_txt = dedent(f"""
    //
    // le epic SCAD file via hex script
    // units: meters internally, scaled to mm at export
    //

    stroke_h_m      = {stroke_h_m:.6f};
    r_total_m       = {r_total_m:.6f};
    r_inner_m       = {r_inner_m:.6f};
    cell_flat_m     = {cell_flat_inner_m:.6f};
    wall_thk_inner  = {wall_thk_inner_m:.6f};
    wall_thk_rim    = {wall_thk_rim_m:.6f};

    // ---------- hex helpers ----------
    module hex_outline(circ_r, apothem) {{
        polygon(points=[
            [ circ_r, 0],
            [ circ_r/2,  apothem],
            [-circ_r/2,  apothem],
            [-circ_r, 0],
            [-circ_r/2, -apothem],
            [ circ_r/2, -apothem]
        ]);
    }}

    // generic hex wall for given wall_thk
    module hex_wall_generic(circ_r, apothem, wall_thk_local) {{
        difference() {{
            hex_outline(circ_r, apothem);
            offset(delta=-wall_thk_local)
                hex_outline(circ_r, apothem);
        }}
    }}

    // one extruded honeycomb cell with chosen thickness
    module one_cell_prism(wall_thk_local) {{
        apothem = cell_flat_m/2;
        circ_r  = apothem / cos(30);
        linear_extrude(height=stroke_h_m)
            hex_wall_generic(circ_r, apothem, wall_thk_local);
    }}

    // tile inner zone (soft walls)
    module tile_inner_zone() {{
        apothem = cell_flat_m/2;
        circ_r  = apothem / cos(30);
        pitch_x = 1.5 * circ_r;
        pitch_y = 2*apothem;

        // overscan boxes
        nx = ceil((2*r_total_m)/pitch_x) + {NX_EXTRA};
        ny = ceil((2*r_total_m)/pitch_y) + {NY_EXTRA};

        // center the grid at (0,0)
        x0 = -0.5*nx*pitch_x;
        y0 = -0.5*ny*pitch_y;

        for (ix=[0:nx-1]) {{
            for (iy=[0:ny-1]) {{
                y_off = iy*pitch_y + ((ix % 2 == 1) ? pitch_y/2 : 0);
                x_off = ix*pitch_x;
                cx = x0 + x_off;
                cy = y0 + y_off;
                r_c = sqrt(cx*cx + cy*cy);
                if (r_c <= r_inner_m) {{
                    translate([cx, cy, 0])
                        one_cell_prism(wall_thk_inner);
                }}
            }}
        }}
    }}

    // tile rim zone (stiffer walls)
    module tile_rim_zone() {{
        apothem = cell_flat_m/2;
        circ_r  = apothem / cos(30);
        pitch_x = 1.5 * circ_r;
        pitch_y = 2*apothem;

        nx = ceil((2*r_total_m)/pitch_x) + {NX_EXTRA};
        ny = ceil((2*r_total_m)/pitch_y) + {NY_EXTRA};

        x0 = -0.5*nx*pitch_x;
        y0 = -0.5*ny*pitch_y;

        for (ix=[0:nx-1]) {{
            for (iy=[0:ny-1]) {{
                y_off = iy*pitch_y + ((ix % 2 == 1) ? pitch_y/2 : 0);
                x_off = ix*pitch_x;
                cx = x0 + x_off;
                cy = y0 + y_off;
                r_c = sqrt(cx*cx + cy*cy);
                if ((r_c > r_inner_m) && (r_c <= r_total_m)) {{
                    translate([cx, cy, 0])
                        one_cell_prism(wall_thk_rim);
                }}
            }}
        }}
    }}

    // union of inner soft + outer stiff
    // then we scale meters->mm so OpenSCAD/Fusion/etc see real size
    scale([1000,1000,1000]) {{
        color([1,0,0,0.4]) tile_inner_zone();
        color([0,0,1,0.4]) tile_rim_zone();
    }}
    """)

    with open(fname, "w") as f:
        f.write(scad_txt)


# =========================
# === MAIN SIZING =========
# =========================

if __name__ == "__main__":

    # figure out how big the rim vs inner should be
    r_inner_m, area_total_m2, area_inner_m2 = inner_radius_from_area_frac(
        R_TOTAL_M,
        RIM_AREA_FRAC
    )
    area_rim_m2 = area_total_m2 - area_inner_m2

    # map plateau factors to wall thicknesses
    wall_thk_inner_m = wall_thickness_from_pf(PF_INNER_NOM, CELL_FLAT_INNER_M)
    wall_thk_rim_m   = wall_thickness_from_pf(PF_RIM_NOM,   CELL_FLAT_INNER_M)

    # inner zone mass / rel density
    inner_stats = estimate_zone_mass_and_rel_density(
        r_zone_inner_m=0.0,
        r_zone_outer_m=r_inner_m,
        stroke_h_m=STROKE_H_M,
        cell_flat_m=CELL_FLAT_INNER_M,
        wall_thk_m=wall_thk_inner_m,
        rho_mat=RHO_MAT
    )

    # rim zone mass / rel density
    rim_stats = estimate_zone_mass_and_rel_density(
        r_zone_inner_m=r_inner_m,
        r_zone_outer_m=R_TOTAL_M,
        stroke_h_m=STROKE_H_M,
        cell_flat_m=CELL_FLAT_INNER_M,
        wall_thk_m=wall_thk_rim_m,
        rho_mat=RHO_MAT
    )

    total_mass_kg = inner_stats["mass_kg"] + rim_stats["mass_kg"]

    # area-weighted effective relative density (p*/p_s).
    total_rel_density = (
        (inner_stats["rel_density"] * area_inner_m2) +
        (rim_stats["rel_density"]   * area_rim_m2)
    ) / (area_inner_m2 + area_rim_m2)

    # plateau stress per zone
    plateau_inner_pa = plateau_stress(PF_INNER_NOM, inner_stats["rel_density"], SIGY_BASE_MPA)
    plateau_rim_pa   = plateau_stress(PF_RIM_NOM,   rim_stats["rel_density"],   SIGY_BASE_MPA)

    # simple effective plateau stress:
    # area-weighted average (inner softer, rim stiffer)
    plateau_eff_pa = (
        plateau_inner_pa * (area_inner_m2/area_total_m2)
        + plateau_rim_pa * (area_rim_m2/area_total_m2)
    )

    # "effective" PF you'd feed to sim (so sim just sees one blob)
    # solve plateau_eff_pa ≈ PF_eff * (rho_rel_total) * SIGY
    sigy_pa = SIGY_BASE_MPA * 1.0e6
    if total_rel_density > 1e-12:
        PF_eff = plateau_eff_pa / (total_rel_density * sigy_pa)
    else:
        PF_eff = 0.0

    # force at crush and energy capacity for full stroke
    plateau_force_N   = plateau_eff_pa * area_total_m2
    energy_capacity_J = plateau_force_N * STROKE_H_M

    # export SCAD (these are just the parameters we plug)
    emit_scad(
        SCAD_FILENAME,
        STROKE_H_M,
        R_TOTAL_M,
        r_inner_m,
        CELL_FLAT_INNER_M,
        wall_thk_inner_m,
        wall_thk_rim_m
    )

    # console report
    print("[CRUSH CORE CONFIG]")
    print(f"Mode=DUAL_ZONE | Shape=cyl")
    print(f"outer_radius:          {R_TOTAL_M:.4f} m ({R_TOTAL_M*1000:.1f} mm)")
    print(f"inner_radius:          {r_inner_m:.4f} m ({r_inner_m*1000:.1f} mm)")
    print(f"stroke_h:              {STROKE_H_M:.4f} m ({STROKE_H_M*1000:.1f} mm)")
    print(f"rim_area_frac:         {RIM_AREA_FRAC:.3f}")
    print("")
    print("[CELL GEOMETRY]")
    print(f"cell_flat_inner:       {CELL_FLAT_INNER_M*1000:.2f} mm")
    print(f"wall_thk_inner:        {wall_thk_inner_m*1000:.3f} mm (PF {PF_INNER_NOM:.4f})")
    print(f"wall_thk_rim:          {wall_thk_rim_m*1000:.3f} mm (PF {PF_RIM_NOM:.4f})")
    print("")
    print("[ZONE STATS]")
    print(f"inner area:            {area_inner_m2:.4f} m^2")
    print(f"rim area:              {area_rim_m2:.4f} m^2")
    print(f"total area:            {area_total_m2:.4f} m^2")
    print(f"inner rel_density:     {inner_stats['rel_density']:.4f}")
    print(f"rim rel_density:       {rim_stats['rel_density']:.4f}")
    print(f"effective rel_density: {total_rel_density:.4f}")
    print(f"mass total:            {total_mass_kg:.3f} kg")
    print("")
    print("[PLATEAU / ENERGY]")
    print(f"plateau_inner:         {plateau_inner_pa/1e6:.3f} MPa")
    print(f"plateau_rim:           {plateau_rim_pa/1e6:.3f} MPa")
    print(f"plateau_eff (avg):     {plateau_eff_pa/1e6:.3f} MPa")
    print(f"plateau_force:         {plateau_force_N:.1f} N")
    print(f"energy_capacity:       {energy_capacity_J:.1f} J (full stroke)")
    print("")
    print("[SIM CONSTANTS -> paste into EDL sim]")
    print(f"HONEYCOMB_REL_DENSITY      = {total_rel_density:.6f}")
    print(f"HONEYCOMB_BASE_MATERIAL_SIGY_MPa = {SIGY_BASE_MPA:.1f}")
    print(f"HONEYCOMB_PLATEAU_FACTOR   = {PF_eff:.6f}")
    print(f"HONEYCOMB_FOOTPRINT_M2     = {area_total_m2:.4f}")
    print(f"HONEYCOMB_STROKE_M         = {STROKE_H_M:.4f}")
    print("")
    print("[FILE OUTPUT]")
    print(f'Wrote OpenSCAD -> "{SCAD_FILENAME}"')
    print("Open SCAD, F6 (CGAL render), export STL. The part should appear ~0.8 m wide (or whatever the val is) because we scale m→mm in the file.")
