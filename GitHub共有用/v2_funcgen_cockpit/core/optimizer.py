"""Pilot-envelope sizing and measured-polar drag-area minimization.

Lengths are millimetres; angles are degrees. Reynolds numbers use the v1
kinematic viscosity convention, 1.51e-5 m²/s. Reynolds queries are clamped to
measured boundaries; unsupported angular ranges are rejected.
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from core.geometry import generate_funcgen_airfoil

KINEMATIC_VISCOSITY = 1.51e-5
GAUSSIAN_LIMIT = 2.807034
PRESET_PILOT_STANDARD = [
    {'name': 'Feet/Pedal', 'x_mm': 100.0, 'width_mm': 320.0, 'clearance_mm': 20.0},
    {'name': 'Knees', 'x_mm': 500.0, 'width_mm': 380.0, 'clearance_mm': 20.0},
    {'name': 'Hips', 'x_mm': 900.0, 'width_mm': 420.0, 'clearance_mm': 20.0},
    {'name': 'Shoulders', 'x_mm': 1200.0, 'width_mm': 460.0, 'clearance_mm': 25.0},
]


def _profile_arrays(pilot_profile, nose_margin_mm):
    if not pilot_profile:
        raise ValueError('At least one pilot profile point is required')
    if not math.isfinite(nose_margin_mm) or nose_margin_mm < 0:
        raise ValueError('nose_margin_mm must be finite and nonnegative')
    rows = np.asarray([[p['x_mm'], p['width_mm'], p.get('clearance_mm', 0.0)]
                       for p in pilot_profile], dtype=float)
    if not np.isfinite(rows).all() or np.any(rows < 0):
        raise ValueError('Profile dimensions must be finite and nonnegative')
    distances = rows[:, 0] + nose_margin_mm
    widths = rows[:, 1] + 2 * rows[:, 2]
    if np.any((distances == 0) & (widths > 0)):
        raise ValueError('A positive width cannot fit at the closed nose')
    return distances, widths


def calculate_required_chord(tc_pct: float, xt_pct: float, kle: float,
                             pilot_profile: list[dict], nose_margin_mm: float = 30.0,
                             tolerance_mm: float = 0.5, generator_method: str = 'warp') -> float:
    """Return a feasible minimum chord to within tolerance_mm (upper bracket).

    Sample densely and interpolate against sqrt(x) to resolve the rounded nose
    without a linear approximation of its singular dy/dx. The profile refers
    to distances from the nose, not fractions of the final chord.
    """
    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError('tolerance_mm must be finite and positive')
    distances, widths = _profile_arrays(pilot_profile, nose_margin_mm)
    x, y = generate_funcgen_airfoil(tc_pct, xt_pct, kle, num_points=2001, method=generator_method)
    nose = int(np.argmin(x))
    surface = PchipInterpolator(np.sqrt(x[:nose+1][::-1]), y[:nose+1][::-1],
                                extrapolate=False)
    def fits(chord):
        if chord <= 0 or np.any(distances > chord):
            return False
        available = 2 * chord * surface(np.sqrt(distances / chord))
        return bool(np.all(available >= widths))
    lower = float(np.max(distances))
    if fits(lower):
        return lower
    upper = max(lower * 2, float(np.max(widths)) / (tc_pct / 100), 1.0)
    for _ in range(80):
        if fits(upper):
            break
        lower, upper = upper, upper * 2
    else:
        raise ValueError('Could not bracket a feasible chord')
    for _ in range(200):
        if upper - lower <= tolerance_mm:
            return float(upper)
        middle = (lower + upper) / 2
        if fits(middle):
            upper = middle
        else:
            lower = middle
    raise ValueError('Chord tolerance is below numerical resolution')


def _wind_limit(velocity_ms, crosswind_ms, max_alpha_deg):
    if not all(math.isfinite(v) for v in (velocity_ms, crosswind_ms, max_alpha_deg)):
        raise ValueError('Wind and angle parameters must be finite')
    if velocity_ms <= 0 or crosswind_ms < 0 or max_alpha_deg < 0:
        raise ValueError('Velocity must be positive; crosswind and angle limit nonnegative')
    angle = math.degrees(math.atan2(crosswind_ms, velocity_ms))
    if crosswind_ms > .001 and angle > max_alpha_deg:
        raise ValueError('Crosswind angle exceeds the supported max_alpha_deg')
    return angle


def calculate_expected_cd(cd_lookup_func, velocity_ms: float = 14.0,
                          crosswind_ms: float = 1.5, max_alpha_deg: float = 12.0) -> float:
    """Integrate the unrenormalized Gaussian over its central 99.5% interval.

    This follows the supplied integral literally. Consequently constant Cd
    yields approximately 0.995*Cd for nonzero wind. Reject unsupported ranges
    instead of clipping the wind distribution or replacing missing Cd values.
    """
    angle = _wind_limit(velocity_ms, crosswind_ms, max_alpha_deg)
    def lookup(alpha):
        value = float(cd_lookup_func(float(alpha)))
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Cd lookup must provide finite positive coefficients')
        return value
    if crosswind_ms <= .001:
        return lookup(0.0)
    sigma = angle / GAUSSIAN_LIMIT
    # Validate both boundaries; quadrature itself need not sample endpoints.
    lookup(-angle)
    lookup(angle)
    def integrand(z):
        return lookup(sigma*z) * math.exp(-z*z/2) / math.sqrt(2*math.pi)
    return float(quad(integrand, -GAUSSIAN_LIMIT, GAUSSIAN_LIMIT,
                      epsabs=1e-10, epsrel=1e-7, limit=200)[0])


def _polar_lookup(group, reynolds, angle):
    """Interpolate alpha at the two bracketing Re, then interpolate Re.

    Reflect positive-only polars because geometry is symmetric. Require full
    angular coverage at each bracketing Re. Out-of-range Reynolds queries use
    the nearest measured boundary; angular extrapolation is not performed.
    """
    levels = np.sort(group.reynolds.unique())
    if not len(levels) or not np.isfinite(levels).all() or not math.isfinite(reynolds):
        raise ValueError('Finite Reynolds data is required')
    reynolds = float(np.clip(reynolds, levels[0], levels[-1]))
    right = int(np.searchsorted(levels, reynolds))
    selected = [levels[right]] if levels[right] == reynolds else [levels[right-1], levels[right]]
    curves = []
    for re in selected:
        polar = group.loc[group.reynolds == re, ['alpha', 'cd']].copy()
        # Symmetry permits data from either sign to represent |alpha|.
        polar['alpha'] = polar.alpha.abs()
        polar = polar.groupby('alpha', as_index=False).cd.mean().sort_values('alpha')
        aa, cc = polar.alpha.to_numpy(), polar.cd.to_numpy()
        if aa[0] > 0 or aa[-1] < angle or not np.isfinite(cc).all() or np.any(cc <= 0):
            raise ValueError('Polar does not cover the requested crosswind interval')
        curves.append((aa, cc))
    def lookup(alpha):
        values = [np.interp(abs(alpha), aa, cc) for aa, cc in curves]
        return float(values[0] if len(values) == 1 else np.interp(reynolds, selected, values))
    return lookup


def optimize_section_3d(db_manager, pilot_profile: list[dict], velocity_ms: float = 14.0,
                         crosswind_ms: float = 1.5, max_chord_mm: float = None,
                         generator_method: str = 'warp', nose_margin_mm: float = 30.0) -> dict:
    """Return the best supported completed foil, or {} if none is feasible.

    Database must already be loaded. The default geometry generator is used;
    the database must contain polars generated with that same geometry method.
    Results expose physical/evaluated Reynolds numbers and a clamping flag.
    """
    angle = _wind_limit(velocity_ms, crosswind_ms, 12.0)
    _profile_arrays(pilot_profile, nose_margin_mm)
    if max_chord_mm is not None and (not math.isfinite(max_chord_mm) or max_chord_mm <= 0):
        raise ValueError('max_chord_mm must be finite and positive')
    if generator_method not in ('warp', 'cst'):
        raise ValueError('generator_method must be warp or cst')
    completed = (db_manager.get_completed_keys() if generator_method == 'warp'
                 else db_manager.get_completed_keys(generator_method=generator_method))
    if not completed or db_manager.df.empty:
        return {}
    frame = db_manager.df
    if 'generator_method' in frame:
        frame = frame.loc[frame.generator_method == generator_method]
    elif generator_method != 'warp':
        return {}
    best = {}
    columns = ['airfoil', 'thickness_pct', 'max_thickness_pos_pct', 'le_radius_ratio']
    for (name, tc, xt, kle), group in frame.groupby(columns, sort=True):
        if (round(tc, 1), round(xt, 1), round(kle, 2)) not in completed:
            continue
        if group.alpha.nunique() < 5:
            continue
        try:
            chord = calculate_required_chord(tc, xt, kle, pilot_profile,
                                              nose_margin_mm=nose_margin_mm,
                                              generator_method=generator_method)
            if max_chord_mm is not None and chord > max_chord_mm:
                continue
            reynolds = velocity_ms * (chord / 1000) / KINEMATIC_VISCOSITY
            lookup = _polar_lookup(group, reynolds, 0.0 if crosswind_ms <= .001 else angle)
            cd_eff = calculate_expected_cd(lookup, velocity_ms, crosswind_ms)
        except ValueError:
            continue
        area = chord * cd_eff
        if not best or area < best['drag_area_mm']:
            best = dict(airfoil=name, thickness_pct=float(tc), max_thickness_pos_pct=float(xt),
                        le_radius_ratio=float(kle), chord_mm=chord, cd_eff=cd_eff,
                        drag_area_mm=area, generator_method=generator_method, reynolds=reynolds,
                        reynolds_evaluated=float(np.clip(reynolds, group.reynolds.min(),
                                                         group.reynolds.max())),
                        reynolds_clamped=bool(reynolds < group.reynolds.min() or
                                              reynolds > group.reynolds.max()))
    return best


def optimize_nose_margin_3d(db_manager, pilot_profile: list[dict],
                            velocity_ms: float = 14.0, crosswind_ms: float = 1.5,
                            max_chord_mm: float = None, generator_method: str = 'warp',
                            margin_min_mm: float = 20.0, margin_max_mm: float = 150.0,
                            margin_step_mm: float = 10.0,
                            progress_callback=None) -> dict:
    """Sweep and automatically find the optimal nose margin that minimizes drag area.

    Evaluates the tradeoff between:
    - Small margin: Shorter overall nose, but tight foot clearance forces fatter/blunter foils.
    - Large margin: Foot moves back into wider foil region enabling thinner/sharper foils,
      balanced against chord growth.

    Returns:
        dict containing:
        - 'best_result': The overall optimal selection dict
        - 'best_margin_mm': The optimal nose margin
        - 'sweep_history': List of dicts for each evaluated margin
    """
    if margin_min_mm < 0 or margin_max_mm <= margin_min_mm or margin_step_mm <= 0:
        raise ValueError("Invalid nose margin sweep range")

    margins = np.arange(margin_min_mm, margin_max_mm + 1e-6, margin_step_mm)
    total_steps = len(margins)
    history = []
    global_best = None

    for idx, margin in enumerate(margins, start=1):
        margin_val = float(round(margin, 2))
        res = optimize_section_3d(db_manager, pilot_profile, velocity_ms=velocity_ms,
                                  crosswind_ms=crosswind_ms, max_chord_mm=max_chord_mm,
                                  generator_method=generator_method, nose_margin_mm=margin_val)
        if res:
            diag = analyze_envelope_bottlenecks(res['thickness_pct'], res['max_thickness_pos_pct'],
                                                res['le_radius_ratio'], pilot_profile,
                                                nose_margin_mm=margin_val,
                                                generator_method=generator_method)
            entry = {
                'nose_margin_mm': margin_val,
                'airfoil': res['airfoil'],
                'thickness_pct': res['thickness_pct'],
                'max_thickness_pos_pct': res['max_thickness_pos_pct'],
                'le_radius_ratio': res['le_radius_ratio'],
                'chord_mm': res['chord_mm'],
                'cd_eff': res['cd_eff'],
                'drag_area_mm': res['drag_area_mm'],
                'bottlenecks': diag['bottlenecks'],
                'full_result': res
            }
            history.append(entry)
            if global_best is None or res['drag_area_mm'] < global_best['drag_area_mm']:
                global_best = dict(res)
                global_best['nose_margin_mm'] = margin_val
                global_best['bottlenecks'] = diag['bottlenecks']

        if progress_callback:
            progress_callback(idx, total_steps, margin_val, res)

    if not history or not global_best:
        return {}

    return {
        'best_result': global_best,
        'best_margin_mm': global_best['nose_margin_mm'],
        'sweep_history': history
    }


def load_pilot_profile_csv(filepath) -> list[dict]:
    """Load an arbitrary pilot or component envelope from a CSV/text file.

    Supports hierarchical height levels (e.g. height_level, z_mm), arbitrary station names,
    flexible header variations, comment lines (#), and validates non-negative dimensions.

    Example CSV formats:
        # 1. Hierarchical with height levels:
        # height_level, z_mm, station_name, x_mm, width_mm, clearance_mm
        1, 100.0, Pedal_Bottom, 150.0, 320.0, 20.0
        2, 350.0, Crank_Axis, 370.0, 380.0, 20.0
        3, 600.0, Shoulders, 1290.0, 510.0, 25.0

        # 2. Flat / Legacy 2D:
        # station_name, x_mm, width_mm, clearance_mm
        Pedal_Tip, 100.0, 320.0, 20.0
        Shoulders, 1200.0, 460.0, 25.0
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Profile file not found: {filepath}")

    points = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = None
        for line_num, row in enumerate(reader, start=1):
            row = [cell.strip() for cell in row]
            if not row or not any(row):
                continue

            # Check if this row looks like a header (even if preceded by #)
            row_joined = " ".join(row).lower()
            if any(key in row_joined for key in ("station", "width", "clearance", "x_mm", "dist", "level", "z_mm", "height")):
                header = [c.lstrip("#").strip().lower().replace(" ", "_") for c in row]
                continue

            # Pure comment line without header keywords
            if row[0].startswith("#"):
                continue

            name = f"Station_{len(points)+1}"
            x_val = None
            w_val = None
            clr_val = 20.0  # Default clearance
            level_val = 1
            z_val = 0.0

            if header is not None:
                row_dict = {}
                for col_idx, col_name in enumerate(header):
                    if col_idx < len(row):
                        row_dict[col_name] = row[col_idx]

                for k, v in row_dict.items():
                    if not v:
                        continue
                    if k in ("level", "height_level", "level_id", "layer", "layer_id", "z_level", "section"):
                        try:
                            level_val = int(v)
                        except ValueError:
                            level_val = str(v)
                    elif k in ("z", "z_mm", "height", "height_mm", "wl", "wl_mm", "z_pos"):
                        z_val = float(v)
                    elif k in ("name", "station", "station_name", "part", "location"):
                        name = str(v)
                    elif k in ("x", "x_mm", "x_pos", "distance", "dist_mm", "dist"):
                        x_val = float(v)
                    elif k in ("width", "width_mm", "w", "w_mm", "breadth"):
                        w_val = float(v)
                    elif k in ("clearance", "clearance_mm", "clr", "clr_mm", "margin"):
                        clr_val = float(v)

            # Fallback to positional if headers were missing or incomplete
            if x_val is None:
                if len(row) >= 6:
                    # level, z_mm, name, x_mm, width_mm, clearance_mm
                    try:
                        level_val = int(row[0])
                    except ValueError:
                        level_val = str(row[0])
                    z_val = float(row[1])
                    name = row[2]
                    x_val = float(row[3])
                    w_val = float(row[4])
                    clr_val = float(row[5])
                elif len(row) == 5:
                    # Check if first is level or name
                    try:
                        level_val = int(row[0])
                        # level, name, x_mm, width_mm, clearance_mm
                        name = row[1]
                        x_val = float(row[2])
                        w_val = float(row[3])
                        clr_val = float(row[4])
                    except ValueError:
                        # name, x_mm, width_mm, clearance_mm, z_mm
                        name = row[0]
                        x_val = float(row[1])
                        w_val = float(row[2])
                        clr_val = float(row[3])
                        z_val = float(row[4])
                elif len(row) >= 3:
                    # name, x_mm, width_mm, [clearance_mm]
                    name = row[0]
                    x_val = float(row[1])
                    w_val = float(row[2])
                    if len(row) >= 4 and row[3]:
                        clr_val = float(row[3])

            if x_val is None or w_val is None:
                raise ValueError(f"Line {line_num} in {path.name}: Could not parse x and width values")
            if not (math.isfinite(x_val) and math.isfinite(w_val) and math.isfinite(clr_val) and math.isfinite(z_val)):
                raise ValueError(f"Line {line_num} in {path.name}: Non-finite dimension detected")
            if x_val < 0 or w_val < 0 or clr_val < 0 or z_val < 0:
                raise ValueError(f"Line {line_num} in {path.name}: Dimensions must be non-negative")

            points.append({
                "level": level_val,
                "z_mm": float(z_val),
                "name": name,
                "x_mm": float(x_val),
                "width_mm": float(w_val),
                "clearance_mm": float(clr_val)
            })

    if not points:
        raise ValueError(f"No valid dimension stations found in {filepath}")

    # Sort stations: primary by level, secondary by x_mm
    def sort_key(p):
        lvl = p["level"]
        lvl_num = lvl if isinstance(lvl, (int, float)) else 999
        return (lvl_num, str(lvl), p["x_mm"])

    points.sort(key=sort_key)

    # Sanitize duplicates within the SAME level and SAME x_mm
    merged = []
    for p in points:
        if merged and merged[-1]["level"] == p["level"] and abs(p["x_mm"] - merged[-1]["x_mm"]) < 1e-3:
            prev_eff = merged[-1]["width_mm"] + 2.0 * merged[-1]["clearance_mm"]
            curr_eff = p["width_mm"] + 2.0 * p["clearance_mm"]
            if curr_eff > prev_eff:
                merged[-1] = {
                    "level": p["level"],
                    "z_mm": p["z_mm"],
                    "name": f"{merged[-1]['name']}/{p['name']}",
                    "x_mm": merged[-1]["x_mm"],
                    "width_mm": p["width_mm"],
                    "clearance_mm": p["clearance_mm"]
                }
        else:
            merged.append(p)

    return merged


def group_profile_by_level(profile: list[dict]) -> dict:
    """Group profile stations by their height level.

    Returns:
        dict: {level_identifier: [station_dict, ...], ...}
    """
    levels = {}
    for p in profile:
        lvl = p.get("level", 1)
        if lvl not in levels:
            levels[lvl] = []
        levels[lvl].append(p)
    return levels


def get_profile_level_summary(profile: list[dict]) -> list[dict]:
    """Return a structured summary of each height level in the profile."""
    grouped = group_profile_by_level(profile)
    summary = []
    for lvl, stations in grouped.items():
        z_vals = [s.get("z_mm", 0.0) for s in stations]
        max_req_w = max(s["width_mm"] + 2.0 * s.get("clearance_mm", 0.0) for s in stations)
        summary.append({
            "level": lvl,
            "z_mm_mean": float(np.mean(z_vals)),
            "z_mm_min": float(np.min(z_vals)),
            "z_mm_max": float(np.max(z_vals)),
            "station_count": len(stations),
            "station_names": [s["name"] for s in stations],
            "max_required_width_mm": max_req_w,
            "stations": stations
        })
    return summary


def flatten_profile_projection(profile: list[dict]) -> list[dict]:
    """Flatten a multi-level 3D profile into a 2D horizontal maximum projection envelope.

    Merges all height levels into a single horizontal envelope by picking the maximum
    effective width (width + 2*clearance) for any closely-spaced x locations.
    Preserves origin level tag in station names for transparency.
    """
    if not profile:
        return []
    # Sort purely by x_mm
    sorted_pts = sorted(profile, key=lambda p: p["x_mm"])
    flattened = []
    for p in sorted_pts:
        lvl_tag = f"[LV{p.get('level', 1)}]"
        name_with_lvl = f"{p['name']} {lvl_tag}" if lvl_tag not in p['name'] else p['name']
        entry = dict(p)
        entry["name"] = name_with_lvl

        if flattened and abs(entry["x_mm"] - flattened[-1]["x_mm"]) < 10.0:
            prev_eff = flattened[-1]["width_mm"] + 2.0 * flattened[-1].get("clearance_mm", 0.0)
            curr_eff = entry["width_mm"] + 2.0 * entry.get("clearance_mm", 0.0)
            if curr_eff > prev_eff:
                flattened[-1] = entry
        else:
            flattened.append(entry)
    return flattened


def save_pilot_profile_csv(profile: list[dict], filepath):
    """Save pilot envelope to an easy-to-read CSV file with height level hierarchy."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["height_level", "z_mm", "station_name", "x_mm", "width_mm", "clearance_mm"])
        for p in profile:
            writer.writerow([
                p.get("level", 1),
                f"{p.get('z_mm', 0.0):.1f}",
                p.get("name", "Station"),
                f"{p.get('x_mm', 0.0):.1f}",
                f"{p.get('width_mm', 0.0):.1f}",
                f"{p.get('clearance_mm', 20.0):.1f}"
            ])


def analyze_envelope_bottlenecks(tc_pct: float, xt_pct: float, kle: float,
                                 pilot_profile: list[dict], nose_margin_mm: float = 30.0,
                                 generator_method: str = "warp") -> dict:
    """Analyze which pilot station is the binding constraint on chord length.

    Computes:
    - Overall required chord (c*)
    - Individual standalone required chord for each station (c_i)
    - Margin at each station with height level tagging
    - Level-by-level bottleneck summaries
    - Identification of the critical bottleneck station(s)
    """
    overall_chord = calculate_required_chord(tc_pct, xt_pct, kle, pilot_profile,
                                             nose_margin_mm=nose_margin_mm,
                                             generator_method=generator_method)

    distances, widths = _profile_arrays(pilot_profile, nose_margin_mm=nose_margin_mm)
    x, y = generate_funcgen_airfoil(tc_pct, xt_pct, kle, num_points=2001, method=generator_method)
    nose = int(np.argmin(x))
    surface = PchipInterpolator(np.sqrt(x[:nose+1][::-1]), y[:nose+1][::-1], extrapolate=False)

    avail_widths = 2.0 * overall_chord * surface(np.sqrt(distances / overall_chord))
    margins = avail_widths - widths

    station_results = []
    for idx, p in enumerate(pilot_profile):
        single_profile = [p]
        chord_single = calculate_required_chord(tc_pct, xt_pct, kle, single_profile,
                                                nose_margin_mm=nose_margin_mm,
                                                generator_method=generator_method)
        is_bottleneck = bool(abs(chord_single - overall_chord) <= 1.0 or margins[idx] <= 0.1)
        station_results.append({
            "name": p.get("name", f"Station_{idx+1}"),
            "level": p.get("level", 1),
            "z_mm": float(p.get("z_mm", 0.0)),
            "x_dist_mm": float(distances[idx]),
            "required_width_mm": float(widths[idx]),
            "body_width_mm": float(p.get("width_mm", 0.0)),
            "clearance_mm": float(p.get("clearance_mm", 0.0)),
            "available_width_mm": float(avail_widths[idx]),
            "margin_mm": float(margins[idx]),
            "standalone_chord_mm": float(chord_single),
            "is_bottleneck": is_bottleneck
        })

    bottleneck_names = [s["name"] for s in station_results if s["is_bottleneck"]]

    # Hierarchical level diagnostic grouping
    level_groups = {}
    for s in station_results:
        lvl = s["level"]
        if lvl not in level_groups:
            level_groups[lvl] = {
                "level": lvl,
                "z_mm": s["z_mm"],
                "stations": [],
                "min_margin_mm": 1e9,
                "max_standalone_chord_mm": 0.0,
                "critical_station": None
            }
        grp = level_groups[lvl]
        grp["stations"].append(s)
        if s["margin_mm"] < grp["min_margin_mm"]:
            grp["min_margin_mm"] = s["margin_mm"]
        if s["standalone_chord_mm"] > grp["max_standalone_chord_mm"]:
            grp["max_standalone_chord_mm"] = s["standalone_chord_mm"]
            grp["critical_station"] = s["name"]

    return {
        "overall_chord_mm": overall_chord,
        "tc_pct": tc_pct,
        "xt_pct": xt_pct,
        "kle": kle,
        "generator_method": generator_method,
        "stations": station_results,
        "level_diagnostics": level_groups,
        "bottlenecks": bottleneck_names
    }


