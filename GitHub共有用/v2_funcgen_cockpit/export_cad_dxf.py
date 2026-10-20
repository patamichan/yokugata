"""Export Real-Scale Fairing Airfoil to Industry-Standard CAD DXF Format.

Features:
- AutoCAD DXF (R2010 compatible)
- Native Millimeter units ($INSUNITS = 4)
- Structured Layers:
  - 0_AIRFOIL_POLYLINE: High-resolution closed outline for extrusion/sketches
  - 1_AIRFOIL_SPLINE: Smooth CAD spline curve
  - 2_CENTERLINE: Chord center line
  - 3_PILOT_ENVELOPE: Pilot clearance boundaries at Feet, Knees, Hips, Shoulders
  - 4_DIMENSIONS: Text annotations and station markers
"""
import os
from pathlib import Path
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment

from core.geometry import generate_funcgen_airfoil
from core.paths import CAD_DIR
from core.optimizer import calculate_required_chord, PRESET_PILOT_STANDARD, _profile_arrays

BASE_DIR = Path(__file__).resolve().parent

def create_fairing_dxf(chord_mm: float, tc: float, xt: float, kle: float, method: str,
                       filepath: Path, pilot_profile: list[dict], nose_margin_mm: float = 30.0):
    """Create a fully featured, ready-to-import DXF file."""
    # Generate high-resolution Selig points
    x_norm, y_norm = generate_funcgen_airfoil(tc, xt, kle, num_points=1001, method=method)
    x_mm = x_norm * chord_mm
    y_mm = y_norm * chord_mm
    
    # Create DXF Document
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4  # 4 = Millimeters
    doc.header["$MEASUREMENT"] = 1  # 1 = Metric
    
    # Setup Layers
    doc.layers.add("0_AIRFOIL_OUTLINE", color=5, lineweight=35)      # Blue, 0.35mm
    doc.layers.add("1_AIRFOIL_SPLINE", color=4, lineweight=25)       # Cyan, 0.25mm
    doc.layers.add("2_CENTERLINE", color=1, linetype="CENTER")       # Red, Centerline
    doc.layers.add("3_PILOT_CLEARANCE", color=3, lineweight=25)      # Green, 0.25mm
    doc.layers.add("4_ANNOTATIONS", color=7)                         # White/Black text
    
    msp = doc.modelspace()
    
    # 1. Closed Polyline for Outline (ideal for 3D extrusion/lofting)
    points_2d = list(zip(x_mm, y_mm))
    msp.add_lwpolyline(points_2d, close=True, dxfattribs={"layer": "0_AIRFOIL_OUTLINE"})
    
    # 2. Smooth Spline representation (decimated points for clean CAD fit)
    spline_step = 5  # ~200 points for clean B-spline fit in CAD
    spline_pts = list(zip(x_mm[::spline_step], y_mm[::spline_step]))
    if spline_pts[-1] != spline_pts[0]:
        spline_pts.append(spline_pts[0])
    msp.add_spline(spline_pts, dxfattribs={"layer": "1_AIRFOIL_SPLINE"})
    
    # 3. Centerline
    msp.add_line((0.0, 0.0), (chord_mm, 0.0), dxfattribs={"layer": "2_CENTERLINE"})
    
    # 4. Pilot Stations & Clearance Envelopes
    distances, widths = _profile_arrays(pilot_profile, nose_margin_mm=nose_margin_mm)
    for p, d, w in zip(pilot_profile, distances, widths):
        body_w = p["width_mm"]
        clr = p.get("clearance_mm", 0.0)
        
        # Pilot body segment (Solid line)
        msp.add_line((d, -body_w / 2.0), (d, body_w / 2.0),
                     dxfattribs={"layer": "3_PILOT_CLEARANCE", "color": 1}) # Red body
        
        # Clearance boundary rectangle
        rect_pts = [
            (d - 15.0, -w / 2.0),
            (d + 15.0, -w / 2.0),
            (d + 15.0, w / 2.0),
            (d - 15.0, w / 2.0),
        ]
        msp.add_lwpolyline(rect_pts, close=True,
                           dxfattribs={"layer": "3_PILOT_CLEARANCE", "color": 3})
        
        # Annotation text
        label_text = f"{p['name']}: Req {w:.0f}mm (Body {body_w:.0f} + Clr {clr*2:.0f})"
        msp.add_text(label_text, dxfattribs={"layer": "4_ANNOTATIONS", "height": 18.0}).set_placement(
            (d + 20.0, w / 2.0 + 10.0), align=TextEntityAlignment.BOTTOM_LEFT
        )
    
    # 5. Header / Title Block
    title_text = (f"BIRDMAN FAIRING CAD - {method.upper()} tc={tc:.1f}% xt={xt:.1f}% "
                  f"kle={kle:.2f} | Chord={chord_mm:.1f}mm")
    msp.add_text(title_text, dxfattribs={"layer": "4_ANNOTATIONS", "height": 30.0}).set_placement(
        (0.0, chord_mm * (tc / 100.0) / 2.0 + 80.0), align=TextEntityAlignment.BOTTOM_LEFT
    )
    
    doc.saveas(filepath)
    print(f"Generated DXF: {filepath}")

def main():
    print("=== Generating Production CAD DXF Files ===")
    
    tc = 24.0
    xt = 42.0
    nose_margin_mm = 30.0
    
    CAD_DIR.mkdir(parents=True, exist_ok=True)
    packages = sorted(list(CAD_DIR.glob("CAD用_推奨実寸確認パッケージ_*")))
    target_dir = packages[-1] if packages else CAD_DIR
    
    # 1. Recommended (kle=1.2, CST)
    c_rec = calculate_required_chord(tc, xt, 1.2, PRESET_PILOT_STANDARD,
                                     nose_margin_mm=nose_margin_mm, generator_method="cst")
    rec_dxf = target_dir / "Recommended_kle1.2" / "fairing_recommended_realscale_mm.dxf"
    create_fairing_dxf(c_rec, tc, xt, 1.2, "cst", rec_dxf, PRESET_PILOT_STANDARD, nose_margin_mm)
    
    # Also place directly in target_dir for instant 1-click access
    root_rec_dxf = target_dir / "【CADで直接開く】推奨翼型_実寸_kle1.2.dxf"
    create_fairing_dxf(c_rec, tc, xt, 1.2, "cst", root_rec_dxf, PRESET_PILOT_STANDARD, nose_margin_mm)
    
    # Project root copy for maximum convenience
    proj_dxf = CAD_DIR / "【CAD用実寸】コックピットフェアリング_CST_kle1.2.dxf"
    create_fairing_dxf(c_rec, tc, xt, 1.2, "cst", proj_dxf, PRESET_PILOT_STANDARD, nose_margin_mm)
    
    # 2. Baseline Comparison (kle=1.0, CST)
    c_base = calculate_required_chord(tc, xt, 1.0, PRESET_PILOT_STANDARD,
                                      nose_margin_mm=nose_margin_mm, generator_method="cst")
    base_dxf = target_dir / "Baseline_kle1.0" / "fairing_baseline_realscale_mm.dxf"
    create_fairing_dxf(c_base, tc, xt, 1.0, "cst", base_dxf, PRESET_PILOT_STANDARD, nose_margin_mm)
    
    print("\nAll CAD DXF files successfully created!")

if __name__ == '__main__':
    main()
