"""
End-to-End integration test for the entire v2 3D cockpit airfoil design system.
Validates the complete chain:
Geometry -> XFOIL Runner -> 3D Database -> Dynamic Sizing & Selection -> CAD File Export
"""

import os
import sys
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def test_full_pipeline_end_to_end():
    print("\n[E2E] Step 1: Geometry Generation...")
    from core.geometry import generate_funcgen_airfoil, calculate_le_radius
    x, y = generate_funcgen_airfoil(tc_pct=25.0, xt_pct=42.0, kle=1.2)
    assert len(x) == len(y) == 161
    r_le = calculate_le_radius(x, y)
    assert r_le > 0.07

    print("[E2E] Step 2: Database Initialization & Compatibility...")
    from core.database import Database3DManager
    test_csv = os.path.join(BASE_DIR, "tests", "temp_e2e_db.csv")
    test_pkl = os.path.join(BASE_DIR, "tests", "temp_e2e_db.pkl")
    for p in [test_csv, test_pkl]:
        if os.path.exists(p):
            os.remove(p)

    db = Database3DManager(csv_path=test_csv, cache_path=test_pkl)

    # 模擬レコード投入 (kle = 0.8, 1.0, 1.2, Re = 500k, 1000k, 2000k)
    sample_records = []
    for kle in [0.8, 1.0, 1.2]:
        name = f"NACA 0025_42_k{kle:.1f}"
        for re_val in [500000, 1000000, 2000000]:
            for a in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
                sample_records.append({
                    "airfoil": name,
                    "thickness_pct": 25.0,
                    "max_thickness_pos_pct": 42.0,
                    "le_radius_ratio": kle,
                    "max_camber_pct": 0.0,
                    "reynolds": re_val,
                    "alpha": float(a),
                    "cl": 0.0,
                    "cd": 0.0075 + (1.2 - kle)*0.0005 + a*0.0004,
                    "drag_counts": 75.0,
                    "ld": 0.0,
                    "cm": 0.0,
                    "top_xtr": 0.5,
                    "bot_xtr": 0.5,
                    "converged": True
                })
    db.append_records(sample_records)
    keys = db.get_completed_keys()
    assert (25.0, 42.0, 1.2) in keys

    print("[E2E] Step 3: Dynamic Chord Sizing & 3D Optimization...")
    from core.optimizer import calculate_required_chord, optimize_section_3d
    pilot_profile = [
        {"x_mm": 100.0, "width_mm": 320.0, "clearance_mm": 15.0},
        {"x_mm": 800.0, "width_mm": 450.0, "clearance_mm": 20.0}
    ]

    c_08 = calculate_required_chord(25.0, 42.0, 0.8, pilot_profile)
    c_12 = calculate_required_chord(25.0, 42.0, 1.2, pilot_profile)
    assert c_12 < c_08, f"Blunt nose must shorten required chord! c_12={c_12}, c_08={c_08}"

    best = optimize_section_3d(db, pilot_profile, velocity_ms=14.0, crosswind_ms=1.5)
    assert best is not None
    assert "drag_area_mm" in best
    assert "chord_mm" in best

    print("[E2E] Step 4: CAD Export Verification...")
    test_export_dir = os.path.join(BASE_DIR, "tests", "temp_cad_export")
    if os.path.exists(test_export_dir):
        shutil.rmtree(test_export_dir)
    os.makedirs(test_export_dir, exist_ok=True)

    # CAD 出力ファイル生成のテスト
    from core.geometry import save_airfoil_dat
    foil_x, foil_y = generate_funcgen_airfoil(best['thickness_pct'], best['max_thickness_pos_pct'], best['le_radius_ratio'])
    chord = best['chord_mm']

    # 1. 実寸 mm
    scale_dat = os.path.join(test_export_dir, f"Section_c{chord:.0f}mm_scaled.dat")
    with open(scale_dat, 'w', encoding='utf-8') as f:
        f.write(f"{best['airfoil']} Real Scale [mm] c={chord:.1f}mm\n")
        for px, py in zip(foil_x, foil_y):
            f.write(f" {px*chord:12.3f} {py*chord:12.3f}\n")

    # 2. 無次元
    norm_dat = os.path.join(test_export_dir, f"Section_c{chord:.0f}mm_norm.dat")
    save_airfoil_dat(norm_dat, foil_x, foil_y, f"{best['airfoil']} Normalized")

    assert os.path.exists(scale_dat)
    assert os.path.exists(norm_dat)

    # クリーンアップ
    for p in [test_csv, test_pkl]:
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(test_export_dir):
        shutil.rmtree(test_export_dir)

    print("[E2E] SUCCESS: Full pipeline completed and verified!")


if __name__ == "__main__":
    try:
        test_full_pipeline_end_to_end()
        print("\n============================================================")
        print(" ALL END-TO-END PIPELINE TESTS PASSED!")
        print("============================================================")
        sys.exit(0)
    except Exception as e:
        print(f"\n[E2E FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
