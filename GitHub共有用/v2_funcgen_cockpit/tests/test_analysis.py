"""
Unit tests for core.xfoil_runner and core.database (3D design space)
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def test_format_foil_name_3d():
    from core.xfoil_runner import format_foil_name_3d
    name1 = format_foil_name_3d(25.0, 42.0, 1.0)
    assert name1 == "NACA 0025_42_k1.0", f"Unexpected: {name1}"

    name2 = format_foil_name_3d(24.5, 41.5, 0.85)
    assert name2 == "NACA 0024.5_41.5_k0.85", f"Unexpected: {name2}"


def test_database_manager_basic():
    from core.database import Database3DManager
    test_csv = os.path.join(BASE_DIR, "tests", "temp_test_db.csv")
    test_pkl = os.path.join(BASE_DIR, "tests", "temp_test_db.pkl")

    for p in [test_csv, test_pkl]:
        if os.path.exists(p):
            os.remove(p)

    db = Database3DManager(csv_path=test_csv, cache_path=test_pkl)

    # テスト用ダミーレコード
    sample_records = [
        {
            "airfoil": "NACA 0025_42_k1.0",
            "thickness_pct": 25.0,
            "max_thickness_pos_pct": 42.0,
            "le_radius_ratio": 1.0,
            "max_camber_pct": 0.0,
            "reynolds": 1000000,
            "alpha": a,
            "cl": 0.0,
            "cd": 0.008 + a*0.0005,
            "drag_counts": 80.0,
            "ld": 0.0,
            "cm": 0.0,
            "top_xtr": 0.5,
            "bot_xtr": 0.5,
            "converged": True
        }
        for a in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ]

    count = db.append_records(sample_records)
    assert count == 6
    assert os.path.exists(test_csv)

    # 読み込みと完了チェック
    keys = db.get_completed_keys()
    assert (25.0, 42.0, 1.0) in keys

    # クリーンアップ
    for p in [test_csv, test_pkl]:
        if os.path.exists(p):
            os.remove(p)


def test_xfoil_single_case_execution():
    from core.geometry import generate_funcgen_airfoil, save_airfoil_dat
    from core.xfoil_runner import run_single_xfoil_case

    test_dat = os.path.join(BASE_DIR, "tests", "temp_single_case.dat")
    x, y = generate_funcgen_airfoil(25.0, 42.0, 1.0)
    save_airfoil_dat(test_dat, x, y, "TestCaseFoil")

    xfoil_exe = os.path.join(BASE_DIR, "xfoil.exe")
    if os.path.exists(xfoil_exe):
        records = run_single_xfoil_case(
            dat_path=test_dat,
            reynolds=1000000,
            alpha_min=0.0,
            alpha_max=2.0,
            alpha_step=1.0,
            timeout_sec=25,
            xfoil_exe=xfoil_exe
        )
        assert len(records) > 0, "XFOIL should return at least 1 record"
        assert "cd" in records[0]
        assert records[0]["reynolds"] == 1000000

    if os.path.exists(test_dat):
        os.remove(test_dat)


if __name__ == "__main__":
    tests = [
        test_format_foil_name_3d,
        test_database_manager_basic,
        test_xfoil_single_case_execution
    ]
    failed = 0
    print("=" * 60)
    print(" Running Unit Tests for core.xfoil_runner & core.database")
    print("=" * 60)
    for t in tests:
        try:
            t()
            print(f" [PASS] {t.__name__}")
        except Exception as e:
            print(f" [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print("-" * 60)
    if failed == 0:
        print(" ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f" {failed} TEST(S) FAILED!")
        sys.exit(1)
