"""
Test suite for core.geometry (FuncGenFoil based airfoil geometry engine)
"""

import os
import sys
import numpy as np

# テスト対象の core パッケージへのパスを通す
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.geometry import (
    generate_funcgen_airfoil,
    save_airfoil_dat,
    calculate_le_radius,
    calculate_max_thickness,
    cosine_projection_x
)


def test_cosine_projection():
    """円周コサイン射影の基本動作テスト"""
    alphas = np.linspace(0.0, 1.0, 101)
    xs = cosine_projection_x(alphas)
    assert np.isclose(xs[0], 1.0, atol=1e-5), "alpha=0 must map to TE (x=1)"
    assert np.isclose(xs[50], 0.0, atol=1e-5), "alpha=0.5 must map to LE (x=0)"
    assert np.isclose(xs[-1], 1.0, atol=1e-5), "alpha=1.0 must map to TE (x=1)"
    assert np.all((xs >= 0.0) & (xs <= 1.0)), "x must stay in [0, 1]"


def test_standard_naca_baseline():
    """kle=1.0 (標準前縁半径) のとき、最大翼厚と位置が指定値と一致するか"""
    tc_pct = 25.0
    xt_pct = 42.0
    kle = 1.0

    x, y = generate_funcgen_airfoil(tc_pct=tc_pct, xt_pct=xt_pct, kle=kle, num_points=161)
    assert len(x) == 161
    assert len(y) == 161

    # 最大厚みと最大厚み位置の検証
    measured_tc, measured_xt = calculate_max_thickness(x, y)
    assert np.isclose(measured_tc, tc_pct, atol=0.2), f"Expected tc={tc_pct}%, got {measured_tc:.2f}%"
    assert np.isclose(measured_xt, xt_pct, atol=0.5), f"Expected xt={xt_pct}%, got {measured_xt:.2f}%"

    # 上下対称性 (完全対称翼)
    mid_idx = 80  # 前縁
    assert np.isclose(x[mid_idx], 0.0, atol=1e-4)
    assert np.isclose(y[mid_idx], 0.0, atol=1e-4)


def test_le_radius_scaling():
    """kle を振ったとき、前縁半径 r_le が意図通りスケールするか"""
    tc_pct = 25.0
    xt_pct = 42.0

    # kle=1.0 (基準)
    x_base, y_base = generate_funcgen_airfoil(tc_pct, xt_pct, kle=1.0)
    r_base = calculate_le_radius(x_base, y_base)

    # kle=0.7 (シャープ前縁)
    x_sharp, y_sharp = generate_funcgen_airfoil(tc_pct, xt_pct, kle=0.7)
    r_sharp = calculate_le_radius(x_sharp, y_sharp)

    # kle=1.3 (鈍頭前縁)
    x_blunt, y_blunt = generate_funcgen_airfoil(tc_pct, xt_pct, kle=1.3)
    r_blunt = calculate_le_radius(x_blunt, y_blunt)

    # r_sharp < r_base < r_blunt の単調増加性
    assert r_sharp < r_base, f"r_sharp ({r_sharp:.5f}) must be smaller than r_base ({r_base:.5f})"
    assert r_blunt > r_base, f"r_blunt ({r_blunt:.5f}) must be larger than r_base ({r_base:.5f})"

    # 比率の概ねの一致 (誤差 ±15% 以内)
    ratio_sharp = r_sharp / r_base
    ratio_blunt = r_blunt / r_base
    assert np.isclose(ratio_sharp, 0.7, atol=0.15), f"Expected ratio ~0.7, got {ratio_sharp:.3f}"
    assert np.isclose(ratio_blunt, 1.3, atol=0.15), f"Expected ratio ~1.3, got {ratio_blunt:.3f}"


def test_local_deformation_isolation():
    """前縁変形が最大厚み位置以降 (x >= 0.3) の層流維持領域に影響を与えないこと (Prior の維持)"""
    tc_pct = 25.0
    xt_pct = 42.0

    x_base, y_base = generate_funcgen_airfoil(tc_pct, xt_pct, kle=1.0)
    x_mod, y_mod = generate_funcgen_airfoil(tc_pct, xt_pct, kle=1.3)

    # 上面側で x >= 0.3 の領域の y の差分
    upper_mask = (y_base >= 0) & (x_base >= 0.30)
    diff = np.abs(y_mod[upper_mask] - y_base[upper_mask])
    max_diff = np.max(diff)

    assert max_diff < 1e-3, f"Shape behind x=0.3 must be preserved (max diff: {max_diff:.6f})"


def test_save_airfoil_and_xfoil_compatibility():
    """保存した .dat ファイルが XFOIL で正常に読み込めるか"""
    tc_pct = 25.0
    xt_pct = 42.0
    kle = 1.2
    x, y = generate_funcgen_airfoil(tc_pct, xt_pct, kle)

    test_dat = os.path.join(BASE_DIR, "tests", "temp_test_foil.dat")
    save_airfoil_dat(test_dat, x, y, "FuncGen_Test_Foil")
    assert os.path.exists(test_dat)

    # XFOIL による読み込み＆PANEテスト
    xfoil_exe = os.path.join(BASE_DIR, "xfoil.exe")
    if os.path.exists(xfoil_exe):
        import subprocess
        cmds = [
            f"load {test_dat}",
            "pane",
            "quit"
        ]
        proc = subprocess.run([xfoil_exe], input="\n".join(cmds)+"\n", text=True, capture_output=True, cwd=os.path.dirname(test_dat))
        assert proc.returncode == 0
        assert "Error" not in proc.stdout

    if os.path.exists(test_dat):
        os.remove(test_dat)


if __name__ == "__main__":
    tests = [
        test_cosine_projection,
        test_standard_naca_baseline,
        test_le_radius_scaling,
        test_local_deformation_isolation,
        test_save_airfoil_and_xfoil_compatibility
    ]
    failed = 0
    print("=" * 60)
    print(" Running Unit Tests for core.geometry")
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

