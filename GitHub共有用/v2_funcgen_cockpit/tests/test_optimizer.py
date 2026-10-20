"""
Unit tests for core.optimizer (3D Section Sizing & Drag Area Minimizer)
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def test_clearance_chord_calculation():
    """前縁半径 kle が大きいほど、最前部（足先）の制約下で必要コード長 c が短縮されるか"""
    from core.optimizer import calculate_required_chord

    # パイロットの足先寸法 (X=100mm で幅 320mm が必要)
    pilot_profile = [
        {"x_mm": 100.0, "width_mm": 320.0, "clearance_mm": 15.0}  # 必要幅 350mm
    ]

    tc_pct = 25.0
    xt_pct = 42.0

    # kle = 0.8 (シャープ前縁)
    chord_sharp = calculate_required_chord(tc_pct, xt_pct, kle=0.8, pilot_profile=pilot_profile)

    # kle = 1.0 (標準前縁)
    chord_base = calculate_required_chord(tc_pct, xt_pct, kle=1.0, pilot_profile=pilot_profile)

    # kle = 1.3 (鈍頭前縁)
    chord_blunt = calculate_required_chord(tc_pct, xt_pct, kle=1.3, pilot_profile=pilot_profile)

    # 前縁が太いほどコード長は短く収まるはず
    assert chord_blunt < chord_base, f"chord_blunt ({chord_blunt:.1f}) must be < chord_base ({chord_base:.1f})"
    assert chord_base < chord_sharp, f"chord_base ({chord_base:.1f}) must be < chord_sharp ({chord_sharp:.1f})"


def test_effective_cd_crosswind_integral():
    """横風確率分布による期待抗力 E[Cd] の計算テスト"""
    from core.optimizer import calculate_expected_cd

    # ダミーの Cd(alpha) 関数 (alpha=0 で 0.0060, alpha=5 で 0.0150)
    def dummy_cd_func(alpha_deg):
        return 0.0060 + 0.0003 * (alpha_deg ** 2)

    # 横風ゼロ (直進のみ): E[Cd] == 0.0060
    cd_eff_zero = calculate_expected_cd(dummy_cd_func, velocity_ms=14.0, crosswind_ms=0.0)
    assert np.isclose(cd_eff_zero, 0.0060, atol=1e-4)

    # 横風あり (V_cross = 1.5 m/s): E[Cd] > 0.0060
    cd_eff_wind = calculate_expected_cd(dummy_cd_func, velocity_ms=14.0, crosswind_ms=1.5)
    assert cd_eff_wind > 0.0060, "Effective Cd with crosswind must be larger than zero-crosswind Cd"


if __name__ == "__main__":
    tests = [
        test_clearance_chord_calculation,
        test_effective_cd_crosswind_integral
    ]
    failed = 0
    print("=" * 60)
    print(" Running Unit Tests for core.optimizer")
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
