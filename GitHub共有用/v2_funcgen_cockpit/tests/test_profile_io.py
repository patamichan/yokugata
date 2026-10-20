"""Unit tests for arbitrary CSV profile loading, bottleneck analysis, and DXF generation."""
import os
import tempfile
from pathlib import Path
import pytest
import ezdxf
from core.paths import INPUT_DIR

from core.optimizer import (
    PRESET_PILOT_STANDARD,
    load_pilot_profile_csv,
    save_pilot_profile_csv,
    analyze_envelope_bottlenecks,
    calculate_required_chord,
    group_profile_by_level,
    get_profile_level_summary,
    flatten_profile_projection
)
from cockpit_selection_3d_cli import _export_dxf_file, export_cad_package

def test_load_template_csv():
    template_path = INPUT_DIR / "pilot_envelope_template.csv"
    assert template_path.is_file()
    profile = load_pilot_profile_csv(template_path)
    assert len(profile) == 8
    assert profile[0]["name"] == "Pedal_Bottom"
    assert profile[0]["level"] == 1
    assert profile[0]["z_mm"] == 100.0
    assert profile[0]["x_mm"] == 150.0
    assert profile[-1]["name"] == "Helmet"
    assert profile[-1]["level"] == 4
    assert profile[-1]["z_mm"] == 850.0

def test_hierarchical_profile_grouping_and_summary():
    template_path = INPUT_DIR / "pilot_envelope_template.csv"
    profile = load_pilot_profile_csv(template_path)
    grouped = group_profile_by_level(profile)
    assert len(grouped) == 4
    assert 1 in grouped and 4 in grouped
    assert len(grouped[1]) == 2
    assert len(grouped[3]) == 3

    summary = get_profile_level_summary(profile)
    assert len(summary) == 4
    assert summary[0]["level"] == 1
    assert summary[0]["z_mm_mean"] == 100.0
    assert summary[2]["level"] == 3
    assert summary[2]["station_count"] == 3

    flattened = flatten_profile_projection(profile)
    assert len(flattened) > 0
    # Make sure maximum width envelope retains critical stations
    names = [f["name"] for f in flattened]
    assert any("Shoulders" in n for n in names)


def test_load_standard_csv():
    std_path = INPUT_DIR / "pilot_standard.csv"
    assert std_path.is_file()
    profile = load_pilot_profile_csv(std_path)
    assert len(profile) == 4
    for orig, loaded in zip(PRESET_PILOT_STANDARD, profile):
        assert orig["name"] == loaded["name"]
        assert orig["x_mm"] == loaded["x_mm"]
        assert orig["width_mm"] == loaded["width_mm"]
        assert orig["clearance_mm"] == loaded["clearance_mm"]

def test_save_and_reload_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "custom_stations.csv"
        custom_data = [
            {"name": "Crank", "x_mm": 250.0, "width_mm": 300.0, "clearance_mm": 15.0},
            {"name": "Chest", "x_mm": 1100.0, "width_mm": 450.0, "clearance_mm": 30.0},
        ]
        save_pilot_profile_csv(custom_data, csv_file)
        assert csv_file.is_file()
        loaded = load_pilot_profile_csv(csv_file)
        assert len(loaded) == 2
        assert loaded[0]["name"] == "Crank"
        assert loaded[1]["name"] == "Chest"

def test_analyze_bottlenecks():
    diag = analyze_envelope_bottlenecks(24.0, 42.0, 1.2, PRESET_PILOT_STANDARD,
                                        nose_margin_mm=30.0, generator_method="cst")
    assert "overall_chord_mm" in diag
    assert "bottlenecks" in diag
    assert len(diag["stations"]) == 4
    # In CST kle=1.2, Feet/Pedal is the binding constraint
    assert "Feet/Pedal" in diag["bottlenecks"]
    feet_stat = next(s for s in diag["stations"] if s["name"] == "Feet/Pedal")
    assert feet_stat["is_bottleneck"] is True
    assert feet_stat["margin_mm"] < 0.1

def test_dxf_export_in_package():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        mock_result = {
            "airfoil": "FAIRING_TEST_01",
            "generator_method": "cst",
            "thickness_pct": 24.0,
            "max_thickness_pos_pct": 42.0,
            "le_radius_ratio": 1.2,
            "chord_mm": 2327.3,
            "cd_eff": 0.0075,
            "drag_area_mm": 17.45,
            "reynolds": 2.0e6
        }
        folder = export_cad_package(mock_result, PRESET_PILOT_STANDARD, 14.0, 1.5,
                                   output_dir=out_dir, nose_margin_mm=30.0)
        assert folder.is_dir()
        dxf_file = folder / "section_scaled_mm.dxf"
        dat_scaled = folder / "section_scaled_mm.dat"
        dat_norm = folder / "section_normalized.dat"
        spec_txt = folder / "選定結果仕様書.txt"

        assert dxf_file.is_file()
        assert dat_scaled.is_file()
        assert dat_norm.is_file()
        assert spec_txt.is_file()

        # Check DXF validity
        doc = ezdxf.readfile(dxf_file)
        assert doc.header.get("$INSUNITS") == 4
        layer_names = [l.dxf.name for l in doc.layers]
        assert "0_AIRFOIL_OUTLINE" in layer_names
        assert "3_PILOT_CLEARANCE" in layer_names

        # Check spec file mentions bottlenecks
        content = spec_txt.read_text(encoding="utf-8")
        assert "ボトルネック" in content


def test_optimize_nose_margin_logic(tmp_path):
    from core.database import Database3DManager
    from core.optimizer import optimize_nose_margin_3d

    # Tests must not import v1 records or rewrite the user's real database.
    db = Database3DManager(csv_path=tmp_path / 'nose_sweep.csv')
    db.append_records([dict(airfoil='test_sweep', thickness_pct=24.,
        max_thickness_pos_pct=42., le_radius_ratio=1.2, max_camber_pct=0.,
        generator_method='warp', reynolds=re, alpha=float(alpha), cl=0., cd=.008,
        drag_counts=80., ld=0., cm=0., top_xtr=.5, bot_xtr=.5, converged=True)
        for re in (1000000, 3000000) for alpha in (0, 2, 4, 6, 8, 10, 12)])

    # Fast sweep across 3 test margins
    sweep_res = optimize_nose_margin_3d(db, PRESET_PILOT_STANDARD, velocity_ms=14.0, crosswind_ms=1.5,
                                       generator_method="warp",
                                       margin_min_mm=30.0, margin_max_mm=60.0, margin_step_mm=15.0)

    assert "best_result" in sweep_res
    assert "best_margin_mm" in sweep_res
    assert "sweep_history" in sweep_res
    assert len(sweep_res["sweep_history"]) == 3
    for entry in sweep_res["sweep_history"]:
        assert "nose_margin_mm" in entry
        assert "chord_mm" in entry
        assert "drag_area_mm" in entry

