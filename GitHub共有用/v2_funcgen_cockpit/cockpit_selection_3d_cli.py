"""Interactive section sizing, bottleneck diagnosis, and method-consistent CAD export.

Supports:
- Arbitrary station names and counts via CSV files (--profile)
- Automatic bottleneck identification (which station binds the chord)
- Native DXF (AutoCAD/SolidWorks) and DAT CAD package export
"""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import time
import sys

import ezdxf
from ezdxf.enums import TextEntityAlignment

from core.database import Database3DManager
from core.optimizer import (
    optimize_section_3d, optimize_nose_margin_3d,
    PRESET_PILOT_STANDARD, _profile_arrays,
    load_pilot_profile_csv, save_pilot_profile_csv,
    analyze_envelope_bottlenecks, group_profile_by_level,
    get_profile_level_summary, flatten_profile_projection
)
from core.geometry import generate_funcgen_airfoil, save_airfoil_dat
from core.paths import DEFAULT_DATABASE, INPUT_DIR, CAD_DIR

BASE_DIR = Path(__file__).resolve().parent


def prompt_number(label, default=None, minimum=0.0, positive=False):
    while True:
        raw = input(f'{label}' + (f' [{default}]' if default is not None else '') + ': ').strip()
        if not raw and default is not None:
            return float(default)
        try:
            value = float(raw)
            if math.isfinite(value) and value >= minimum and (not positive or value > 0):
                return value
        except ValueError:
            pass
        print('有効な数値を入力してください。')


def choose_profile_interactive():
    print('\n【人体・機材プロファイル選択】')
    print('  [1] 標準4部位プリセット (Feet, Knees, Hips, Shoulders)')
    print('  [2] CSVファイルから読み込み (任意部位名・任意点数・高さ階層対応)')
    print('  [3] テンプレートCSVを新規作成 (pilot_envelope_template.csv)')
    print('  [4] 画面上で手動入力 (旧方式)')
    choice = input('選択 [1]: ').strip()

    if not choice or choice == '1':
        return [dict(p) for p in PRESET_PILOT_STANDARD]
    elif choice == '2':
        csv_path_str = input('CSVファイルパス [pilot_envelope_template.csv]: ').strip()
        if not csv_path_str:
            csv_path_str = 'pilot_envelope_template.csv'
        p = Path(csv_path_str)
        if not p.is_file():
            p = INPUT_DIR / csv_path_str
        if not p.is_file():
            raise FileNotFoundError(f'指定されたCSVが見つかりません: {csv_path_str}')
        profile = load_pilot_profile_csv(p)
        print_profile_hierarchy(profile, p.name)
        return profile
    elif choice == '3':
        tmpl_path = INPUT_DIR / 'pilot_envelope_template.csv'
        print(f'-> テンプレートファイル: {tmpl_path.resolve()}')
        print('このファイルをExcel等で編集後、再度 [2] から読み込んでください。')
        profile = load_pilot_profile_csv(tmpl_path)
        print_profile_hierarchy(profile, tmpl_path.name)
        return profile
    elif choice == '4':
        count = prompt_number('測定点数', minimum=1, positive=True)
        if not count.is_integer():
            raise ValueError('測定点数は整数で指定してください')
        points = []
        for i in range(int(count)):
            name = input(f'点 {i+1} の名前 (例: 足先/肩/ヘルメット) [Point {i+1}]: ').strip() or f'Point {i+1}'
            lvl = input('  高さ階層レベル番号 (例: 1, 2, 3...) [1]: ').strip() or '1'
            try:
                lvl = int(lvl)
            except ValueError:
                pass
            z = prompt_number('  基準底面からの高さ z [mm]', default=0.0)
            x = prompt_number('  ノーズからの前後位置 x [mm]')
            w = prompt_number('  必要幅 [mm]')
            clr = prompt_number('  片側クリアランス [mm]', 20.0)
            points.append(dict(name=name, level=lvl, z_mm=z, x_mm=x, width_mm=w, clearance_mm=clr))
        return points
    else:
        raise ValueError('1〜4の番号で選択してください。')


def print_profile_hierarchy(profile, source_name="Profile"):
    """Print an easy-to-read tree of height levels and stations."""
    summaries = get_profile_level_summary(profile)
    print(f"\n階層構造プロファイル読み込み成功: {source_name} (計 {len(profile)} 点 / {len(summaries)} 階層)")
    for s in summaries:
        st_names = ", ".join(s['station_names'])
        print(f"  ├─ [Level {s['level']}: Z≈{s['z_mm_mean']:.0f}mm] {s['station_count']} 点: ({st_names})")


def _export_dxf_file(folder, x_norm, y_norm, chord_mm, method, result, pilot_profile, nose_margin_mm=30.0):
    """Export production-ready AutoCAD R2010 DXF with real-scale mm coordinates and hierarchical layers."""
    import numpy as np
    x_mm = x_norm * chord_mm
    y_mm = y_norm * chord_mm
    filepath = folder / 'section_scaled_mm.dxf'

    doc = ezdxf.new('R2010', setup=True)
    doc.header['$INSUNITS'] = 4   # mm
    doc.header['$MEASUREMENT'] = 1 # Metric

    doc.layers.add('0_AIRFOIL_OUTLINE', color=5, lineweight=35) # Blue
    doc.layers.add('1_AIRFOIL_SPLINE', color=4, lineweight=25)  # Cyan
    doc.layers.add('2_CENTERLINE', color=1, linetype='CENTER')  # Red
    doc.layers.add('3_PILOT_CLEARANCE', color=3, lineweight=25) # Green default
    doc.layers.add('4_ANNOTATIONS', color=7)                    # White/Black

    # Color palette for hierarchical height levels
    LEVEL_COLORS = {
        1: 2,  # Yellow (Bottom / Feet / Seat bottom)
        2: 3,  # Green (Mid-low / Crank / Hips)
        3: 6,  # Magenta (Mid-high / Knees / Shoulders)
        4: 4,  # Cyan (Top / Helmet)
        5: 1,  # Red
        6: 5,  # Blue
    }

    msp = doc.modelspace()
    # 1. Closed Outline Polyline
    points_2d = list(zip(x_mm, y_mm))
    msp.add_lwpolyline(points_2d, close=True, dxfattribs={'layer': '0_AIRFOIL_OUTLINE'})

    # 2. Smooth Spline representation
    spline_pts = list(zip(x_mm[::5], y_mm[::5]))
    if spline_pts[-1] != spline_pts[0]:
        spline_pts.append(spline_pts[0])
    msp.add_spline(spline_pts, dxfattribs={'layer': '1_AIRFOIL_SPLINE'})

    # 3. Centerline
    msp.add_line((0.0, 0.0), (chord_mm, 0.0), dxfattribs={'layer': '2_CENTERLINE'})

    # 4. Pilot Stations & Boxes categorized by height levels
    if pilot_profile:
        distances, widths = _profile_arrays(pilot_profile, nose_margin_mm=nose_margin_mm)
        grouped = group_profile_by_level(pilot_profile)

        for lvl_idx, (lvl, stations) in enumerate(grouped.items(), start=1):
            color = LEVEL_COLORS.get(lvl if isinstance(lvl, int) else lvl_idx, 30 + (lvl_idx % 20) * 10)
            z_mean = np.mean([s.get('z_mm', 0.0) for s in stations])
            layer_name = f"3_PILOT_LV{lvl}_Z{z_mean:.0f}"
            doc.layers.add(layer_name, color=color, lineweight=25)

        for p, d, w in zip(pilot_profile, distances, widths):
            lvl = p.get('level', 1)
            z_val = p.get('z_mm', 0.0)
            body_w = p.get('width_mm', 0.0)
            clr = p.get('clearance_mm', 0.0)

            layer_name = f"3_PILOT_LV{lvl}_Z{z_val:.0f}"
            if layer_name not in doc.layers:
                doc.layers.add(layer_name, color=3, lineweight=25)

            # Center line segment for pilot
            msp.add_line((d, -body_w / 2.0), (d, body_w / 2.0),
                         dxfattribs={'layer': layer_name})
            # Clearance boundary box
            rect_pts = [(d - 15.0, -w / 2.0), (d + 15.0, -w / 2.0),
                        (d + 15.0, w / 2.0), (d - 15.0, w / 2.0)]
            msp.add_lwpolyline(rect_pts, close=True,
                               dxfattribs={'layer': layer_name})
            # Label
            label = f"[LV{lvl}:Z={z_val:.0f}] {p.get('name', 'Station')}: W={w:.0f} (Body {body_w:.0f}+Clr {clr*2:.0f})"
            msp.add_text(label, dxfattribs={'layer': '4_ANNOTATIONS', 'height': 16.0}).set_placement(
                (d + 20.0, w / 2.0 + 8.0), align=TextEntityAlignment.BOTTOM_LEFT)

    # 5. Title
    title = f"AIRFOIL CAD: {result['airfoil']} ({method.upper()}) Chord={chord_mm:.1f}mm"
    msp.add_text(title, dxfattribs={'layer': '4_ANNOTATIONS', 'height': 28.0}).set_placement(
        (0.0, chord_mm * (result['thickness_pct'] / 100.0) / 2.0 + 80.0),
        align=TextEntityAlignment.BOTTOM_LEFT)

    doc.saveas(filepath)
    return filepath


def export_cad_package(result, pilot_profile, velocity_ms, crosswind_ms, output_dir=None, nose_margin_mm=30.0):
    method = result.get('generator_method', 'warp')
    x, y = generate_funcgen_airfoil(result['thickness_pct'], result['max_thickness_pos_pct'],
                                    result['le_radius_ratio'], num_points=2001, method=method)
    root = Path(output_dir) if output_dir else CAD_DIR
    folder = root / (f'CAD用_3D確定翼型_{datetime.now():%Y%m%d_%H%M%S}_V{velocity_ms:g}')
    folder.mkdir(parents=True, exist_ok=False)
    chord = result['chord_mm']

    # 1. Real-scale mm DAT file
    save_airfoil_dat(str(folder / 'section_scaled_mm.dat'), x * chord, y * chord,
                     f"{result['airfoil']} Real Scale [mm] method={method} Chord={chord:.2f}mm")

    # 2. Normalized DAT file
    save_airfoil_dat(str(folder / 'section_normalized.dat'), x, y,
                     f"{result['airfoil']} Normalized method={method}")

    # 3. Direct CAD DXF file (AutoCAD/SolidWorks)
    dxf_path = _export_dxf_file(folder, x, y, chord, method, result, pilot_profile, nose_margin_mm)

    # 4. Bottleneck Diagnosis
    diag = analyze_envelope_bottlenecks(result['thickness_pct'], result['max_thickness_pos_pct'],
                                        result['le_radius_ratio'], pilot_profile,
                                        nose_margin_mm=nose_margin_mm, generator_method=method) if pilot_profile else None

    lines = ['======================================================================',
             '鳥人間コンテスト 3Dフェアリング 選定結果仕様書',
             f'生成日時: {datetime.now():%Y-%m-%d %H:%M:%S}',
             f'生成方式: {method}',
             f'翼型名: {result["airfoil"]}',
             f'設計機速: {velocity_ms:g} m/s | 最大想定横風: {crosswind_ms:g} m/s',
             f'ノーズマージン: {nose_margin_mm:g} mm',
             '======================================================================',
             '',
             '【選定翼型パラメータ】',
             f"  ・翼弦長 (Chord):        {chord:.2f} mm",
             f"  ・最大厚さ比 (t/c):       {result['thickness_pct']:.1f} %",
             f"  ・最大厚さ位置 (x_t):     {result['max_thickness_pos_pct']:.1f} %",
             f"  ・前縁半径比 (k_le):      {result['le_radius_ratio']:.2f}",
             f"  ・実効抗力係数 (Cd_eff):  {result['cd_eff']:.5f}",
             f"  ・全機抵抗面積 (Chord*Cd): {result['drag_area_mm']:.2f} mm",
             f"  ・評価Reynolds数:        {result.get('reynolds', 0.0):.0f}" +
             (" (データ境界にクランプ)" if result.get('reynolds_clamped') else ""),
             '']
    if diag:
        lines.extend([
            '======================================================================',
            '【階層別断面クリアランス＆ボトルネック診断】',
            '======================================================================'
        ])
        level_diag = diag.get('level_diagnostics', {})
        if level_diag:
            for lvl, info in level_diag.items():
                lines.append(f"\n▼ [Level {lvl}: Z≈{info['z_mm']:.0f}mm] (最厳余裕: {info['min_margin_mm']:+.2f}mm / 支配部位: {info['critical_station']})")
                for st in info['stations']:
                    tag = ' ★ボトルネック (全体長を決定)' if st['is_bottleneck'] else ''
                    lines.append(f"  ・{st['name']:14s} (x={st['x_dist_mm']:6.1f}mm): "
                                 f"必要幅={st['required_width_mm']:5.1f}mm | 内部幅={st['available_width_mm']:5.1f}mm | "
                                 f"余裕={st['margin_mm']:+6.2f}mm | 単独必要Chord={st['standalone_chord_mm']:6.1f}mm{tag}")
        else:
            for st in diag['stations']:
                tag = ' ★ボトルネック (全体長を決定)' if st['is_bottleneck'] else ''
                lines.append(f"  ・{st['name']:14s} (x={st['x_dist_mm']:6.1f}mm): "
                             f"必要幅={st['required_width_mm']:5.1f}mm | 内部幅={st['available_width_mm']:5.1f}mm | "
                             f"余裕={st['margin_mm']:+6.2f}mm | 単独必要Chord={st['standalone_chord_mm']:6.1f}mm{tag}")

    lines.extend([
        '',
        '======================================================================',
        '【出力CADファイルのご案内】',
        '======================================================================',
        f'1. section_scaled_mm.dxf  : ★最も推奨。AutoCAD, SolidWorks, Fusion 360で直接開けます。',
        f'                            実寸mm・階層別レイヤー分け・クリアランス枠・寸法注記入り。',
        f'2. section_scaled_mm.dat  : 実寸mmのSelig形式座標テキスト。',
        f'3. section_normalized.dat : 全長1.0に正規化された無次元Selig座標テキスト。',
        '======================================================================'
    ])
    (folder / '選定結果仕様書.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return folder


def print_bottleneck_report(diag):
    print('\n' + '=' * 70)
    print('【階層別パイロット断面クリアランス＆ボトルネック診断】')
    print('=' * 70)
    level_diag = diag.get('level_diagnostics', {})
    if level_diag:
        for lvl, info in level_diag.items():
            print(f"\n▼ [Level {lvl}: Z≈{info['z_mm']:.0f}mm] (最厳余裕: {info['min_margin_mm']:+.2f}mm / 支配部位: {info['critical_station']})")
            for st in info['stations']:
                star = ' ★全体長を決定(ボトルネック)' if st['is_bottleneck'] else ''
                print(f"  ・{st['name']:14s} (x={st['x_dist_mm']:6.1f}mm): "
                      f"内部幅={st['available_width_mm']:5.1f}mm / 必要幅={st['required_width_mm']:5.1f}mm "
                      f"[余裕={st['margin_mm']:+6.2f}mm] (単独Chord={st['standalone_chord_mm']:6.1f}mm){star}")
    else:
        for st in diag['stations']:
            star = ' ★全体長を決定(ボトルネック)' if st['is_bottleneck'] else ''
            print(f"・{st['name']:14s} (x={st['x_dist_mm']:6.1f}mm): "
                  f"内部幅={st['available_width_mm']:5.1f}mm / 必要幅={st['required_width_mm']:5.1f}mm "
                  f"[余裕={st['margin_mm']:+6.2f}mm] (単独Chord={st['standalone_chord_mm']:6.1f}mm){star}")
    print('-' * 70)
    b_names = diag['bottlenecks']
    if b_names:
        print(f"👉 診断結果: 全長は 『{', '.join(b_names)}』 によって拘束されています。")
        print(f"   この部位の幅を詰める、または位置を後方に下げると、全長をさらに短縮できます。")
    print('=' * 70 + '\n')


def print_nose_sweep_report(sweep_data):
    history = sweep_data['sweep_history']
    best_m = sweep_data['best_margin_mm']
    best_res = sweep_data['best_result']

    print('\n' + '=' * 82)
    print('【ノーズマージン自動探索結果（トレードオフ比較表）】')
    print('=' * 82)
    print(f"{'マージン [mm]':^12} | {'最良翼型':^18} | {'Chord [mm]':^11} | {'Cd_eff':^9} | {'抵抗面積 [mm]':^12} | {'判定'}")
    print('-' * 82)
    for entry in history:
        m = entry['nose_margin_mm']
        star = '★ 最適解 (Best!)' if abs(m - best_m) < 1e-4 else ''
        print(f"  {m:6.1f} mm   | {entry['airfoil']:<18} | {entry['chord_mm']:8.1f} mm | {entry['cd_eff']:7.5f} | "
              f"{entry['drag_area_mm']:9.2f} mm | {star}")
    print('-' * 82)
    init_area = history[0]['drag_area_mm']
    best_area = best_res['drag_area_mm']
    diff_pct = (best_area - init_area) / init_area * 100.0
    print(f"👉 自動探索完了: 最適ノーズマージンは 【{best_m:.1f} mm】 です！")
    print(f"   初期マージン({history[0]['nose_margin_mm']:.1f}mm)比で全機抵抗面積が {diff_pct:+.1f}% 改善しました。")
    print('=' * 82 + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['warp', 'cst'], default='warp',
                        help='翼型生成方式 (warp: 従来スプライン / cst: Class-Shape Transformation)')
    parser.add_argument('--profile', type=Path, default=None,
                        help='パイロット寸法CSVファイルのパス (指定なしの場合は対話選択)')
    parser.add_argument('--velocity', type=float, default=None,
                        help='設計機速 [m/s] (デフォルト: 14.0)')
    parser.add_argument('--crosswind', type=float, default=None,
                        help='想定最大横風 [m/s] (デフォルト: 1.5)')
    parser.add_argument('--nose-margin', type=float, default=None,
                        help='前縁ノーズマージン [mm] (固定指定時)')
    parser.add_argument('--auto-nose', action='store_true',
                        help='ノーズマージンを自動スイープ探索して抵抗最小位置を特定')
    parser.add_argument('--nose-range', type=str, default='30:120:15',
                        help='ノーズ自動探索範囲 (min:max:step, デフォルト: 30:120:15)')
    parser.add_argument('--max-chord', type=float, default=None,
                        help='最大許容翼弦長 [mm]')
    parser.add_argument('--csv', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output-dir', type=Path, default=CAD_DIR)
    parser.add_argument('--auto-export', action='store_true',
                        help='対話確認をスキップして即座にCADパッケージ(DXF+DAT)を出力')
    parser.add_argument('--dry-run', action='store_true', help='設定のみ表示。DB変更・選定・出力なし')
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f'Dry run: method={args.method}, profile={args.profile}, CSV={args.csv}')
        return 0

    if not args.csv.is_file():
        print(f'データベースがありません: {args.csv}。承認済みの解析データを用意してください。')
        return 1

    db = Database3DManager(csv_path=args.csv)
    db.load_database(legacy_csv_path=BASE_DIR / '__no_implicit_legacy_import__.csv')

    # 1. Velocity & Crosswind
    velocity = args.velocity if args.velocity is not None else prompt_number('設計機速 [m/s]', 14.0, positive=True)
    wind = args.crosswind if args.crosswind is not None else prompt_number('想定最大横風 [m/s]', 1.5)

    # 2. Max chord limit
    limit = args.max_chord
    if limit is None and args.velocity is None: # interactive mode
        raw_limit = input('最大翼弦長 [mm]（無制限は空欄）: ').strip()
        limit = float(raw_limit) if raw_limit else None

    # 3. Profile loading
    if args.profile:
        if not args.profile.is_file() and not args.profile.is_absolute():
            args.profile = INPUT_DIR / args.profile
        if not args.profile.is_file():
            print(f'指定されたプロファイルファイルが見つかりません: {args.profile}')
            return 1
        profile = load_pilot_profile_csv(args.profile)
        print_profile_hierarchy(profile, args.profile.name)
    else:
        profile = choose_profile_interactive()

    # 4. Nose Margin Selection (Fixed or Auto)
    auto_nose = args.auto_nose
    nose_margin = args.nose_margin

    if not auto_nose and nose_margin is None:
        raw_margin = input('\nノーズマージン [mm] (数値を入力、または "A" で自動最適化) [30.0]: ').strip()
        if raw_margin.lower() in ('a', 'auto'):
            auto_nose = True
        elif raw_margin:
            try:
                nose_margin = float(raw_margin)
            except ValueError:
                print('無効な値のためデフォルト 30.0mm を使用します。')
                nose_margin = 30.0
        else:
            nose_margin = 30.0

    # 5. Optimization Execution
    if auto_nose:
        # Parse nose range
        try:
            parts = [float(p) for p in args.nose_range.split(':')]
            m_min, m_max, m_step = parts[0], parts[1], parts[2]
        except Exception:
            m_min, m_max, m_step = 30.0, 120.0, 15.0

        print(f'\n【ノーズマージン自動探索開始】 範囲: {m_min:.1f}mm 〜 {m_max:.1f}mm (刻み {m_step:.1f}mm)...')
        start = time.perf_counter()
        sweep_data = optimize_nose_margin_3d(db, profile, velocity_ms=velocity,
                                            crosswind_ms=wind, max_chord_mm=limit,
                                            generator_method=args.method,
                                            margin_min_mm=m_min, margin_max_mm=m_max,
                                            margin_step_mm=m_step)
        elapsed = time.perf_counter() - start
        if not sweep_data:
            print('探索範囲内の全条件で制約を満たす翼型がありませんでした。')
            return 1

        print_nose_sweep_report(sweep_data)
        result = sweep_data['best_result']
        nose_margin = sweep_data['best_margin_mm']
    else:
        print(f'\n最適翼型を選定中... (ノーズマージン: {nose_margin:.1f} mm)')
        start = time.perf_counter()
        result = optimize_section_3d(db, profile, velocity, wind, limit,
                                     generator_method=args.method,
                                     nose_margin_mm=nose_margin)
        elapsed = time.perf_counter() - start

        if not result:
            print('寸法・迎角データの条件を満たす候補がありません。')
            return 1

    print(f'【選定結果】（探索時間 {elapsed:.4f} 秒）')
    print(f"  ・最良翼型:       {result['airfoil']} ({result['generator_method'].upper()})")
    print(f"  ・最大厚さ比:     {result['thickness_pct']:.1f} %")
    print(f"  ・最大厚位置:     {result['max_thickness_pos_pct']:.1f} %")
    print(f"  ・前縁半径比:     {result['le_radius_ratio']:.2f}")
    print(f"  ・ノーズマージン: {nose_margin:.1f} mm")
    print(f"  ・必要翼弦長:     {result['chord_mm']:.2f} mm")
    print(f"  ・実効抗力係数:   {result['cd_eff']:.5f}")
    print(f"  ・全機抵抗面積:   {result['drag_area_mm']:.2f} mm")
    if result.get('reynolds_clamped'):
        print(f"  ※注意: Re={result['reynolds']:.0f} をデータ境界 {result['reynolds_evaluated']:.0f} で評価")

    # 6. Bottleneck Analysis & Print
    diag = analyze_envelope_bottlenecks(result['thickness_pct'], result['max_thickness_pos_pct'],
                                        result['le_radius_ratio'], profile,
                                        nose_margin_mm=nose_margin,
                                        generator_method=args.method)
    print_bottleneck_report(diag)

    # 6. Export CAD
    do_export = args.auto_export
    if not do_export:
        ans = input('CADデータ (DXF & DAT) を出力しますか? [Y/n]: ').strip().lower()
        do_export = (ans == '' or ans == 'y')

    if do_export:
        out_folder = export_cad_package(result, profile, velocity, wind,
                                        output_dir=args.output_dir,
                                        nose_margin_mm=nose_margin)
        print(f'✅ CADパッケージ（実寸DXF・DAT・仕様書）を出力しました:')
        print(f'   📁 {out_folder.resolve()}')

    return 0


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'エラー: {error}')
        raise SystemExit(1)
    except (KeyboardInterrupt, EOFError):
        print('\n操作を中止しました。')
        raise SystemExit(130)
