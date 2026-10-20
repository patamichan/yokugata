"""Plot visual clearance confirmation comparing Recommended (kle=1.2) vs Baseline (kle=1.0)."""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from core.geometry import generate_funcgen_airfoil
from core.paths import CAD_DIR
from core.optimizer import calculate_required_chord, PRESET_PILOT_STANDARD, _profile_arrays

plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_plot():
    CAD_DIR.mkdir(parents=True, exist_ok=True)
    packages = list(CAD_DIR.glob('CAD用_推奨実寸確認パッケージ_*'))
    if not packages:
        out_dir = CAD_DIR
    else:
        out_dir = sorted(packages)[-1]
    
    tc, xt = 24.0, 42.0
    nose_margin = 30.0
    
    c_rec = calculate_required_chord(tc, xt, 1.2, PRESET_PILOT_STANDARD, nose_margin_mm=nose_margin, generator_method='cst')
    x_rec, y_rec = generate_funcgen_airfoil(tc, xt, 1.2, num_points=2001, method='cst')
    
    c_base = calculate_required_chord(tc, xt, 1.0, PRESET_PILOT_STANDARD, nose_margin_mm=nose_margin, generator_method='cst')
    x_base, y_base = generate_funcgen_airfoil(tc, xt, 1.0, num_points=2001, method='cst')
    
    distances, widths = _profile_arrays(PRESET_PILOT_STANDARD, nose_margin_mm=nose_margin)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2.2, 1]})
    
    # --- Top Plot: Real-Scale Fairing & Pilot Enclosure ---
    ax1.plot(x_base * c_base, y_base * c_base, 'gray', linestyle='--', linewidth=1.8, label=f'従来型 kle=1.0 (Chord={c_base:.1f}mm, 周長=5535.9mm)')
    ax1.plot(x_rec * c_rec, y_rec * c_rec, 'blue', linewidth=2.4, label=f'推奨型 kle=1.2 (Chord={c_rec:.1f}mm, 周長=4975.2mm [-10.1%])')
    
    # Pilot envelopes
    colors = ['#d95f02', '#7570b3', '#e7298a', '#1b9e77']
    for p, d, w, col in zip(PRESET_PILOT_STANDARD, distances, widths, colors):
        # Body width box
        body_w = p['width_mm']
        clr = p.get('clearance_mm', 0.0)
        # Pilot body
        ax1.plot([d, d], [-body_w/2, body_w/2], color=col, linewidth=5, solid_capstyle='round')
        # Clearance boundary
        ax1.plot([d, d], [-w/2, w/2], color=col, linestyle=':', linewidth=2)
        ax1.scatter([d, d], [-w/2, w/2], color=col, s=35, zorder=5)
        ax1.annotate(f"{p['name']}\n幅{body_w:.0f}mm (+{clr*2:.0f})",
                     xy=(d, w/2), xytext=(d - 10, w/2 + 35),
                     fontsize=9, weight='bold', color=col,
                     arrowprops=dict(arrowstyle='->', color=col, lw=1))

    ax1.axhline(0, color='black', linestyle='-', linewidth=0.6, alpha=0.5)
    ax1.set_title('コックピットフェアリング実寸断面比較 & パイロット包絡クリアランス (mm)', fontsize=13, weight='bold')
    ax1.set_xlabel('前後位置 X [mm] (前縁ノーズ先端 = 0 mm)', fontsize=11)
    ax1.set_ylabel('幅方向 Y [mm] (中心線 = 0 mm)', fontsize=11)
    ax1.set_aspect('equal', 'datalim')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # --- Bottom Plot: Local Nose Region Zoom (Feet/Pedal interference) ---
    ax2.plot(x_base * c_base, y_base * c_base, 'gray', linestyle='--', linewidth=1.8, label='従来型 kle=1.0')
    ax2.plot(x_rec * c_rec, y_rec * c_rec, 'blue', linewidth=2.4, label='推奨型 kle=1.2')
    
    p_feet = PRESET_PILOT_STANDARD[0]
    d_feet = distances[0]
    w_feet = widths[0]
    ax2.plot([d_feet, d_feet], [-w_feet/2, w_feet/2], color=colors[0], linewidth=4, label='足先/ペダル必要幅 (360mm)')
    ax2.axvline(d_feet, color=colors[0], linestyle=':', alpha=0.5)
    ax2.annotate('足先境界 (x=130mm, 半幅180mm)\n★ kle=1.2によりジャストフィットして全長を273mm短縮！',
                 xy=(d_feet, w_feet/2), xytext=(d_feet + 30, w_feet/2 - 15),
                 fontsize=10, weight='bold', color='#b30000',
                 arrowprops=dict(facecolor='#b30000', shrink=0.05, width=1.5, headwidth=6))

    ax2.set_xlim(-20, 600)
    ax2.set_ylim(-250, 250)
    ax2.set_title('前縁近傍拡大（足先・ペダル収まり状況）', fontsize=11, weight='bold')
    ax2.set_xlabel('前後位置 X [mm]', fontsize=10)
    ax2.set_ylabel('幅 Y [mm]', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    img_path = out_dir / '実寸断面_クリアランス確認図.png'
    fig.savefig(img_path, dpi=200)
    plt.close()
    print(f"Saved visualization to: {img_path}")

if __name__ == '__main__':
    create_plot()
