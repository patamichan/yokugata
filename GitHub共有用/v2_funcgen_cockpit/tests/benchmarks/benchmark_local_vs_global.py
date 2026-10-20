"""Minimal Benchmark: Local Deformation vs Global Chord Scaling in Cockpit Sizing.

Evaluates:
1. Which pilot envelope constraint binds the chord length in Method A (global scaling)?
2. How much margin exists at other body stations (Knees, Hips, Shoulders) because of Feet binding?
3. If Feet/Pedal is accommodated locally, what would be the ideal chord governed by Shoulders?
4. What is the theoretical wetted area reduction?
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from core.geometry import generate_funcgen_airfoil
from core.optimizer import calculate_required_chord, PRESET_PILOT_STANDARD, _profile_arrays

def run_benchmark():
    print("=== Pilot Envelope Stations ===")
    distances, widths = _profile_arrays(PRESET_PILOT_STANDARD, nose_margin_mm=30.0)
    for p, d, w in zip(PRESET_PILOT_STANDARD, distances, widths):
        print(f"Point: {p['name']:12s} | Dist: {d:6.1f} mm | Req Width: {w:6.1f} mm")

    tc, xt = 24.0, 42.0
    print("\n=== Method A: Binding & Margin Analysis ===")
    for kle in [0.8, 1.0, 1.2]:
        for method in ['warp', 'cst']:
            c = calculate_required_chord(tc, xt, kle, PRESET_PILOT_STANDARD, generator_method=method)
            x, y = generate_funcgen_airfoil(tc, xt, kle, num_points=2001, method=method)
            nose = int(np.argmin(x))
            surface = PchipInterpolator(np.sqrt(x[:nose+1][::-1]), y[:nose+1][::-1], extrapolate=False)
            avail = 2 * c * surface(np.sqrt(distances / c))
            margins = avail - widths
            
            # Theoretical chord if ONLY Shoulder was considered (the widest structural point: 510 mm)
            # Max possible thickness is 2 * c * (tc/100) = 0.24 * 2 * c = 0.48 * c
            # At shoulder (x=1230mm), what chord would barely fit Shoulder?
            # Let's solve minimum chord required if Feet didn't exist!
            shoulder_profile = [PRESET_PILOT_STANDARD[-1]]
            c_shoulder_only = calculate_required_chord(tc, xt, kle, shoulder_profile, generator_method=method)
            
            wetted_A = 2 * c # proportional to chord
            wetted_Shoulder = 2 * c_shoulder_only
            wetted_reduction_pct = (1.0 - c_shoulder_only / c) * 100.0

            print(f"\n[{method.upper():4s}] kle={kle:.1f}:")
            print(f"  Chord Required (All pts): {c:.1f} mm")
            print(f"  Margins: Feet={margins[0]:.1f}mm, Knees={margins[1]:.1f}mm, Hips={margins[2]:.1f}mm, Shoulders={margins[3]:.1f}mm")
            print(f"  If Feet decoupled (Shoulder-only bound): Chord = {c_shoulder_only:.1f} mm (Reduction: {wetted_reduction_pct:.1f}%)")

if __name__ == '__main__':
    run_benchmark()
