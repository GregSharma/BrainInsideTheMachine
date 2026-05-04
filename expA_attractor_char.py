"""Experiment A: Attractor Characterization
Fit exponential decay to Exp E propagation curves.
Extract time constant τ at each swap layer. Compare to gate_z from Exp 6.
"""

import json
import numpy as np
from scipy.optimize import curve_fit

# Load data
with open("output/expE_propagation_tracking.json") as f:
    expE = json.load(f)

with open("output/exp6_mlp_nullspace.json") as f:
    exp6 = json.load(f)

# Build gate_z lookup: layer -> gate_z
gate_z_by_layer = {d["layer"]: d["gate_z"] for d in exp6["per_layer"]}

results = {}

for swap_layer_str, tracking in expE["tracking"].items():
    swap_layer = int(swap_layer_str)

    # Extract flip_fraction for layers AFTER the swap
    layers = []
    flip_fractions = []
    for layer_str, data in sorted(tracking.items(), key=lambda x: int(x[0])):
        layer = int(layer_str)
        if layer >= swap_layer and data["flip_fraction"] > 0:
            layers.append(layer)
            flip_fractions.append(data["flip_fraction"])

    layers = np.array(layers)
    flip_fractions = np.array(flip_fractions)

    print(f"\n=== Swap at L{swap_layer} ===")
    print(f"Layers with data: {layers.tolist()}")
    print(f"Flip fractions: {[f'{x:.3f}' for x in flip_fractions]}")

    # Model: flip(L) = 1.0 + A * exp(-(L - L_swap) / τ)
    # where 1.0 = perfect convergence (100%), A = overshoot amplitude
    # flip_fraction is already in units where 1.0 = 100%

    def decay_model(L, A, tau):
        return 1.0 + A * np.exp(-(L - swap_layer) / tau)

    # Initial guesses
    A0 = flip_fractions[0] - 1.0  # overshoot at swap layer
    tau0 = 5.0  # initial guess for time constant

    try:
        popt, pcov = curve_fit(decay_model, layers, flip_fractions,
                               p0=[A0, tau0], maxfev=10000,
                               bounds=([0, 0.1], [10, 50]))
        A_fit, tau_fit = popt
        perr = np.sqrt(np.diag(pcov))

        # Compute R²
        predicted = decay_model(layers, A_fit, tau_fit)
        ss_res = np.sum((flip_fractions - predicted)**2)
        ss_tot = np.sum((flip_fractions - np.mean(flip_fractions))**2)
        r_squared = 1 - ss_res / ss_tot

        print(f"\nFit: flip(L) = 1.0 + {A_fit:.3f} * exp(-(L-{swap_layer})/{tau_fit:.2f})")
        print(f"A (overshoot) = {A_fit:.4f} ± {perr[0]:.4f}")
        print(f"τ (time constant) = {tau_fit:.2f} ± {perr[1]:.2f} layers")
        print(f"R² = {r_squared:.4f}")
        print(f"Half-life = {tau_fit * np.log(2):.2f} layers")

        # Mean gate_z for layers after swap
        post_swap_gate_z = [gate_z_by_layer[l] for l in range(swap_layer, 36) if l in gate_z_by_layer]
        mean_gate_z = np.mean(post_swap_gate_z)
        abs_mean_gate_z = np.mean([abs(g) for g in post_swap_gate_z])

        # Also compute mean of POSITIVE gate_z only (strong readers)
        pos_gate_z = [g for g in post_swap_gate_z if g > 0]
        mean_pos_gate_z = np.mean(pos_gate_z) if pos_gate_z else 0

        print(f"\nMean gate_z (L{swap_layer}-L35): {mean_gate_z:.3f}")
        print(f"Mean |gate_z|: {abs_mean_gate_z:.3f}")
        print(f"Mean positive gate_z: {mean_pos_gate_z:.3f}")
        print(f"1/τ = {1/tau_fit:.4f}")

        # Store results
        results[f"swap_L{swap_layer}"] = {
            "A_overshoot": float(A_fit),
            "A_err": float(perr[0]),
            "tau": float(tau_fit),
            "tau_err": float(perr[1]),
            "r_squared": float(r_squared),
            "half_life": float(tau_fit * np.log(2)),
            "mean_gate_z_post_swap": float(mean_gate_z),
            "mean_abs_gate_z_post_swap": float(abs_mean_gate_z),
            "mean_positive_gate_z": float(mean_pos_gate_z),
            "inverse_tau": float(1/tau_fit),
            "n_decay_layers": int(35 - swap_layer),
            "initial_flip": float(flip_fractions[0]),
            "final_flip": float(flip_fractions[-1]),
            "fit_layers": layers.tolist(),
            "fit_predicted": predicted.tolist(),
            "fit_actual": flip_fractions.tolist(),
        }

    except Exception as e:
        print(f"Fit failed: {e}")
        # Try manual analysis
        print("Falling back to manual half-life estimation...")

        # Find when overshoot drops to half
        overshoot_0 = flip_fractions[0] - 1.0
        half_target = 1.0 + overshoot_0 / 2
        for i, (l, f) in enumerate(zip(layers, flip_fractions)):
            if f <= half_target:
                print(f"  Half-life reached at L{l} (flip={f:.3f}, target={half_target:.3f})")
                break

# Cross-swap comparison
print("\n\n=== CROSS-SWAP COMPARISON ===")
if "swap_L12" in results and "swap_L26" in results:
    r12 = results["swap_L12"]
    r26 = results["swap_L26"]
    print(f"{'Metric':<30} {'Swap@L12':>12} {'Swap@L26':>12}")
    print("-" * 56)
    print(f"{'Overshoot (A)':<30} {r12['A_overshoot']:>12.3f} {r26['A_overshoot']:>12.3f}")
    print(f"{'τ (layers)':<30} {r12['tau']:>12.2f} {r26['tau']:>12.2f}")
    print(f"{'Half-life (layers)':<30} {r12['half_life']:>12.2f} {r26['half_life']:>12.2f}")
    print(f"{'R²':<30} {r12['r_squared']:>12.4f} {r26['r_squared']:>12.4f}")
    print(f"{'Mean gate_z post-swap':<30} {r12['mean_gate_z_post_swap']:>12.3f} {r26['mean_gate_z_post_swap']:>12.3f}")
    print(f"{'Mean |gate_z| post-swap':<30} {r12['mean_abs_gate_z_post_swap']:>12.3f} {r26['mean_abs_gate_z_post_swap']:>12.3f}")
    print(f"{'Mean positive gate_z':<30} {r12['mean_positive_gate_z']:>12.3f} {r26['mean_positive_gate_z']:>12.3f}")
    print(f"{'1/τ':<30} {r12['inverse_tau']:>12.4f} {r26['inverse_tau']:>12.4f}")
    print(f"{'Layers to converge':<30} {r12['n_decay_layers']:>12d} {r26['n_decay_layers']:>12d}")

    # Is τ ∝ 1/gate_z?
    ratio_tau = r26["tau"] / r12["tau"]
    ratio_gate_z = r12["mean_positive_gate_z"] / r26["mean_positive_gate_z"]
    print(f"\nτ ratio (L26/L12): {ratio_tau:.3f}")
    print(f"Inverse gate_z ratio (L12/L26): {ratio_gate_z:.3f}")
    print(f"If τ ∝ 1/gate_z, these ratios should be similar.")
    print(f"Match quality: {min(ratio_tau, ratio_gate_z)/max(ratio_tau, ratio_gate_z)*100:.1f}%")

# Per-layer gate_z profile for reference
print("\n\n=== GATE_Z PROFILE (all layers) ===")
print(f"{'Layer':>5} {'gate_z':>8} {'|gate_z|':>8}")
for l in range(36):
    gz = gate_z_by_layer.get(l, 0)
    marker = " ***" if abs(gz) > 5 else ""
    print(f"L{l:>3}  {gz:>8.2f} {abs(gz):>8.2f}{marker}")

# Save
with open("output/expA_attractor_characterization.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to output/expA_attractor_characterization.json")
