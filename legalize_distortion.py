"""
Legalization distortion experiment.

Answers: "If an ML model generates positions with noise sigma,
how much HPWL is preserved after CP-SAT legalization?"

For each noise level σ:
  1. Start from reference legal placement
  2. Add Gaussian noise N(0, σ) to all macro centers → illegal
  3. Run CP-SAT legalize() (minimize_displacement=True)
  4. Record: HPWL before/after noise, after legalization,
             mean displacement, overlap count

Usage:
    python legalize_distortion.py --circuit ibm01
    python legalize_distortion.py --circuit ibm01 --n_trials 5
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import legalize, compute_net_hpwl, check_overlap

BG_DARK  = '#1a1a2e'
BG_PANEL = '#16213e'


def run_distortion_experiment(circuit, benchmark_base, noise_sigmas,
                               n_trials=3, seed=42):
    circuit_dir = os.path.join(benchmark_base, 'iccad04', 'extracted', circuit)
    data = load_bookshelf_circuit(circuit_dir, circuit, macros_only=True, seed=seed)

    positions = data['positions'].copy()
    sizes     = data['node_features']
    nets      = data['nets']

    # Ensure starting positions are legal
    _, ov = check_overlap(positions, sizes)
    if ov > 0:
        print(f"Legalizing reference ({ov} overlapping pairs)...")
        positions = legalize(positions, sizes, time_limit=300.0,
                             window_fraction=0.3, num_workers=4)
        if positions is None:
            raise RuntimeError("Reference legalization failed")

    ref_hpwl = compute_net_hpwl(positions, sizes, nets)
    _, ref_ov = check_overlap(positions, sizes)
    print(f"Circuit: {circuit}, N={len(positions)}, "
          f"ref HPWL={ref_hpwl:.4f}, overlaps={ref_ov}")

    rng = np.random.default_rng(seed)
    results = []

    for sigma in noise_sigmas:
        trial_results = []
        for trial in range(n_trials):
            # Add Gaussian noise to all macro centers
            noise = rng.normal(0.0, sigma, size=positions.shape)
            noisy = positions + noise

            # Clamp to chip boundary (prevent going fully outside [-1,1])
            noisy = np.clip(noisy, -1.0 + 1e-4, 1.0 - 1e-4)

            hpwl_noisy = compute_net_hpwl(noisy, sizes, nets)
            _, ov_noisy = check_overlap(noisy, sizes)

            # CP-SAT legalization (minimize displacement from noisy positions)
            legal = legalize(noisy, sizes, time_limit=60.0,
                             window_fraction=0.5, num_workers=4,
                             minimize_displacement=True)

            if legal is None:
                print(f"  σ={sigma:.3f} trial={trial}: legalization FAILED")
                continue

            hpwl_legal = compute_net_hpwl(legal, sizes, nets)
            _, ov_legal = check_overlap(legal, sizes)

            # Mean displacement from noisy → legal (how much did CP-SAT move things)
            disp_legal = float(np.abs(legal - noisy).mean())
            # Mean displacement from reference → legal (total distortion)
            disp_ref   = float(np.abs(legal - positions).mean())

            trial_results.append({
                'sigma':       float(sigma),
                'trial':       trial,
                'hpwl_ref':    float(ref_hpwl),
                'hpwl_noisy':  float(hpwl_noisy),
                'hpwl_legal':  float(hpwl_legal),
                'ratio_noisy': float(hpwl_noisy / ref_hpwl),
                'ratio_legal': float(hpwl_legal / ref_hpwl),
                'ov_noisy':    int(ov_noisy),
                'ov_legal':    int(ov_legal),
                'disp_legal':  disp_legal,
                'disp_ref':    disp_ref,
            })
            print(f"  σ={sigma:.3f} t={trial}: "
                  f"noisy={hpwl_noisy:.3f} ({hpwl_noisy/ref_hpwl:.3f}x)  "
                  f"legal={hpwl_legal:.3f} ({hpwl_legal/ref_hpwl:.3f}x)  "
                  f"ov_in={ov_noisy} ov_out={ov_legal}  "
                  f"disp={disp_legal:.4f}")

        if trial_results:
            # Average across trials
            avg = {
                'sigma':       float(sigma),
                'hpwl_ref':    float(ref_hpwl),
                'hpwl_noisy':  float(np.mean([r['hpwl_noisy'] for r in trial_results])),
                'hpwl_legal':  float(np.mean([r['hpwl_legal'] for r in trial_results])),
                'ratio_noisy': float(np.mean([r['ratio_noisy'] for r in trial_results])),
                'ratio_legal': float(np.mean([r['ratio_legal'] for r in trial_results])),
                'ov_noisy':    float(np.mean([r['ov_noisy']   for r in trial_results])),
                'disp_legal':  float(np.mean([r['disp_legal'] for r in trial_results])),
                'disp_ref':    float(np.mean([r['disp_ref']   for r in trial_results])),
            }
            results.append(avg)

    return results, ref_hpwl


def plot_results(results, circuit, ref_hpwl, output_dir):
    sigmas      = [r['sigma']       for r in results]
    ratio_noisy = [r['ratio_noisy'] for r in results]
    ratio_legal = [r['ratio_legal'] for r in results]
    disp_legal  = [r['disp_legal']  for r in results]
    ov_noisy    = [r['ov_noisy']    for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG_DARK)

    def _style(ax, xlabel, ylabel, title):
        ax.set_facecolor(BG_PANEL)
        ax.set_xlabel(xlabel, color='white', fontsize=10)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='white', labelsize=9)
        for s in ('bottom', 'left'):
            ax.spines[s].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Panel 1: HPWL ratio vs sigma
    ax = axes[0]
    ax.plot(sigmas, ratio_noisy, 'o--', color='#f4a261', lw=1.8,
            label='Noisy (pre-legal, target for ML)')
    ax.plot(sigmas, ratio_legal, 's-',  color='#e94560', lw=1.8,
            label='After CP-SAT legalize')
    ax.axhline(1.0, color='white', lw=1, ls=':', alpha=0.6, label='Reference (1.0×)')
    ax.legend(fontsize=8, facecolor=BG_DARK, labelcolor='white')
    _style(ax, 'Noise σ (fraction of chip)', 'HPWL / reference HPWL',
           'HPWL distortion from legalization')

    # Panel 2: Mean displacement legalize induced vs sigma
    ax = axes[1]
    ax.plot(sigmas, disp_legal, 's-', color='#a8dadc', lw=1.8)
    ax.axhline(0, color='white', lw=0.5, ls=':', alpha=0.4)
    _style(ax, 'Noise σ', 'Mean |Δpos| per macro',
           'How far does CP-SAT move macros\nto fix overlaps?')

    # Panel 3: Overlap pairs in noisy placement
    ax = axes[2]
    ax.plot(sigmas, ov_noisy, 'o--', color='#ffd700', lw=1.8)
    _style(ax, 'Noise σ', 'Overlapping macro pairs',
           'Overlap count in noisy placement\n(input to legalization)')

    plt.suptitle(
        f'Legalization distortion — {circuit}  |  '
        f'ref HPWL={ref_hpwl:.2f}',
        color='white', fontsize=13)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f'{circuit}_legalize_distortion.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--circuit',        default='ibm01')
    parser.add_argument('--benchmark_base', default='benchmarks')
    parser.add_argument('--output_dir',     default='viz_output')
    parser.add_argument('--n_trials', type=int, default=3,
                        help='Trials per sigma (averaged)')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--sigmas', nargs='+', type=float,
                        default=[0.01, 0.02, 0.05, 0.08, 0.10,
                                 0.15, 0.20, 0.30, 0.50],
                        help='Noise levels to test (fraction of chip)')
    args = parser.parse_args()

    results, ref_hpwl = run_distortion_experiment(
        args.circuit, args.benchmark_base,
        noise_sigmas=args.sigmas,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    # Save raw numbers
    os.makedirs(args.output_dir, exist_ok=True)
    json_out = os.path.join(args.output_dir, f'{args.circuit}_distortion.json')
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw results → {json_out}")

    # Summary table
    print(f"\n{'σ':>6}  {'HPWL_noisy/ref':>14}  {'HPWL_legal/ref':>14}  "
          f"{'ov_noisy':>8}  {'disp_legal':>10}")
    print('-' * 62)
    for r in results:
        print(f"{r['sigma']:6.3f}  {r['ratio_noisy']:14.3f}  "
              f"{r['ratio_legal']:14.3f}  {r['ov_noisy']:8.0f}  "
              f"{r['disp_legal']:10.4f}")

    plot_results(results, args.circuit, ref_hpwl, args.output_dir)


if __name__ == '__main__':
    main()
