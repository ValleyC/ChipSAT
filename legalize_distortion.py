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

            # ── displacement metrics (noisy → legal) ──────────────────────────
            per_macro_disp = np.sqrt(((legal - noisy) ** 2).sum(axis=1))  # (N,)
            disp_mean   = float(per_macro_disp.mean())
            disp_max    = float(per_macro_disp.max())
            disp_p90    = float(np.percentile(per_macro_disp, 90))
            # fraction of macros displaced more than σ (the noise level itself)
            frac_moved_gt_sigma = float((per_macro_disp > sigma).mean())
            # fraction displaced more than one full window (wf=0.10)
            frac_moved_gt_wf    = float((per_macro_disp > 0.10).mean())

            # ── topology preservation (reference → legal) ─────────────────────
            # Spearman rank correlation of x and y coordinates separately.
            # 1.0 = perfect order preserved, 0 = random, -1 = reversed.
            from scipy.stats import spearmanr
            rho_x = float(spearmanr(positions[:, 0], legal[:, 0]).statistic)
            rho_y = float(spearmanr(positions[:, 1], legal[:, 1]).statistic)

            # total distortion from reference baseline
            disp_ref = float(np.sqrt(((legal - positions) ** 2).sum(axis=1)).mean())

            trial_results.append({
                'sigma':              float(sigma),
                'trial':              trial,
                'hpwl_ref':           float(ref_hpwl),
                'hpwl_noisy':         float(hpwl_noisy),
                'hpwl_legal':         float(hpwl_legal),
                'ratio_noisy':        float(hpwl_noisy / ref_hpwl),
                'ratio_legal':        float(hpwl_legal / ref_hpwl),
                'ov_noisy':           int(ov_noisy),
                'ov_legal':           int(ov_legal),
                'disp_mean':          disp_mean,
                'disp_max':           disp_max,
                'disp_p90':           disp_p90,
                'frac_moved_gt_sigma':frac_moved_gt_sigma,
                'frac_moved_gt_wf':   frac_moved_gt_wf,
                'spearman_x':         rho_x,
                'spearman_y':         rho_y,
                'disp_ref':           disp_ref,
            })
            print(f"  σ={sigma:.3f} t={trial}: "
                  f"noisy={hpwl_noisy/ref_hpwl:.3f}x  "
                  f"legal={hpwl_legal/ref_hpwl:.3f}x  "
                  f"disp_mean={disp_mean:.4f} max={disp_max:.4f} p90={disp_p90:.4f}  "
                  f"moved>σ={frac_moved_gt_sigma:.1%}  "
                  f"ρx={rho_x:.3f} ρy={rho_y:.3f}")

        if trial_results:
            def _avg(key): return float(np.mean([r[key] for r in trial_results]))
            avg = {
                'sigma':               float(sigma),
                'hpwl_ref':            float(ref_hpwl),
                'hpwl_noisy':          _avg('hpwl_noisy'),
                'hpwl_legal':          _avg('hpwl_legal'),
                'ratio_noisy':         _avg('ratio_noisy'),
                'ratio_legal':         _avg('ratio_legal'),
                'ov_noisy':            _avg('ov_noisy'),
                'disp_mean':           _avg('disp_mean'),
                'disp_max':            _avg('disp_max'),
                'disp_p90':            _avg('disp_p90'),
                'frac_moved_gt_sigma': _avg('frac_moved_gt_sigma'),
                'frac_moved_gt_wf':    _avg('frac_moved_gt_wf'),
                'spearman_x':          _avg('spearman_x'),
                'spearman_y':          _avg('spearman_y'),
                'disp_ref':            _avg('disp_ref'),
            }
            results.append(avg)

    return results, ref_hpwl


def plot_results(results, circuit, ref_hpwl, output_dir):
    sigmas               = [r['sigma']               for r in results]
    ratio_noisy          = [r['ratio_noisy']          for r in results]
    ratio_legal          = [r['ratio_legal']          for r in results]
    disp_mean            = [r['disp_mean']            for r in results]
    disp_max             = [r['disp_max']             for r in results]
    disp_p90             = [r['disp_p90']             for r in results]
    frac_gt_sigma        = [r['frac_moved_gt_sigma']  for r in results]
    frac_gt_wf           = [r['frac_moved_gt_wf']     for r in results]
    spearman_x           = [r['spearman_x']           for r in results]
    spearman_y           = [r['spearman_y']           for r in results]
    ov_noisy             = [r['ov_noisy']             for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
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
            label='Noisy input (ML target)')
    ax.plot(sigmas, ratio_legal, 's-',  color='#e94560', lw=1.8,
            label='After CP-SAT legalize')
    ax.axhline(1.0, color='white', lw=1, ls=':', alpha=0.6, label='Reference')
    ax.legend(fontsize=8, facecolor=BG_DARK, labelcolor='white')
    _style(ax, 'Noise σ', 'HPWL / ref', 'HPWL: before vs after legalization')

    # Panel 2: Displacement distribution vs sigma
    ax = axes[1]
    ax.plot(sigmas, disp_mean, 'o-',  color='#a8dadc', lw=1.8, label='Mean')
    ax.plot(sigmas, disp_p90,  's--', color='#ffd700', lw=1.8, label='90th pct')
    ax.plot(sigmas, disp_max,  '^:',  color='#f4a261', lw=1.5, label='Max')
    ax.plot(sigmas, sigmas,    'w:',  lw=1.0,                   label='σ (noise level)')
    ax.legend(fontsize=8, facecolor=BG_DARK, labelcolor='white')
    _style(ax, 'Noise σ', 'Displacement (fraction of chip)',
           'How far does CP-SAT move macros\nto fix overlaps?')

    # Panel 3: Fraction of macros displaced beyond thresholds
    ax = axes[2]
    ax.plot(sigmas, frac_gt_sigma, 'o-',  color='#e94560', lw=1.8,
            label='Displaced > σ (input noise)')
    ax.plot(sigmas, frac_gt_wf,   's--', color='#f4a261', lw=1.8,
            label='Displaced > wf=0.10 (one full window)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor=BG_DARK, labelcolor='white')
    _style(ax, 'Noise σ', 'Fraction of macros',
           'What fraction of macros get moved\nmore than the noise level?')

    # Panel 4: Topology preservation — Spearman rank correlation
    ax = axes[3]
    ax.plot(sigmas, spearman_x, 'o-',  color='#a8dadc', lw=1.8, label='ρ(x)')
    ax.plot(sigmas, spearman_y, 's--', color='#e94560', lw=1.8, label='ρ(y)')
    ax.axhline(1.0, color='white', lw=0.8, ls=':', alpha=0.5)
    ax.axhline(0.0, color='white', lw=0.8, ls=':', alpha=0.3)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, facecolor=BG_DARK, labelcolor='white')
    _style(ax, 'Noise σ', 'Spearman ρ (ref vs legal)',
           'Topology preservation:\ndoes relative macro ordering survive legalization?')

    # Panel 5: Overlap count in noisy input
    ax = axes[4]
    ax.plot(sigmas, ov_noisy, 'o--', color='#ffd700', lw=1.8)
    _style(ax, 'Noise σ', 'Overlapping pairs',
           'Overlap count in noisy input\n(legalization workload)')

    # Panel 6: HPWL gap closed (how much does legalization recover?)
    gap_closed = [
        (r['ratio_noisy'] - r['ratio_legal']) / max(r['ratio_noisy'] - 1.0, 1e-8)
        for r in results
    ]
    ax = axes[5]
    ax.plot(sigmas, gap_closed, 'o-', color='#a8dadc', lw=1.8)
    ax.axhline(0.0, color='white', lw=0.8, ls=':', alpha=0.4)
    ax.axhline(1.0, color='white', lw=0.8, ls=':', alpha=0.4)
    _style(ax, 'Noise σ', 'Fraction of HPWL gap recovered',
           'Does legalization recover HPWL lost to noise?\n'
           '(1.0 = fully recovers, 0 = makes it worse)')

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
    print(f"\n{'σ':>6}  {'noisy/ref':>9}  {'legal/ref':>9}  "
          f"{'disp_mean':>9}  {'disp_p90':>8}  {'moved>σ':>7}  "
          f"{'ρx':>6}  {'ρy':>6}")
    print('-' * 75)
    for r in results:
        print(f"{r['sigma']:6.3f}  {r['ratio_noisy']:9.3f}  "
              f"{r['ratio_legal']:9.3f}  {r['disp_mean']:9.4f}  "
              f"{r['disp_p90']:8.4f}  {r['frac_moved_gt_sigma']:7.1%}  "
              f"{r['spearman_x']:6.3f}  {r['spearman_y']:6.3f}")

    plot_results(results, args.circuit, ref_hpwl, args.output_dir)


if __name__ == '__main__':
    main()
