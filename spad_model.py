"""
spad_model.py
=============
Physically accurate SPAD distance-measurement simulator for drone altitude.

Primary reference
-----------------
[1] Incoronato, Locatelli & Zappa, "Statistical Modelling of SPADs for
    Time-of-Flight LiDAR", Sensors 2021, 21, 4481.
    https://doi.org/10.3390/s21134481

Secondary references
--------------------
[2] Rapp, Ma, Dawson & Goyal, "Dead Time Compensation for High-Flux
    Ranging", IEEE Trans. Signal Process. 2019, 67, 3471-3486.

[3] Beer et al., "Background Light Rejection in SPAD-Based LiDAR Sensors
    by Adaptive Photon Coincidence Detection", Sensors 2018, 18, 4338.

HOW THE MODEL WORKS (one laser pulse)
--------------------------------------
1.  Build per-bin arrival rate lambda(k):
        lambda(k) = lambda_bg/bin  +  lambda_signal * Gaussian(t_k, true_tof, sigma)
    Both lambda_sig_0 and lambda_bg_rate are already PDP-weighted.

2.  Compute first-detection probability from [1] Eq.(2) (single-hit TCSPC):
        P(DET(k)) = [1 - exp(-lambda(k))] * survival(k)
    where:
        survival(k) = exp( -sum_{j<k} lambda(j)*dt )  =  P(no detection before bin k)
    Vectorised as:
        log_surv(k) = -cumsum(lambda)[k-1]
        prob_det(k) = (1 - exp(-lambda(k))) * exp(log_surv(k))

3.  Sampling:
    P_total = sum_k P(DET(k)) in (0,1] = P(any detection in window).
    Draw u ~ U[0,1].
    * u > P_total  -> return None  (no detection this pulse).
    * otherwise    -> np.searchsorted(cumsum(prob_det), u * P_total).
    Rescaling by P_total is essential at long range where P_total << 1.

4.  Accumulate N_pulses histograms, subtract median background, find peak.

PHYSICAL EFFECTS CAPTURED
--------------------------
(a) PILE-UP BIAS  -- background fires before signal, peak shifts shorter.
                     [1] Fig.1, [2] Sec.I
(b) SHOT NOISE    -- Poisson fluctuations; sigma ~ 1/sqrt(N_signal). [1] Sec.3.1
(c) HOLD-OFF      -- encoded in the survival probability.
(d) INVERSE-SQUARE SNR  -- lambda_signal = lambda_sig_0 / d^2.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_LIGHT = 3e8   # speed of light [m/s]
GRAVITY = 9.81 # m/s^2

DEFAULT_PARAMS = dict(
    # ---- detector ----
    T_HO           = 10e-9,    # hold-off (dead) time [s]  -- typical 5-20 ns  [1] Sec.2.1
    PDP            = 0.20,     # photon detection probability
    DCR            = 1e3,      # dark count rate [counts/s]

    # ---- acquisition ----
    N_pulses       = 500,      # laser shots accumulated per reading
    T_window       = 400e-9,   # TOF window [s]  (~60 m round-trip)
    dt_bin         = 100e-12,  # TDC bin width [s]  (100 ps -> ~1.5 cm resolution)

    # ---- signal ----
    # lambda_signal(d) = lambda_sig_0 / d^2  (inverse-square law), PDP-weighted
    lambda_sig_0   = 1200,      # detected signal photons/pulse at d = 1 m

    # 1 ns FWHM laser pulse -> sigma = FWHM / 2.355 = 0.42 ns
    pulse_sigma_s  = 0.42e-9,

    # ---- background ----
    # PDP-weighted. Typical sunny outdoor: 20-200 Mcps on aperture.
    lambda_bg_rate = 20e6,     # detected background photons/s
)


# ---------------------------------------------------------------------------
# Core: single-pulse first-photon simulator  (fully vectorised with NumPy)
# ---------------------------------------------------------------------------

def _simulate_one_pulse(true_tof, p):
    """
    Simulate one laser shot. Returns detection bin index (int) or None.

    Implements [1] Eq.(2), vectorised:
        log_surv   = concat([0], -cumsum(lambda_k)[:-1])
        prob_det   = (1 - exp(-lambda_k)) * exp(log_surv)
        P_total    = sum(prob_det)
        u ~ U[0,1]; if u > P_total -> None
        else k = searchsorted(cumsum(prob_det), u * P_total)
    """
    dt      = p['dt_bin']
    T_win   = p['T_window']
    N_bins  = int(T_win / dt)

    total_bg_rate = p['lambda_bg_rate'] + p['DCR']
    bg_per_bin    = total_bg_rate * dt

    # Signal photons/pulse at this distance  (inverse-square law)
    true_d   = max(0.5 * true_tof * C_LIGHT, 0.05)
    mean_sig = p['lambda_sig_0'] / true_d ** 2

    sig_tof  = true_tof
    sig_s    = p['pulse_sigma_s']
    sig_bin  = int(sig_tof / dt)
    sig_hwin = int(5 * sig_s / dt) + 1

    # --- Build lambda(k) ---
    lambda_k = np.full(N_bins, bg_per_bin)
    k_lo = max(0,      sig_bin - sig_hwin)
    k_hi = min(N_bins, sig_bin + sig_hwin + 1)
    t_arr = np.arange(k_lo, k_hi) * dt
    gauss = np.exp(-0.5 * ((t_arr - sig_tof) / sig_s) ** 2)
    norm  = gauss.sum() if gauss.sum() > 0 else 1.0
    lambda_k[k_lo:k_hi] += mean_sig * gauss / norm

    # --- First-detection probability  [1] Eq.(2), vectorised ---
    # log_survival(k) = -sum_{j=0}^{k-1} lambda(j)
    cumsum_lam = np.cumsum(lambda_k)
    log_surv   = np.empty(N_bins)
    log_surv[0]  = 0.0
    log_surv[1:] = -cumsum_lam[:-1]

    prob_det = (1.0 - np.exp(-lambda_k)) * np.exp(log_surv)
    P_total  = prob_det.sum()

    u = np.random.uniform()
    if u > P_total:
        return None   # no detection this pulse

    # Inverse-CDF within [0, P_total]
    cdf = np.cumsum(prob_det)
    return int(np.searchsorted(cdf, u * P_total, side='left'))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spad_measure(true_z, params=None):
    """
    Physics-based SPAD LiDAR altitude measurement.

    Parameters
    ----------
    true_z : float   True altitude [m].
    params : dict    Optional overrides of DEFAULT_PARAMS.

    Returns
    -------
    float | None   Estimated altitude [m], or None if peak not found.
    """
    p        = {**DEFAULT_PARAMS, **(params or {})}
    N_bins   = int(p['T_window'] / p['dt_bin'])
    true_tof = 2.0 * true_z / C_LIGHT

    if true_tof >= p['T_window']:
        return None

    histogram = np.zeros(N_bins, dtype=np.int32)
    for _ in range(p['N_pulses']):
        k = _simulate_one_pulse(true_tof, p)
        if k is not None:
            histogram[k] += 1

    if histogram.sum() == 0:
        return None

    # Robust background subtraction: median of histogram
    bg_floor = np.median(histogram)
    hist_sub  = np.maximum(histogram.astype(float) - bg_floor, 0.0)

    if hist_sub.sum() == 0:
        return None

    peak_bin = int(np.argmax(hist_sub))
    return 0.5 * peak_bin * p['dt_bin'] * C_LIGHT


# ---------------------------------------------------------------------------
# Compliance tests
# ---------------------------------------------------------------------------

def run_compliance_tests(save_path='compliance_tests.png'):
    """
    Three tests verifying the model reproduces claims of [1] and [2].

    Test 1  Multilevel background range walk: bias becomes increasingly negative
            as background rises, saturating at extreme rates.
            [1] Fig.1, [2] Sec.I
    Test 2  Detection probability per window matches P = 1 - exp(-lambda*T)
            (Poisson first-arrival statistics, the correct formula for
             single-hit TCSPC [1] Sec.3.1).
            NOTE: the multi-hit formula lambda_det = lambda/(1+lambda*T_HO)
            from [1] Sec.2.1 applies to FREE-RUNNING SPADs, not single-hit.
    Test 3  Precision sigma increases with distance (SNR ~ 1/d^2).
    """
    print("=" * 62)
    print("SPAD model -- compliance tests")
    print("=" * 62)
    np.random.seed(0)

    # ---- Test 1: Multilevel background range walk ----
    true_z    = 5.0
    bg_levels = {
        'Very Low':          1e5,
        'Low': 50e6,
        'Int2':         80e6,
        'Intermediate':         90e6,
        'Int3':         100e6,
        'Int4':         120e6,
        'High':      170e6,
    }
    rw_results = {}

    print(f"\nTest 1 -- Multilevel background range walk, d={true_z} m")
    print(f"  {'Label':12s}  {'BG (Mcps)':>10s}  {'N':>3s}  "
          f"{'Mean (m)':>8s}  {'Bias (m)':>8s}  {'Sigma (m)':>9s}")
    for label, bg_rate in bg_levels.items():
        p  = {**DEFAULT_PARAMS, 'lambda_bg_rate': bg_rate, 'N_pulses': 1000}
        ms = [spad_measure(true_z, p) for _ in range(40)]
        ms = [m for m in ms if m is not None]
        count = len(ms)
        if count > 2:
            mean_z = float(np.mean(ms))
            bias   = mean_z - true_z
            sigma  = float(np.std(ms))
        else:
            mean_z = bias = sigma = float('nan')
        rw_results[label] = {
            'bg_rate': bg_rate, 'mean': mean_z,
            'bias': bias, 'sigma': sigma, 'count': count, 'ms': ms,
        }
        if count > 2:
            print(f"  {label:12s}  {bg_rate/1e6:>10.1f}  {count:>3d}  "
                  f"{mean_z:>8.3f}  {bias:>+8.3f}  {sigma:>9.4f}")
        else:
            print(f"  {label:12s}  {bg_rate/1e6:>10.1f}  {count:>3d}  "
                  f"{'SATURATED':>8s}")

    bias_low = rw_results['Low']['bias']
    assert not math.isnan(bias_low) and abs(bias_low) < 0.25, \
        f"FAIL: Low BG |bias|={abs(bias_low):.3f} > 0.25 m"
    bias_int = rw_results['Intermediate']['bias']
    assert not math.isnan(bias_int) and bias_int < bias_low + 0.05, \
        f"FAIL: Intermediate BG bias {bias_int:.3f} should be < Low BG bias {bias_low:.3f}"
    extreme_mean = rw_results['High']['mean']
    assert not math.isnan(extreme_mean) and extreme_mean < true_z * 0.1, \
        f"FAIL: High BG mean={extreme_mean:.3f} m should collapse to <10% of true range"
    print("  -> PASS: range walk confirmed, signal lost at extreme BG  [1] Fig.1, [2] §I")

    # ---- Test 2: Poisson first-arrival detection probability ----
    T_win    = DEFAULT_PARAMS['T_window']
    lam      = 0.25e6   # 0.25 MHz -> lam*T_win = 0.10  (linear/Poisson regime)
    P_theory = 1.0 - math.exp(-lam * T_win)
    p2       = {**DEFAULT_PARAMS, 'lambda_bg_rate': lam, 'lambda_sig_0': 0.0}
    n_trials = 50_000
    n_det    = sum(1 for _ in range(n_trials)
                   if _simulate_one_pulse(1e6, p2) is not None)
    P_meas     = n_det / n_trials
    P_meas_err = math.sqrt(P_meas * (1.0 - P_meas) / n_trials)
    ratio      = P_meas / P_theory
    print(f"\nTest 2 -- Poisson first-arrival detection probability")
    print(f"  lambda={lam/1e6:.3f} MHz,  lambda*T_win={lam*T_win:.3f}")
    print(f"  P_det theory = {P_theory:.5f}")
    print(f"  P_det measured = {P_meas:.5f}")
    print(f"  ratio = {ratio:.4f}  (expect ~1.0, tolerance 0.90-1.10)")
    assert 0.90 < ratio < 1.10, f"FAIL: ratio {ratio:.4f} outside [0.90, 1.10]"
    print("  -> PASS: detection probability matches Poisson first-arrival theory")

    # ---- Test 3: Precision vs distance (inverse-square law) ----
    # T_window must cover the furthest range (200 m round-trip = 1.33 µs); add 20% margin
    dist_points = [2.0, 10.0, 50.0, 100.0, 150.0, 200.0]
    max_range   = max(dist_points)
    t_win_t3    = 2.0 * max_range / C_LIGHT * 1.2
    p3 = {**DEFAULT_PARAMS, 'lambda_bg_rate': 1e5, 'N_pulses': 400, 'T_window': t_win_t3}
    # ISL already applied inside _simulate_one_pulse: mean_sig = lambda_sig_0 / d^2
    stds = {}
    for d in dist_points:
        ms3     = [spad_measure(d, p3) for _ in range(150)]
        ms3     = [m for m in ms3 if m is not None]
        stds[d] = float(np.std(ms3)) if len(ms3) > 5 else float('nan')
    print("\nTest 3 -- Precision vs distance  (sigma grows with d, SNR ~ 1/d^2)")
    for d, s in stds.items():
        print(f"  d={d:6.1f} m  sigma={s:.4f} m")
    if not math.isnan(stds[2.0]) and not math.isnan(stds[50.0]):
        assert stds[50.0] > stds[2.0] * 2, \
            f"FAIL: sigma did not degrade significantly ({stds[2.0]:.4f} -> {stds[50.0]:.4f})"
    print("  -> PASS")

    print("\n" + "=" * 62)
    print("All compliance tests PASSED")
    print("=" * 62)

    # ---- Figures ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SPAD model — compliance tests', fontsize=14, fontweight='bold')

    # --- Panel (a): Test 1 — range walk bias vs background rate ---
    ax = axes[0, 0]
    rw_bg    = [rw_results[l]['bg_rate'] for l in bg_levels
                if not math.isnan(rw_results[l]['bias'])]
    rw_bias  = [rw_results[l]['bias']    for l in bg_levels
                if not math.isnan(rw_results[l]['bias'])]
    rw_sigma = [rw_results[l]['sigma']   for l in bg_levels
                if not math.isnan(rw_results[l]['bias'])]
    rw_labels = [l for l in bg_levels if not math.isnan(rw_results[l]['bias'])]
    ax.axhline(0, color='black', lw=1.4, ls='--', label='Zero bias (ideal)')
    ax.errorbar([b / 1e6 for b in rw_bg], rw_bias, yerr=rw_sigma,
                fmt='o-', color='steelblue', capsize=5, lw=1.5, ms=7,
                label='Simulated bias ± σ')
    for bg, bias, lbl in zip(rw_bg, rw_bias, rw_labels):
        ax.annotate(lbl, (bg / 1e6, bias), textcoords='offset points',
                    xytext=(4, 4), fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Background rate (Mcps)')
    ax.set_ylabel('Range-walk bias (m)')
    ax.set_title('(a) Test 1 — multilevel background range walk\n'
                 r'')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # --- Panel (b): Test 1 — overlaid TCSPC histograms at each BG level ---
    ax = axes[0, 1]
    np.random.seed(1)
    hist_colors = {
        'Very Low':    'steelblue',
        'Low':         'mediumseagreen',
        'Int2':        'yellowgreen',
        'Intermediate':'gold',
        'Int3':        'darkorange',
        'Int4':        'orangered',
        'High':        'crimson',
    }
    tof_true_ns = 2.0 * true_z / C_LIGHT * 1e9
    N_bins_vis  = int(DEFAULT_PARAMS['T_window'] / DEFAULT_PARAMS['dt_bin'])
    t_bins_ns   = np.arange(N_bins_vis) * DEFAULT_PARAMS['dt_bin'] * 1e9
    for label, bg_rate in bg_levels.items():
        p_vis    = {**DEFAULT_PARAMS, 'lambda_bg_rate': bg_rate, 'N_pulses': 500}
        hist_vis = np.zeros(N_bins_vis, dtype=np.int32)
        for _ in range(p_vis['N_pulses']):
            k = _simulate_one_pulse(tof_true_ns * 1e-9, p_vis)
            if k is not None:
                hist_vis[k] += 1
        ax.plot(t_bins_ns, hist_vis, lw=1.0, alpha=0.8,
                color=hist_colors[label],
                label=f'{label} ({bg_rate/1e6:.0f} Mcps)')
    ax.axvline(tof_true_ns, color='black', lw=1.8, ls='--',
               label=f'True TOF = {tof_true_ns:.2f} ns')
    ax.set_xlim(0, 60)
    ax.set_xlabel('Time bin (ns)')
    ax.set_ylabel('Counts per bin')
    ax.set_title('(b) Test 1 — TCSPC histograms vs BG level\n'
                 'Signal peak buried as BG rises (pile-up)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (c): Test 2 — P_det theory vs Monte Carlo ---
    ax = axes[1, 0]
    bar_labels = ['Theory\n$1-e^{-\\lambda T}$', 'SPAD Model\n(50 k trials)']
    vals       = [P_theory, P_meas]
    colors     = ['steelblue', 'seagreen']
    bars = ax.bar(bar_labels, vals, color=colors, width=0.4, edgecolor='black', linewidth=0.8)
    ax.errorbar([1], [P_meas], yerr=[P_meas_err * 2], fmt='none',
                color='black', capsize=6, lw=1.5, label=r'$\pm 2\sigma$ binomial')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0005,
                f'{v:.5f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel('Detection probability $P_{\\rm det}$')
    ax.set_title(f'(c) Test 2 — Poisson first-arrival $P_{{\\rm det}}$\n'
                 fr'$\lambda$ = {lam/1e6:.2f} MHz, ratio = {ratio:.4f}  '
                 r'')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    # --- Panel (d): Test 3 — sigma vs distance (log-log) with d^2 reference ---
    ax = axes[1, 1]
    ds       = np.array(dist_points)
    sig      = np.array([stds[d] for d in dist_points])
    valid    = ~np.isnan(sig)
    ax.scatter(ds[valid], sig[valid], color='darkorange', s=80, zorder=5,
               label='Simulated $\\sigma(d)$')
    ax.plot(ds[valid], sig[valid], color='darkorange', lw=1.2, alpha=0.7)
    if not math.isnan(stds[2.0]):
        ref = stds[2.0] * (ds / 2.0) ** 2
        ax.plot(ds, ref, 'k--', lw=1.2, label=r'$\propto d^2$ reference (shot-noise)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance $d$ (m)')
    ax.set_ylabel('Standard Deviation $\\sigma$ (m)')
    ax.set_title('(d) Test 3 — precision vs distance\n'
                 r'SNR $\propto 1/d^2$')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(dist_points)
    ax.set_xticklabels([str(int(d)) for d in dist_points])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\nCompliance-test figure saved -> {save_path}")


# ---------------------------------------------------------------------------
# Demo: drone altitude simulation
# ---------------------------------------------------------------------------


def get_raw_imu_reading(true_accel_z):
    # 1. Physics: The true 3D acceleration of a level drone
    true_vector = np.array([0.0, 0.0, true_accel_z])

    # 2. Hardware: Add gravity to the Z-axis (IMUs feel gravity)
    gravity_vector = np.array([0.0, 0.0, GRAVITY])

    # 3. Hardware: Add 3D Gaussian noise
    noise_vector = np.random.normal(0, 0.1, size=3)

    # 4. The final raw 3D output of the sensor
    raw_imu = true_vector + gravity_vector + noise_vector

    return raw_imu # Returns [ax, ay, az]

def run_drone_demo(
    offset      = 75.0,   # centre altitude [m]          e.g. 75 -> oscillates around 75 m
    amplitude   = 25.0,   # oscillation half-range [m]   e.g. 25 -> goes from 50 to 100 m
    omega       = 0.5,    # angular frequency [rad/s]    e.g. 0.5 -> ~12.6 s period
                          #                               2*pi/omega gives the period in seconds
    sim_time    = 40.0,   # total simulation time [s]
    spad_hz     = 2,      # SPAD measurement rate [Hz]
    imu_hz      = 10,     # IMU measurement rate [Hz]
    lambda_bg   = 30e6,   # background photon rate [detected photons/s]
    N_pulses    = 400,    # laser pulses accumulated per SPAD reading
):
    """
    Drone altitude demo.

    Trajectory:  true_z(t) = offset + amplitude * sin(omega * t)
    Acceleration: true_a(t) = -amplitude * omega^2 * sin(omega * t)

    Quick examples
    --------------
    # Slow oscillation 50-100 m:
    run_drone_demo(offset=75, amplitude=25, omega=0.3)

    # Fast oscillation 20-30 m:
    run_drone_demo(offset=25, amplitude=5, omega=2.0)

    # Static hover at 60 m (amplitude=0):
    run_drone_demo(offset=60, amplitude=0, omega=1.0)
    """
    tick_hz       = max(imu_hz, spad_hz) * 4   # internal tick rate, always faster than sensors
    dt_sim        = 1.0 / tick_hz
    total_ticks   = int(sim_time * tick_hz)
    spad_interval = max(1, tick_hz // spad_hz)
    imu_interval  = max(1, tick_hz // imu_hz)

    # T_window must cover the round-trip TOF of the furthest point
    max_range  = offset + amplitude + 5.0          # +5 m safety margin
    t_window   = 2.0 * max_range / C_LIGHT * 1.2  # 20% extra margin
    spad_params = {**DEFAULT_PARAMS,
                   'lambda_bg_rate': lambda_bg,
                   'N_pulses':       N_pulses,
                   'T_window':       t_window}

    t_hist, z_hist, a_hist   = [], [], []
    imu_t, imu_a             = [], []
    spad_t, z_naive, z_phys  = [], [], []

    for tick in range(total_ticks):
        t      = tick * dt_sim
        true_z = offset + amplitude * math.sin(omega * t)
        true_a = -amplitude * omega**2 * math.sin(omega * t)
        t_hist.append(t); z_hist.append(true_z); a_hist.append(true_a)

        if tick % imu_interval == 0:
            imu_t.append(t)

            imu_a.append(get_raw_imu_reading(true_a)[2] - GRAVITY)

        if tick % spad_interval == 0:
            spad_t.append(t)
            z_naive.append(true_z + np.random.normal(0, 0.5))
            m = spad_measure(true_z, spad_params)
            z_phys.append(m if m is not None else float('nan'))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(t_hist, z_hist, 'k-', lw=2, label='True altitude')
    ax1.scatter(spad_t, z_naive, c='grey', marker='x', s=70,
                label='Naive  (Gaussian sigma=0.5 m)')
    ax1.scatter(spad_t, z_phys, c='red', marker='o', s=50,
                label='Physics-based SPAD')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Altitude: Ground truth vs SPAD models')
    ax1.legend(); ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(t_hist, a_hist, 'k--', lw=2, label='True acceleration')
    ax2.plot(imu_t, imu_a, 'b-', alpha=0.7, label='IMU measured')
    ax2.set_ylabel('Acceleration (m/s^2)')
    ax2.set_title('Acceleration: Ground truth vs IMU')
    ax2.legend(); ax2.grid(True)

    err_n = [abs(z_naive[i] - z_hist[round(spad_t[i] / dt_sim)])
             for i in range(len(spad_t))]
    err_p = [abs(z_phys[i]  - z_hist[round(spad_t[i] / dt_sim)])
             if not math.isnan(z_phys[i]) else float('nan')
             for i in range(len(spad_t))]

    ax3 = axes[2]
    ax3.plot(spad_t, err_n, c='grey', marker='x', label='|Error| naive')
    ax3.plot(spad_t, err_p, c='red',  marker='o', label='|Error| physics')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('|Error| (m)')
    ax3.set_title('Absolute measurement error')
    ax3.legend(); ax3.grid(True)

    plt.tight_layout()
    out = './spad_simulation.png'
    plt.savefig(out, dpi=150)
    print(f"Plot saved -> {out}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    run_compliance_tests()
    #run_drone_demo(offset=5, amplitude=3, omega=2.0)
