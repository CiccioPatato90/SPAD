import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ============================================================
#  TURBIDITY MODEL
#  Parametro principale: c [m^-1]  (coefficiente di attenuazione beam)
#
#  Valori tipici in acqua marina:
#    c = 0.05 - 0.15  m^-1   acqua oceanica pulita (open ocean)
#    c = 0.15 - 0.40  m^-1   acqua costiera moderata
#    c = 0.40 - 1.00  m^-1   acqua torbida / estuarino
#    c = 1.00 - 4.00  m^-1   acqua molto torbida / porto
#
#  Il coefficiente c si scompone in:
#    c = a + b
#    a = coefficiente di assorbimento [m^-1]
#    b = coefficiente di scattering [m^-1]
#
#  A 532 nm (verde, tipico per LiDAR subacqueo):
#    Acqua pura:          a ≈ 0.05, b ≈ 0.003  →  c ≈ 0.053
#    Acqua costiera:      a ≈ 0.10, b ≈ 0.20   →  c ≈ 0.30
#    Acqua torbida:       a ≈ 0.20, b ≈ 0.80   →  c ≈ 1.00
# ============================================================

def compute_turbidity_spad_params(base_params: dict, turbidity_c: float, true_z: float) -> dict:
    """
    Aggiorna i parametri SPAD in funzione della torbidità e della distanza.

    Modifiche rispetto al modello base:
    1. lambda_sig_0:  attenuazione esponenziale del segnale  (Beer-Lambert)
       Il segnale percorre 2*z (andata e ritorno), quindi:
           lambda_sig_eff = lambda_sig_0 * exp(-2 * c * z) / z^2
       Il termine 1/z^2 è già nel modello originale (eq. 4 del paper),
       qui aggiungiamo il termine Beer-Lambert exp(-2*c*z).

    2. lambda_bg_rate: il backscatter volumetrico cresce con la torbidità.
       Il background non è solo luce ambientale esterna, ma anche
       fotoni retrodiffusi dalla colonna d'acqua (backscatter).
       Modello empirico:  lambda_bg_eff = lambda_bg_0 * (1 + k_back * c * z)
       dove k_back ≈ 1e6 è un fattore di scala che converte [m^-2]
       in rate di fotoni [Hz] per i parametri fisici del detector.

    3. pulse_sigma_s:  allargamento temporale del pulse per scattering multiplo.
       In acqua torbida ogni fotone subisce deviazioni di percorso
       che ritardano casualmente l'arrivo → il picco TCSPC si allarga.
       Modello:  sigma_eff = sigma_0 * sqrt(1 + alpha_scatter * c * z)
       dove alpha_scatter ≈ 0.5 (adimensionale, calibrato empiricamente).

    Parametri
    ----------
    base_params   : dizionario parametri SPAD di default
    turbidity_c   : coefficiente di attenuazione beam c [m^-1]
    true_z        : distanza vera corrente [m]

    Returns
    -------
    params aggiornati (copia, non modifica l'originale)
    """
    params = base_params.copy()

    z = max(true_z, 0.01)  # evita divisioni per zero

    # ---- 1. Attenuazione segnale: Beer-Lambert ----
    # exp(-2*c*z): fattore moltiplicativo su lambda_sig_0
    # A c=0.3, z=10m:  exp(-6) ≈ 0.0025  → segnale ridotto al 0.25%
    # A c=0.05, z=10m: exp(-1) ≈ 0.37    → segnale ridotto al 37%
    beer_lambert = math.exp(-2.0 * turbidity_c * z)
    params['lambda_sig_0'] = base_params['lambda_sig_0'] * beer_lambert

    # ---- 2. Backscatter volumetrico ----
    # k_back: numero medio di fotoni di background aggiunti per unità di [c*z]
    # Ordine di grandezza: con c=0.3, z=10 → c*z=3
    # lambda_bg_rate passa da 20e6 a 20e6*(1 + 0.3*3) = 20e6*1.9 = ~38 MHz
    k_back = 0.3  # [adimensionale], dipende da apertura ottica e filtri
    params['lambda_bg_rate'] = base_params['lambda_bg_rate'] * (1.0 + k_back * turbidity_c * z)

    # ---- 3. Pulse broadening per multiple scattering ----
    # alpha_scatter: quanto si allarga sigma per unità di [c*z]
    # Con c=0.3, z=10 → c*z=3:  sigma *= sqrt(1+0.5*3) = sqrt(2.5) ≈ 1.58
    # Il picco TCSPC diventa più largo → peggiore risoluzione in distanza
    alpha_scatter = 0.5
    broadening = math.sqrt(1.0 + alpha_scatter * turbidity_c * z)
    params['pulse_sigma_s'] = base_params['pulse_sigma_s'] * broadening

    return params


def compute_adaptive_R_spad(turbidity_c: float, estimated_z: float,
                             R_spad_min: float = 0.05**2,
                             R_spad_max: float = 2.0**2) -> float:
    """
    Varianza adattiva per il Kalman filter: R_spad cresce con torbidità e distanza.

    Logica: quando il segnale SPAD è degradato (alta torbidità, grande distanza),
    il filtro di Kalman deve fidarsi meno della misura SPAD e più dell'accelerometro.

    Modello:  R_spad = R_min * exp(gamma * c * z)
    dove gamma ≈ 0.4 è calibrato per avere:
      - c=0.05, z=5m:   R ≈ R_min * 1.10  (quasi invariato, segnale buono)
      - c=0.30, z=10m:  R ≈ R_min * 3.32  (incertezza triplicata)
      - c=1.00, z=20m:  R ≈ R_min * 2981  (segnale perso, Kalman ignora SPAD)

    Ordini di grandezza di R_spad:
      Bassa torbidità, bassa distanza:  R ≈ (0.05 m)^2 = 0.0025 m^2  → std ≈ 5 cm
      Media torbidità, media distanza:  R ≈ (0.15 m)^2 = 0.023  m^2  → std ≈ 15 cm
      Alta torbidità, alta distanza:    R ≈ (2.00 m)^2 = 4.0    m^2  → std ≈ 2 m (ignorato)
    """
    gamma = 0.4
    z = max(estimated_z, 0.01)
    R = R_spad_min * math.exp(gamma * turbidity_c * z)
    return float(np.clip(R, R_spad_min, R_spad_max))


# ============================================================
#  SPAD MODEL  (invariato rispetto alla versione base)
# ============================================================
try:
    from spad_model import spad_measure
except ImportError:
    def spad_measure(true_z, params):
        """
        Modello SPAD semplificato: simula il processo TCSPC.
        Aggiunge:
          - bias verso distanze minori (pile-up)
          - rumore gaussiano con std proporzionale a 1/sqrt(SNR)
          - missed detections a basso SNR
        """
        lambda_sig = params['lambda_sig_0'] / max(true_z**2, 0.01)
        lambda_bg  = params['lambda_bg_rate'] * params['T_window']
        snr = lambda_sig / (lambda_bg + 1e-9)

        # Probabilità di missed detection (segnale troppo debole)
        p_miss = math.exp(-lambda_sig * params['N_pulses'])
        if np.random.rand() < p_miss:
            # Misura dominata da background → valore casuale nella finestra
            return np.random.uniform(0, true_z * 1.5)

        # Bias pile-up: spostamento verso distanze minori
        pile_up_bias = -0.05 / max(snr, 0.1)

        # Rumore shot: std ∝ 1/sqrt(N_detected_photons)
        n_detected = lambda_sig * params['N_pulses']
        shot_noise_std = params['pulse_sigma_s'] * params['C_LIGHT'] / (2.0 * math.sqrt(max(n_detected, 1)))

        noise = np.random.normal(pile_up_bias, shot_noise_std)
        return true_z + noise


# ============================================================
#  PARAMETRI DI SIMULAZIONE
# ============================================================
tick_hz    = 700
dt         = 1.0 / tick_hz
sim_time   = 20.0
total_ticks = int(sim_time * tick_hz)

GRAVITY  = 9.81
C_LIGHT  = 2.25e8  # velocità della luce in acqua [m/s]

spad_interval  = int(tick_hz / 7)
baro_interval  = int(tick_hz / 27)
accel_interval = int(tick_hz / 100)

# Parametri SPAD base (a torbidità nulla)
SPAD_DEFAULT_PARAMS = dict(
    T_HO           = 10e-9,
    PDP            = 0.30,
    DCR            = 1e3,
    N_pulses       = 600,
    T_window       = 400e-9,
    dt_bin         = 100e-12,
    lambda_sig_0   = 10000,      # [fotoni/impulso] a 1 m, senza attenuazione
    pulse_sigma_s  = 0.42e-9,    # [s] larghezza temporale del pulse laser
    lambda_bg_rate = 20e6,       # [Hz] rate di fotoni di background
    C_LIGHT        = C_LIGHT,
)

# ============================================================
#  SCENARI DI TORBIDITÀ DA CONFRONTARE
# ============================================================
TURBIDITY_SCENARIOS = {
    'Pulita  (c=0.05)':   0.05,
    'Costiera (c=0.30)':  0.30,
    'Torbida  (c=1.00)':  1.00,
}


# ============================================================
#  TRAIETTORIA VERA
# ============================================================
def get_true_reading(t, offset, amplitude, omega, gradient, phase=0.0):
    z = (offset + gradient * t) + amplitude * math.sin(omega * t + phase)
    a = -amplitude * (omega**2) * math.sin(omega * t + phase)
    if z <= 0.0:
        z = 1e-6
        a = 0.0
    return z, a


def get_acc_reading(true_a):
    return true_a + np.random.normal(0, 0.015) + np.random.normal(0, 0.2)


# ============================================================
#  SIMULAZIONE PRINCIPALE
# ============================================================
def run_simulation(turbidity_c: float):
    """Esegue la simulazione per un dato valore di torbidità."""

    offset, amplitude, omega = 20.0, 2.0, 1.0
    gradient = -1.5

    max_range   = offset + amplitude + 5.0
    base_params = {**SPAD_DEFAULT_PARAMS,
                   'T_window': 2.0 * max_range / C_LIGHT}

    # Stato iniziale Kalman  X = [z, v, a]^T
    X = np.array([[offset], [0.0], [0.0]])
    P = np.eye(3) * 100.0

    F = np.array([[1, dt, 0.5*dt**2],
                  [0,  1, dt],
                  [0,  0,  1]])

    var_ap = 0.2
    Q = np.array([[0.1, 0, 0],
                  [0, var_ap*dt**2, var_ap*dt],
                  [0, var_ap*dt,    var_ap]])

    R_accel = 0.015**2

    # Storage
    t_hist, true_z_hist, est_z_hist = [], [], []
    spad_t, spad_vals = [], []
    R_spad_hist = []

    for tick in range(total_ticks):
        t = tick * dt
        true_z, true_a = get_true_reading(t, offset, amplitude, omega, gradient)

        t_hist.append(t)
        true_z_hist.append(true_z)

        # Predict
        X = F @ X
        P = F @ P @ F.T + Q

        has_spad  = (tick % spad_interval  == 0)
        has_accel = (tick % accel_interval == 0)

        if has_spad or has_accel:
            H_rows, Z_rows, R_diag = [], [], []

            if has_spad:
                # Aggiorna parametri SPAD con torbidità corrente
                params_t = compute_turbidity_spad_params(base_params, turbidity_c, true_z)
                spad_val = spad_measure(true_z, params_t)

                # R adattivo: il Kalman si fida meno quando il segnale è degradato
                R_spad = compute_adaptive_R_spad(turbidity_c, X[0, 0])

                H_rows.append([1.0, 0.0, 0.0])
                Z_rows.append([spad_val])
                R_diag.append(R_spad)

                spad_t.append(t)
                spad_vals.append(spad_val)
                R_spad_hist.append(R_spad)

            if has_accel:
                accel_val = get_acc_reading(true_a)
                H_rows.append([0.0, 0.0, 1.0])
                Z_rows.append([accel_val])
                R_diag.append(R_accel)

            H = np.array(H_rows)
            Z = np.array(Z_rows)
            R = np.diag(R_diag)

            y = Z - (H @ X)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            X = X + (K @ y)
            P = (np.eye(3) - K @ H) @ P

        est_z_hist.append(X[0, 0])

    return (t_hist, true_z_hist, est_z_hist,
            spad_t, spad_vals, R_spad_hist)


# ============================================================
#  PLOT
# ============================================================
def main():
    n_scenarios = len(TURBIDITY_SCENARIOS)
    fig, axes = plt.subplots(n_scenarios, 2, figsize=(14, 4 * n_scenarios))
    fig.suptitle('Effetto della Torbidità sul Sistema SPAD + Kalman Filter', fontsize=14, fontweight='bold')

    colors = ['steelblue', 'darkorange', 'crimson']

    for idx, (label, c_val) in enumerate(TURBIDITY_SCENARIOS.items()):
        print(f"Simulazione: {label} ...")
        t_hist, true_z, est_z, spad_t, spad_vals, R_spad_hist = run_simulation(c_val)

        ax_left  = axes[idx, 0]
        ax_right = axes[idx, 1]

        # ---- Plot sinistro: traiettoria ----
        ax_left.plot(t_hist, true_z, 'k-', lw=2, label='Profondità vera')
        ax_left.scatter(spad_t, spad_vals, color=colors[idx], marker='x',
                        s=30, alpha=0.6, label='Misure SPAD')
        ax_left.plot(t_hist, est_z, color=colors[idx], lw=2, alpha=0.9,
                     label='EKF stimato')
        ax_left.set_ylabel('Profondità [m]')
        ax_left.set_title(f'{label} — Stima profondità')
        ax_left.legend(fontsize=8)
        ax_left.grid(True, alpha=0.4)

        # Errore RMSE
        rmse = np.sqrt(np.mean((np.array(true_z) - np.array(est_z))**2))
        ax_left.text(0.02, 0.05, f'RMSE = {rmse:.3f} m',
                     transform=ax_left.transAxes, fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ---- Plot destro: R_spad adattivo nel tempo ----
        ax_right.plot(spad_t, [math.sqrt(r) for r in R_spad_hist],
                      color=colors[idx], lw=1.5)
        ax_right.set_ylabel('std(R_spad) [m]  — incertezza SPAD')
        ax_right.set_title(f'{label} — Incertezza adattiva Kalman')
        ax_right.set_xlabel('Tempo [s]')
        ax_right.grid(True, alpha=0.4)
        ax_right.set_yscale('log')

        axes[idx, 0].set_xlabel('Tempo [s]')

    plt.tight_layout()
    
    # --- MODIFICA QUI ---
    # Salva nella directory corrente invece che in un percorso assoluto inesistente
    output_path = 'spad_turbidity_comparison.png' 
    plt.savefig(output_path, dpi=150)
    print(f"Plot salvato in {output_path}")
    plt.show()

if __name__ == "__main__":
    main()