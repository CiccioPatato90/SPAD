import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# ==========================================
# 1. CARICAMENTO DATI BATIMETRICI (CSV)
# ==========================================
interp_z = None
total_distance = 0.0
sim_time_from_csv = 20.0  # Valore di default, sovrascritto dal CSV

def load_bathymetry_data(filename='profilo_geometrico.csv'):
    global interp_z, total_distance
    try:
        df = pd.read_csv(filename)
        s = df['Distance'].values        # asse spaziale [m]
        z = df['Z_AUV'].values           # quota drone [m]
        total_distance = s[-1]
        interp_z = interp1d(s, z, kind='linear',
                            fill_value=(z[0], z[-1]),
                            bounds_error=False)
        print(f"Profilo geometrico caricato. Distanza totale: {total_distance:.2f} m")
    except FileNotFoundError:
        print(f"ERRORE: File '{filename}' non trovato. Assicurati di aver eseguito lo script MATLAB.")
        exit(1)

# ==========================================
# 2. CONFIGURAZIONE SPAD E SENSORI
# ==========================================
print("Warning: Uso il rumore SPAD simulato base (Test di velocità attivo).")
def spad_measure(true_z, params):
    return true_z + np.random.normal(0, 0.05)

tick_hz = 700
dt      = 1.0 / tick_hz

GRAVITY = 9.81  # m/s^2
C_LIGHT = 2.25e8  # speed of light in water [m/s]
SPAD_FREQ_HZ = 0.7  # Hz
ACCEL_FIXED_BIAS = 0.0002 * 9.81  # bias fisso per mantenere costante l'errore tra le simulazioni

SPAD_DEFAULT_PARAMS = dict(
    T_HO           = 10e-9,
    PDP            = 0.30,
    DCR            = 1e3,
    N_pulses       = 600,
    T_window       = 400e-9,
    dt_bin         = 100e-12,
    lambda_sig_0   = 10000,
    pulse_sigma_s  = 0.42e-9,
    lambda_bg_rate = 20e6,
)

accel_interval = int(tick_hz / 100)
def get_acc_reading(true_a):
    white_noise = np.random.normal(0, 9.8 * 0.00016 * np.sqrt(100))
    return true_a + white_noise + ACCEL_FIXED_BIAS

def moving_average(v, k):
    return np.convolve(v, np.ones(k) / k, mode='valid')

# ==========================================
# 3. CICLO PRINCIPALE E FILTRO DI KALMAN (OTTIMIZZATO)
# ==========================================
def main():
    # --- Carica i dati reali PRIMA di iniziare ---
    load_bathymetry_data('profilo_geometrico.csv')

    # Inizializzazione vettori per confronto simulazioni (Partenza alzata a 3.0 m/s per velocità di calcolo)
    v_vec = np.linspace(1.0, 500.0, 50)
    distance = 10170.97
    rmse = np.zeros(len(v_vec))

    num_runs = 20  # Numero di simulazioni ripetute per calcolare la confidenza statistica
    
    # Array per salvare i limiti dell'intervallo di confidenza al 95% calcolati con il Bootstrap
    rmse_ci_lower = np.zeros(len(v_vec))
    rmse_ci_upper = np.zeros(len(v_vec))

    # === SCATOLA 1: CICLO DELLE VELOCITÀ (4 spazi) ===
    for i in range(len(v_vec)):
        v = v_vec[i]
        sim_time = distance / v
        total_ticks = int(sim_time * tick_hz)
        
        SPAD_FREQ_HZ = 0.7
        n_spad = sim_time * SPAD_FREQ_HZ          
        spad_interval = max(1, int(total_ticks / n_spad))
        print(f"\n[Velocità {i+1}/{len(v_vec)}] v={v:.1f} m/s | Esecuzione di {num_runs} run Monte Carlo...")

        t_array = np.arange(total_ticks) * dt
        s_array = v * t_array              

        quota = interp_z(s_array)
        true_vz_array = np.gradient(quota, dt)     
        true_a_array  = np.gradient(true_vz_array, dt)

        tau   = 0.6
        alpha = dt / (tau + dt)
        true_a_filtered = np.zeros(total_ticks)
        true_a_filtered[0] = true_a_array[0]
        for k_idx in range(1, total_ticks):
            true_a_filtered[k_idx] = alpha * true_a_array[k_idx] + (1 - alpha) * true_a_filtered[k_idx-1]

        true_a_array = true_a_filtered
        true_z_array = 3.0 + (true_a_array * (tau**2))

        # Array temporaneo per salvare l'RMSE di ogni singola run a questa specifica velocità
        rmse_runs = np.zeros(num_runs)

        # === SCATOLA 2: CICLO MONTE CARLO NIDIFICATO (8 spazi) ===
        for run in range(num_runs):
            initial_z = 3.0  
            max_range = initial_z + 10.0
            spad_params = {**SPAD_DEFAULT_PARAMS, 'T_window': 2.0 * max_range / C_LIGHT}
            res_z_history = np.zeros(total_ticks)

            X = np.array([[initial_z], [0.0], [0.0]])  
            P = np.eye(3) * 100.0
            F = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])

            var_accel_process = 0.2
            Q = np.array([[0.1, 0, 0],
                        [0, var_accel_process * dt**2, var_accel_process * dt],
                        [0, var_accel_process * dt, var_accel_process]])

            R_spad = 0.05 ** 2   
            R_accel = 0.015 ** 2 

            H_spad = np.array([[1.0, 0.0, 0.0]])
            H_accel = np.array([[0.0, 0.0, 1.0]])
            H_both = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

            R_spad_mat = np.array([[R_spad]])
            R_accel_mat = np.array([[R_accel]])
            R_both_mat = np.array([[R_spad, 0.0], [0.0, R_accel]])
            I = np.eye(3)
            
            # === SCATOLA 3: CICLO INTERNO DEI TICK (12 spazi) ===
            for tick in range(total_ticks):
                X = F @ X
                P = F @ P @ F.T + Q

                has_spad = (tick % spad_interval == 0)
                has_accel = (tick % accel_interval == 0)

                if has_spad or has_accel:
                    if has_spad and has_accel:
                        z_spad = spad_measure(true_z_array[tick], spad_params)
                        z_acc = get_acc_reading(true_a_array[tick])
                        Z = np.array([[z_spad], [z_acc]])
                        y = Z - (H_both @ X)
                        S = H_both @ P @ H_both.T + R_both_mat
                        K = P @ H_both.T @ np.linalg.inv(S) 
                        X = X + (K @ y)
                        P = (I - K @ H_both) @ P
                    elif has_spad:
                        z_spad = spad_measure(true_z_array[tick], spad_params)
                        Z = np.array([[z_spad]])
                        y = Z - (H_spad @ X)
                        S = H_spad @ P @ H_spad.T + R_spad_mat
                        K = (P @ H_spad.T) / S[0, 0] 
                        X = X + (K @ y)
                        P = (I - K @ H_spad) @ P
                    elif has_accel:
                        z_acc = get_acc_reading(true_a_array[tick])
                        Z = np.array([[z_acc]])
                        y = Z - (H_accel @ X)
                        S = H_accel @ P @ H_accel.T + R_accel_mat
                        K = (P @ H_accel.T) / S[0, 0] 
                        X = X + (K @ y)
                        P = (I - K @ H_accel) @ P

                res_z_history[tick] = X[0, 0]

            # Fine della singola run: applicazione Moving Average originale
            k = 900
            MA_z_history = moving_average(res_z_history, k)

            # Allineamento e calcolo RMSE per la run corrente
            true_z_tagliato = true_z_array[(k-1):]
            rmse_runs[run] = np.sqrt(np.mean((true_z_tagliato - MA_z_history)**2)) / np.sqrt(np.mean((true_z_tagliato)**2))

        # === CALCOLO STATISTICO CON BOOTSTRAP DI SCIPY (Allineato a 8 spazi) ===
        from scipy.stats import bootstrap
        
        # Salviamo la media complessiva degli RMSE di questa velocità nel vettore globale
        rmse[i] = np.mean(rmse_runs)
        
        # Esecuzione dell'algoritmo di ricampionamento Bootstrap al 95%
        data_tuple = (rmse_runs,)
        res = bootstrap(data_tuple, np.mean, confidence_level=0.95, method='percentile', n_resamples=1000)
        rmse_ci_lower[i] = res.confidence_interval.low
        rmse_ci_upper[i] = res.confidence_interval.high

    # ==========================================
    # 4. GENERAZIONE GRAFICO SWEEP CON BOOTSTRAP CI (Estratto completamente dai cicli - 4 spazi)
    # ==========================================
    print("\nSimulazione e analisi Bootstrap completate. Generazione grafico...")
    
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Disegna la curva blu degli RMSE medi lungo lo sweep delle velocità
    ax.plot(v_vec, rmse, 'b-o', linewidth=2, label=f'Mean RMSRE (MA k={k})')
    
    # 2. Genera la fascia grigia dell'intervallo di confidenza al 95% calcolata con SciPy Bootstrap
    ax.fill_between(v_vec, rmse_ci_lower, rmse_ci_upper, color='gray', alpha=0.3, label='95% Confidence Interval')
    
    ax.set_xlabel('Drone speed (m/s)')
    ax.set_ylabel('Depth relative error RSMRE (m)')
    ax.set_title('Relative error vs. Drone speed (95% Confidence Interval)')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show(block=True)
    plt.close('all')

# ==========================================
# 5. ESECUZIONE SCRIPT
# ==========================================
if __name__ == "__main__":
    main()
