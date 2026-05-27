import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# ==========================================
# 1. CARICAMENTO DATI BATIMETRICI (CSV)
# ==========================================
interp_z = None
interp_a = None
sim_time_from_csv = 20.0 # Valore di default, sovrascritto dal CSV

def load_bathymetry_data(filename='bathymetry_profile.csv'):
    global interp_z, interp_a, sim_time_from_csv
    try:
        df = pd.read_csv(filename)
        t = df['Time'].values
        z = df['Depth'].values
        a = df['Acceleration'].values
        
        sim_time_from_csv = t[-1]
        
        # Funzioni di interpolazione per interrogare la posizione a qualsiasi istante t
        interp_z = interp1d(t, z, kind='linear', fill_value=(z[0], z[-1]), bounds_error=False)
        interp_a = interp1d(t, a, kind='linear', fill_value=(a[0], a[-1]), bounds_error=False)
        print(f"Dati batimetrici caricati con successo. Durata traiettoria: {sim_time_from_csv:.2f} s")
    except FileNotFoundError:
        print(f"ERRORE: File '{filename}' non trovato. Assicurati di aver eseguito lo script MATLAB.")
        exit(1)

# ==========================================
# 2. CONFIGURAZIONE SPAD E SENSORI
# ==========================================
# DISABILITATO TEMPORANEAMENTE IL MODELLO ESTERNO PER TESTARE LA VELOCITA'
# try:
#     from spad_model import spad_measure
# except ImportError:

print("Warning: Uso il rumore SPAD simulato base (Test di velocità attivo).")
def spad_measure(true_z, params):
    return true_z + np.random.normal(0, 0.05)

tick_hz = 700
dt      = 1.0 / tick_hz

GRAVITY = 9.81 # m/s^2
C_LIGHT = 2.25e8 # speed of light in water [m/s]

spad_interval = int(tick_hz/0.7)
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

accel_interval = int(tick_hz/100)
def get_acc_reading(true_a):
    white_noise = np.random.normal(0, 0.015)
    ACCEL_FIXED_BIAS = np.random.normal(0, 0.2)
    return true_a + white_noise + ACCEL_FIXED_BIAS

# ==========================================
# 3. CICLO PRINCIPALE E FILTRO DI KALMAN (OTTIMIZZATO)
# ==========================================
def main():
    # --- Carica i dati reali PRIMA di iniziare ---
    load_bathymetry_data('bathymetry_profile.csv')
    
    # Aggiorna i tick totali basati sul tempo del CSV
    total_ticks = int(sim_time_from_csv * tick_hz)

    # Configurazione iniziale basata sul primo valore del CSV
    initial_z = float(interp_z(0))
    max_range = initial_z + 10.0
    spad_params = {**SPAD_DEFAULT_PARAMS, 'T_window': 2.0 * max_range / C_LIGHT}

    # --- PRE-CALCOLO VETTORIZZATO ---
    print("Pre-calcolo della traiettoria in corso...")
    t_array = np.arange(total_ticks) * dt
    true_z_array = interp_z(t_array)
    true_a_array = interp_a(t_array)
    
    true_z_history = true_z_array.tolist()
    true_a_history = true_a_array.tolist()
    t_history = t_array.tolist()

    spad_history, spad_t_history = [], []
    accel_history, accel_t_history = [], []

    # --- PRE-ALLOCAZIONE ARRAY ---
    res_z_history = np.zeros(total_ticks)
    res_v_history = np.zeros(total_ticks)
    res_a_history = np.zeros(total_ticks)

    # --- Inizializzazione Stato Filtro di Kalman ---
    X = np.array([[initial_z], [0.0], [0.0]])
    P = np.eye(3) * 100.0
    F = np.array([[1, dt, 0.5 * dt**2], 
                  [0, 1, dt], 
                  [0, 0, 1]])

    var_accel_process = 0.2
    Q = np.array([[0.1, 0, 0],
                  [0, var_accel_process * dt**2, var_accel_process * dt],
                  [0, var_accel_process * dt, var_accel_process]])

    R_spad = 0.05 ** 2   
    R_accel = 0.015 ** 2 

    # --- Matrici pre-calcolate per evitare di crearle nel ciclo ---
    H_spad = np.array([[1.0, 0.0, 0.0]])
    H_accel = np.array([[0.0, 0.0, 1.0]])
    H_both = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    R_spad_mat = np.array([[R_spad]])
    R_accel_mat = np.array([[R_accel]])
    R_both_mat = np.array([[R_spad, 0.0], [0.0, R_accel]])
    
    I = np.eye(3)

    # --- Simulazione ---
    print(f"Inizio simulazione EKF ({total_ticks} ticks). Allacciati le cinture...")
    
    for tick in range(total_ticks):
        # --- INDICATORE DI PROGRESSO A SCHERMO ---
        if tick % 10000 == 0 and tick > 0:
            percentuale = (tick / total_ticks) * 100
            print(f"Avanzamento: {tick} / {total_ticks} ({percentuale:.1f}%)")

        t = t_array[tick]
        true_z = true_z_array[tick]
        true_a = true_a_array[tick]

        # STEP B: EKF PREDICT STEP
        X = F @ X
        P = F @ P @ F.T + Q

        # STEP C: EKF UPDATE STEP
        has_spad = (tick % spad_interval == 0)
        has_accel = (tick % accel_interval == 0)

        if has_spad or has_accel:
            # Caso 1: Entrambi i sensori aggiornano allo stesso millisecondo
            if has_spad and has_accel:
                z_spad = spad_measure(true_z, spad_params)
                z_acc = get_acc_reading(true_a)
                Z = np.array([[z_spad], [z_acc]])
                
                spad_t_history.append(t)
                spad_history.append(z_spad)
                accel_t_history.append(t)
                accel_history.append(z_acc)

                y = Z - (H_both @ X)
                S = H_both @ P @ H_both.T + R_both_mat
                K = P @ H_both.T @ np.linalg.inv(S) 
                X = X + (K @ y)
                P = (I - K @ H_both) @ P

            # Caso 2: Solo SPAD
            elif has_spad:
                z_spad = spad_measure(true_z, spad_params)
                Z = np.array([[z_spad]])
                
                spad_t_history.append(t)
                spad_history.append(z_spad)

                y = Z - (H_spad @ X)
                S = H_spad @ P @ H_spad.T + R_spad_mat
                K = (P @ H_spad.T) / S[0, 0] 
                X = X + (K @ y)
                P = (I - K @ H_spad) @ P

            # Caso 3: Solo Accelerometro
            elif has_accel:
                z_acc = get_acc_reading(true_a)
                Z = np.array([[z_acc]])
                
                accel_t_history.append(t)
                accel_history.append(z_acc)

                y = Z - (H_accel @ X)
                S = H_accel @ P @ H_accel.T + R_accel_mat
                K = (P @ H_accel.T) / S[0, 0] 
                X = X + (K @ y)
                P = (I - K @ H_accel) @ P

        # Salvataggio velocissimo negli array pre-allocati
        res_z_history[tick] = X[0, 0]
        res_v_history[tick] = X[1, 0]
        res_a_history[tick] = X[2, 0]

    # ==========================================
    # 4. GENERAZIONE GRAFICI
    # ==========================================
    print("Simulazione completata. Generazione grafici...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax1.plot(t_history, true_z_history, 'k-', linewidth=2, label='True Depth (Altitudine dal fondo)')
    ax1.scatter(spad_t_history, spad_history, color='red', marker='x', s=10, label='SPAD Measurements', alpha=0.5)
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Sensore: SPAD Altitude (Interpolato da Batimetria)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t_history, true_a_history, 'k-', linewidth=2, label='True Relative Acceleration')
    ax2.plot(accel_t_history, accel_history, 'g-', alpha=0.3, label='Accelerometer Readings')
    ax2.set_ylabel('Acceleration (m/s^2)')
    ax2.set_title('Sensore: Accelerometro (Rumoroso)')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(t_history, true_z_history, 'k-', linewidth=2, label='True Depth')
    ax3.plot(t_history, res_z_history, 'b-', linewidth=2, alpha=0.8, label='EKF Estimated Depth')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('Risultato: Fusione EKF (SPAD + Accel) su Profilo Reale')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. ESECUZIONE SCRIPT
# ==========================================
if __name__ == "__main__":
    main()