import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from numba import njit

# ==========================================
# 1. CARICAMENTO DATI BATIMETRICI (CSV)
# ==========================================
interp_z = None
total_distance = 0.0

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
        raise FileNotFoundError(f"ERRORE: File '{filename}' non trovato. Assicurati di aver eseguito lo script MATLAB.")

# ==========================================
# 2. COSTANTI E CONFIGURAZIONE
# ==========================================
tick_hz = 700
dt = 1.0 / tick_hz
SPAD_FREQ_HZ = 0.7  # Hz
accel_interval = int(tick_hz / 100)

# ==========================================
# 3. CORE DEL FILTRO DI KALMAN (COMPILATO CON NUMBA)
# ==========================================
@njit
def run_ekf_numba(total_ticks, true_z_array, true_a_array, dt, spad_interval, accel_interval, accel_bias, v):
    """
    Ciclo EKF compilato in C tramite Numba per massime prestazioni.
    Sostituisce il lentissimo loop Python puro.
    """
    # Inizializzazione Stato e Covarianza
    X = np.array([[true_z_array[0]], [0.0], [0.0]])
    P = np.eye(3) * 100.0
    
    F = np.array([[1.0, dt, 0.5 * dt**2], 
                  [0.0, 1.0, dt], 
                  [0.0, 0.0, 1.0]])

    # La varianza di processo (Q) scala con la velocità per assorbire meglio 
    # le variazioni brusche del fondale ad alte velocità.
    var_accel_process = 0.2 * max(1.0, v)
    Q = np.array([[0.1, 0.0, 0.0],
                  [0.0, var_accel_process * dt**2, var_accel_process * dt],
                  [0.0, var_accel_process * dt, var_accel_process]])

    R_spad = 0.05 ** 2   
    R_accel = 0.015 ** 2 

    H_spad = np.array([[1.0, 0.0, 0.0]])
    H_accel = np.array([[0.0, 0.0, 1.0]])
    H_both = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    R_spad_mat = np.array([[R_spad]])
    R_accel_mat = np.array([[R_accel]])
    R_both_mat = np.array([[R_spad, 0.0], [0.0, R_accel]])
    
    I = np.eye(3)
    res_z_history = np.zeros(total_ticks)

    for tick in range(total_ticks):
        true_z = true_z_array[tick]
        true_a = true_a_array[tick]

        # --- PREDICT ---
        X = F @ X
        P = F @ P @ F.T + Q

        # --- UPDATE ---
        has_spad = (tick % spad_interval == 0)
        has_accel = (tick % accel_interval == 0)

        if has_spad or has_accel:
            if has_spad and has_accel:
                # Misurazioni simulate (rumore aggiunto in-place per performance)
                z_spad = true_z + np.random.normal(0.0, 0.05)
                z_acc = true_a + np.random.normal(0.0, 0.015) + accel_bias
                Z = np.array([[z_spad], [z_acc]])
                
                y = Z - (H_both @ X)
                S = H_both @ P @ H_both.T + R_both_mat
                K = P @ H_both.T @ np.linalg.inv(S) 
                X = X + (K @ y)
                P = (I - K @ H_both) @ P

            elif has_spad:
                z_spad = true_z + np.random.normal(0.0, 0.05)
                Z = np.array([[z_spad]])
                
                y = Z - (H_spad @ X)
                S = H_spad @ P @ H_spad.T + R_spad_mat
                K = (P @ H_spad.T) / S[0, 0] 
                X = X + (K @ y)
                P = (I - K @ H_spad) @ P

            elif has_accel:
                z_acc = true_a + np.random.normal(0.0, 0.015) + accel_bias
                Z = np.array([[z_acc]])
                
                y = Z - (H_accel @ X)
                S = H_accel @ P @ H_accel.T + R_accel_mat
                K = (P @ H_accel.T) / S[0, 0] 
                X = X + (K @ y)
                P = (I - K @ H_accel) @ P

        res_z_history[tick] = X[0, 0]

    return res_z_history


# ==========================================
# 4. CICLO PRINCIPALE DELLE SIMULAZIONI
# ==========================================
def main():
    load_bathymetry_data('profilo_geometrico.csv')

    v_vec = np.linspace(1.0, 20.0, 50) # Espanso a 50 velocità per testare la performance
    distance = 10170.97
    rmse = np.zeros(len(v_vec))

    print("Inizio batch di simulazioni. Numba compilerà il codice al primo passaggio...")

    for i in range(len(v_vec)):
        v = v_vec[i]
        sim_time = distance / v
        total_ticks = int(sim_time * tick_hz)
        
        # Corretto n_spad a intero
        n_spad = int(sim_time * SPAD_FREQ_HZ)
        spad_interval = max(1, int(total_ticks / n_spad)) if n_spad > 0 else total_ticks
        
        # Il bias accelerometrico ora cambia ad ogni simulazione per maggiore realismo
        accel_bias = np.random.normal(0, 0.03)

        t_array = np.arange(total_ticks) * dt
        s_array = v * t_array
        true_z_array = interp_z(s_array)

        # Derivate numeriche per velocità e accelerazione
        true_vz_array = np.gradient(true_z_array, dt)
        true_a_array_raw = np.gradient(true_vz_array, dt)

        # Filtraggio passa-basso scalato dinamicamente con la velocità
        # Per v elevate, tau diminuisce in modo da rispondere più rapidamente ai cambi di gradiente
        tau = 0.6 / max(1.0, v * 0.1) 
        alpha = dt / (tau + dt)
        
        true_a_filtered = np.zeros(total_ticks)
        true_a_filtered[0] = true_a_array_raw[0]
        for k in range(1, total_ticks):
            true_a_filtered[k] = alpha * true_a_array_raw[k] + (1 - alpha) * true_a_filtered[k-1]

        true_a_array = true_a_filtered

        # Esecuzione dell'EKF ottimizzato
        res_z_history = run_ekf_numba(
            total_ticks, 
            true_z_array, 
            true_a_array, 
            dt, 
            spad_interval, 
            accel_interval, 
            accel_bias,
            v
        )

        # Calcolo RMSE
        rmse_current_sim = np.sqrt(np.mean((true_z_array - res_z_history)**2))
        rmse[i] = rmse_current_sim
        
        print(f"[{i+1}/{len(v_vec)}] v={v:5.1f} m/s | ticks={total_ticks:7d} | misure SPAD={n_spad:5d} | RMSE={rmse_current_sim:.3f} m")

    # ==========================================
    # 5. GENERAZIONE GRAFICI
    # ==========================================
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(v_vec, rmse, 'b-o', linewidth=2)
    ax.set_xlabel('Velocità drone (m/s)')
    ax.set_ylabel('RMSE errore profondità (m)')
    ax.set_title('Errore EKF vs. Velocità del Drone')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()