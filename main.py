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
sim_time_from_csv = 20.0 # Valore di default, sovrascritto dal CSV

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
SPAD_FREQ_HZ = 0.7 #Hz
#ACCEL_FIXED_BIAS = np.random.normal(0, 0.03) # m/s^2
ACCEL_FIXED_BIAS=0.0002*9.81 # m/s^2, bias fisso per mantenere costante l'errore tra le simulazioni a diverse velocità


#spad_interval = int(tick_hz/0.7)
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
   #white_noise = np.random.normal(0, 0.015)
    white_noise=np.random.normal(0, 9.8*0.00016*np.sqrt(100))
    #ACCEL_FIXED_BIAS = np.random.normal(0, 0.2) spsostato fuori dal ciclo per mantenere costante il bias tra le simulazioni a diverse velocità
    return true_a + white_noise + ACCEL_FIXED_BIAS


def bandpass_filter(signal, t, f_low, f_high):
   sr = 1 / (t[1] - t[0])
   spectrum = np.fft.rfft(signal)
   freqs = np.fft.rfftfreq(len(signal), 1/sr)
   spectrum[(freqs < f_low) | (freqs > f_high)] = 0
   return np.fft.irfft(spectrum, len(signal))


# ==========================================
# 3. CICLO PRINCIPALE E FILTRO DI KALMAN (OTTIMIZZATO)
# ==========================================
def main():
    # --- Carica i dati reali PRIMA di iniziare ---
    load_bathymetry_data('profilo_geometrico.csv')

    #Inizializzazione vettori per confronto simulazioni
    v_vec=np.linspace(1.0, 500.0, 50)
    distance=10170.97
    rmse = np.zeros(len(v_vec))

    for i in range(len(v_vec)):
        #np.random.seed(42) # Per garantire la compatibilità dell'errore tra le simulazioni a diverse velocità

        # Defining simulation time based on distance and velocity
        v=v_vec[i]
        sim_time = distance / v

        
        # Aggiorna i tick totali basati sul tempo del CSV e il numero didati catturati da SPAD
        total_ticks = int(sim_time * tick_hz)
        SPAD_FREQ_HZ = 0.7
        n_spad        = sim_time * SPAD_FREQ_HZ          # misure totali durante la missione
        spad_interval = max(1, int(total_ticks / n_spad))
        print(f"v={v:.1f} m/s | tick={total_ticks} | misure SPAD={n_spad}")

        t_array    = np.arange(total_ticks) * dt

       # Posizione spaziale sul percorso
        s_array = v * t_array              

        # 1. Calcoliamo la cinematica ASSOLUTA sulla mappa per ricavare le accelerazioni reali dei dossi
        quota = interp_z(s_array)
        true_vz_array = np.gradient(quota, dt)     
        true_a_array  = np.gradient(true_vz_array, dt)

        # Filtraggio passa-basso (inerzia drone)
        tau   = 0.6
        alpha = dt / (tau + dt)
        true_a_filtered = np.zeros(total_ticks)
        true_a_filtered[0] = true_a_array[0]
        for k in range(1, total_ticks):
            true_a_filtered[k] = alpha * true_a_array[k] + (1 - alpha) * true_a_filtered[k-1]

        true_a_array = true_a_filtered

        # Forza true_z_array a essere la distanza relativa pancia-fondo (attorno al target di 3 metri)
        true_z_array = 3.0 + (true_a_array * (tau**2))

        # Configurazione iniziale basata sulla distanza relativa target (3 metri)
        initial_z = 3.0  
        max_range = initial_z + 10.0
        spad_params = {**SPAD_DEFAULT_PARAMS, 'T_window': 2.0 * max_range / C_LIGHT}
        
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
        X = np.array([[initial_z], [0.0], [0.0]])  # Ora lo stato iniziale del filtro parte da 3.0
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

        # --- APPLICAZIONE FILTRO PASSA-BANDA  ---
        f_low = 0.0   # Frequenza di taglio inferiore [Hz] (puoi cambiarla)
        f_high = 5.0 # Frequenza di taglio superiore [Hz] (puoi cambiarla)
        BP_z_history = bandpass_filter(res_z_history, t_array, f_low, f_high)

        # Calcolo RMSE 
        rmse_current_sim = np.sqrt(np.mean((true_z_array - BP_z_history)**2)) / np.sqrt(np.mean((true_z_array)**2))
        rmse[i] = rmse_current_sim
        

    # ==========================================
    # 4. GENERAZIONE GRAFICI UNICA SIMULAZIONE
    # ==========================================
    # print("Simulazione completata. Generazione grafici...")

   

    # # Ora il calcolo dell'errore ha array della stessa identica dimensione (2373224,)
    # z_error_history = true_z_array - BP_z_history
 
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 24), sharex=True) # Aumentata altezza a 22 per non far accavallare i testi
    
    # # Condividi in automatico la scala verticale tra il plot del risultato e quello dell'errore
    # #ax4.sharey(ax3)

    # #AX1: SPAD (Usiamo i segnali interi perché scatter non ha problemi di shape)
    # ax1.plot(t_history, true_z_array, 'k-', linewidth=2, label='True Depth (Altitudine dal fondo)')
    # ax1.scatter(spad_t_history, spad_history, color='red', marker='x', s=10, label='SPAD Measurements', alpha=0.5)
    # ax1.set_ylabel('Depth (m)')
    # ax1.set_title('Sensore: SPAD Altitude (Interpolato da Batimetria)')
    # ax1.legend()
    # ax1.grid(True)

    # # AX2: Accelerometro (Usiamo la cronologia allineata per l'accelerazione vera)
    # ax2.plot(t_history, true_a_history, 'k-', linewidth=2, label='True Relative Acceleration')
    # ax2.plot(accel_t_history, accel_history, 'g-', alpha=0.3, label='Accelerometer Readings')
    # ax2.set_ylabel('Acceleration (m/s^2)')
    # ax2.set_title('Sensore: Accelerometro (Rumoroso)')
    # ax2.legend()
    # ax2.grid(True)

    # # AX3: Risultato di Fusione (Allineato)
    # ax3.plot(t_history, true_z_array, 'k-', linewidth=2, label='True Depth')
    # ax3.plot(t_history, BP_z_history, 'b-', linewidth=2, alpha=0.8, label='EKF + MA Estimated Depth')
    # ax3.set_xlabel('Time (s)')
    # ax3.set_ylabel('Depth (m)')
    # ax3.set_title(f'Risultato: Fusione EKF + Filtro BP su Profilo Reale') # Corretto con la f davanti alle virgolette
    # ax3.legend()
    # ax3.grid(True)

    # # AX4: Errore Residuo (Allineato e con asse Y condiviso)
    # ax4.plot(t_history, np.zeros(len(t_history)), 'k-', linewidth=2)
    # ax4.plot(t_history, z_error_history, 'r-', linewidth=2, label='Error')
    # ax4.set_xlabel('Time (s)')
    # ax4.set_ylabel('Depth (m)')
    # ax4.set_title('Errore tra EKF+BP e Profilo Reale')
    # ax4.legend()
    # ax4.grid(True)

    # fig.subplots_adjust(hspace=0.5, left=0.10, right=0.95, top=0.95, bottom=0.05)

   # ==========================================
    # ANALISI IN FREQUENZA CHIESTA DAL COLLEGA
    # ==========================================
    # from scipy.signal import welch, spectrogram

    # # 1.  GRAFICO FFT 
    # freqs, psd = welch(res_z_history, fs=tick_hz, nperseg=1024)

    # fig_fft, ax_fft = plt.subplots(figsize=(10, 5))
    # ax_fft.plot(freqs, psd, color='blue', linewidth=2)
    # ax_fft.set_title("FFT of the Kalman Filter Output (Before MA)") 
    # ax_fft.set_xlabel('Frequency (Hz)')
    # ax_fft.set_ylabel('Spectrum Power')
    # ax_fft.grid(True, which='both')

    # # 2.  GRAFICO SPETTROGRAMMA
    # f_spec, t_spec, Sxx = spectrogram(res_z_history, fs=tick_hz, nperseg=256)

    # fig_spec, ax_spec = plt.subplots(figsize=(10, 5))
    # pcm = ax_spec.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
    # fig_spec.colorbar(pcm, ax=ax_spec, label='Intensity (dB)')
    # ax_spec.set_title("Spectrogram of the Kalman Filter Output (Before MA)") 
    # ax_spec.set_xlabel('Time (s)')
    # ax_spec.set_ylabel('Frequency (Hz)')
    # ax_spec.set_ylim([0, tick_hz / 2]) 

    # fig.tight_layout(pad=3.0)
    # plt.show(block=True)
    
    # # ==========================================
    # # 4. GENERAZIONE GRAFICI CONFRONTO SIMULAZIONI
    # # ==========================================
    
    fig2, ax = plt.subplots(figsize=(9, 5))
    ax.plot(v_vec, rmse, 'b-o', linewidth=2)
    ax.set_xlabel('Drone Speed (m/s)')
    ax.set_ylabel('Depth relative error RMSRE (m)')
    ax.set_title('Relative Error vs. Drone Speed')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
     
    plt.close('all')

# ==========================================
# 5. ESECUZIONE SCRIPT
# ==========================================
if __name__ == "__main__":
    main()