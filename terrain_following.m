%% Estrazione Profilo Batimetrico e Altitudine SPAD (Scenario Terrain-Following)
clc; clear; close all;

% =========================================================================
% --- 1. Caricamento dati Batimetrici ---
% =========================================================================
file = 'E6_2024.nc';
lon = ncread(file,'lon');
lat = ncread(file,'lat');
Z   = ncread(file,'elevation');

% Limiti geografici
lon_max = 14; lat_min = 45.5; lat_max = 46;
idx_lon = lon <= lon_max;
idx_lat = lat >= lat_min & lat <= lat_max;
Z = Z(idx_lon, idx_lat)';   
lon = lon(idx_lon);
lat = lat(idx_lat);

% =========================================================================
% --- 2. Conversione in coordinate cartesiane (metri) ---
% =========================================================================
lon0 = mean(lon); lat0 = mean(lat);
[X, Y] = meshgrid(lon, lat);
X = (X - lon0) * 111320 * cosd(lat0);
Y = (Y - lat0) * 111320;
valid = ~isnan(Z);

% Creazione della funzione interpolante per la batimetria
Fdepth = scatteredInterpolant(X(valid), Y(valid), Z(valid), 'natural','none');

% =========================================================================
% --- 3. IMPOSTAZIONI TRAIETTORIA SOTTOMARINO (TERRAIN FOLLOWING) ---
% =========================================================================
% Taglio per beccare una zona con dei dislivelli reali
start_X_frac = 0.50; 
start_Y_frac = 0.50;  
end_X_frac   = 0.80;  
end_Y_frac   = 0.50;  

X_valid = X(valid);
Y_valid = Y(valid);
Xmin = min(X_valid); Xmax = max(X_valid);
Ymin = min(Y_valid); Ymax = max(Y_valid);

x_start = Xmin + (Xmax - Xmin) * start_X_frac;
y_start = Ymin + (Ymax - Ymin) * start_Y_frac;
x_end   = Xmin + (Xmax - Xmin) * end_X_frac;
y_end   = Ymin + (Ymax - Ymin) * end_Y_frac;

% Quota DESIDERATA di volo sopra il fondale
h_target = 3.0; % Il drone cerca di restare a 3 metri dal fondo

% =========================================================================
% --- 4. Calcolo Cinematica per Python (Inerzia e Sensori) ---
% =========================================================================
num_points = 1500; % Risoluzione
x_path = linspace(x_start, x_end, num_points);
y_path = linspace(y_start, y_end, num_points);

% Vettore temporale base
dist_path = sqrt((x_path - x_start).^2 + (y_path - y_start).^2); 
speed_xy = 2.0; % [m/s]
t_path = dist_path / speed_xy;
dt_path = mean(diff(t_path));

% Quota assoluta del fondale
Z_fondale = Fdepth(x_path, y_path);

% Traiettoria Ideale: copiatura perfetta del fondale
Z_target = Z_fondale + h_target;

% MODELLO FISICO DEL DRONE (Filtro Passa-Basso)
% ---> FIX: Parametro aggiornato per inerzia BlueROV2
tau_drone = 0.6; % Costante di tempo in secondi (tra 0.4 e 0.8 s)
alpha = dt_path / (tau_drone + dt_path);

Z_AUV = zeros(1, num_points);
Z_AUV(1) = Z_target(1); % Partenza alla quota perfetta

% Calcolo della traiettoria REALE del drone (Smoothed)
for k = 2:num_points
    Z_AUV(k) = alpha * Z_target(k) + (1 - alpha) * Z_AUV(k-1);
end

% --- MISURAZIONI SENSORIALI PER L'EKF ---
% 1. SPAD (Distanza Relativa): La distanza vera tra la pancia del drone e il fondo
distanza_spad = Z_AUV - Z_fondale;
distanza_spad(distanza_spad < 0) = 0.001; % Prevenzione crash numerico

% 2. Accelerometro (Accelerazione RELATIVA): CORRETTO PER L'EKF DI PYTHON
V_z_rel = gradient(distanza_spad) / dt_path; 
A_z_rel = gradient(V_z_rel) / dt_path;
A_z_rel = smoothdata(A_z_rel, 'gaussian', 15); % Smussamento

% =========================================================================
% --- 5. Esportazione CSV per Python ---
% =========================================================================
% Esporta solo la geometria: posizione spaziale lungo il percorso
T_export = table(dist_path', Z_fondale', Z_AUV', ...
    'VariableNames', {'Distance', 'Z_fondale', 'Z_AUV'});
writetable(T_export, 'profilo_geometrico.csv');

% =========================================================================
% --- 6. PLOT 3D: MAPPA BATIMETRICA E VOLO SOTTOMARINO ---
% =========================================================================
figure('Name', 'Mappa 3D: Volo Sottomarino', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
colormap parula; colorbar; hold on;

% Piano di taglio semitrasparente (Altezza Z modificata a 1.25)
z_bottom = min(Z(:)) - 10;
z_top = 1.25; 
patch([x_start, x_end, x_end, x_start], ...
      [y_start, y_end, y_end, y_start], ...
      [z_bottom, z_bottom, z_top, z_top], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');

% Disegno 1: Profilo del fondale (blu scuro)
plot3(x_path, y_path, Z_fondale, 'b', 'LineWidth', 2);
% Disegno 2: Traiettoria Ideale (magenta tratteggiata)
plot3(x_path, y_path, Z_target, 'm--', 'LineWidth', 1.5);
% Disegno 3: Traiettoria Reale del Drone ritardata (verde neon)
plot3(x_path, y_path, Z_AUV, 'g', 'LineWidth', 3);

% Raggi SPAD virtuali tra il drone e il fondo
step = floor(num_points/15);
for k = 1:step:num_points
    plot3([x_path(k) x_path(k)], [y_path(k) y_path(k)], [Z_AUV(k), Z_fondale(k)], 'k-', 'LineWidth', 1);
end

xlabel('X [m]'); ylabel('Y [m]'); zlabel('Quota Assoluta [m]');
title('Traiettoria AUV Reale (Verde) e Target (Magenta)');
view(-15, 30); grid on;
legend('Mappa', 'Superficie', 'Fondale Sotto AUV', 'Target 3m', 'Traiettoria Reale AUV', 'AutoUpdate', 'off');

% =========================================================================
% --- 7. PLOT 2D: L'INPUT CHE ANDRA' A PYTHON ---
% =========================================================================
figure('Name', 'Input per Python (EKF)', 'NumberTitle', 'off', 'Position', [1050, 100, 600, 600]);

subplot(2,1,1); 
plot(t_path, distanza_spad, 'k', 'LineWidth', 2); hold on;
yline(h_target, 'm--', 'Target 3m', 'LineWidth', 1.5);
title('Misurazione SPAD: Distanza Relativa dal Fondale (Errore di tracking)');
ylabel('Distanza Z (m)'); grid on;

subplot(2,1,2); 
plot(t_path, A_z_rel, 'r', 'LineWidth', 1.5);
title('Misurazione IMU: Accelerazione Verticale Relativa (Corretta)');
xlabel('Tempo (s)'); ylabel('Accelerazione (m/s^2)'); grid on;

% =========================================================================
% --- CALCOLO DELLA DISTANZA REALE PERCORSA DAL DRONE (3D) ---
% =========================================================================
% Calcola i delta (le differenze) tra ogni punto consecutivo della traiettoria
dx = diff(x_path);
dy = diff(y_path);
dz = diff(Z_AUV); % Spostamento verticale reale del drone

% Distanza 3D per ogni singolo passetto
passi_3d = sqrt(dx.^2 + dy.^2 + dz.^2);

% Somma tutti i passetti per avere la distanza totale
distanza_totale_3d = sum(passi_3d);

% Per confronto, calcoliamo anche solo quella planare (orizzontale)
distanza_orizzontale = dist_path(end);

% Stampa i risultati a schermo nel Command Window
fprintf('\n==================================================\n');
fprintf('Analisi della Traiettoria del Drone:\n');
fprintf('-> Distanza Orizzontale (in linea retta XY): %.2f metri\n', distanza_orizzontale);
fprintf('-> DISTANZA REALE PERCORSA (3D con salite/discese): %.2f metri\n', distanza_totale_3d);
fprintf('==================================================\n');