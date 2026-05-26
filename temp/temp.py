    ic = 0

    v_est = 0.0
    last_a = 0.0
    z_est = offset



    p_0 = np.diag([100, 100, 100])
    F = np.array([[1,dt,(1/2)*(dt**2)],
              [0,0,dt],
              [0,0,1]])

    # 0.1 is guassina noise from SPAD library measuremnt
    # 0.2 is variance from noise for the accelerometer
    Q = np.array([[0.1,0,0],
              [0,(0.2)*(dt**2),0.2*dt],
              [0,0.2*dt,0.2]])

    p_pred = np.array([[],[],[]])
    p_updated = np.array([[],[],[]])


    K_Gain = np.array([[],[],[]])

    # --- 2. Run the Simulation ---
    for tick in range(total_ticks):
        t = tick * dt

        # ==========================================
        # STEP A: SOURCE OF TRUTH (Updates every tick)
        # ==========================================
        true_z, true_a = get_true_reading(t,"SIN", offset, amplitude, omega)

        # Store Truth Data
        t_history.append(t)
        true_history.append([true_z, true_a, 0])




        # ==========================================
        # STEP C: SPAD SAMPLING
        # ==========================================
        # if tick % spad_interval == 0:
            # qui hai lo spad
            # z_est = spad_measure(true_z, spad_params)
        # else:
            # Kinematic integration over one tick (dt seconds):
            # z_{k+1} = z_k + v_k*dt + 0.5*a*dt^2
            # v_{k+1} = v_k + a*dt
        z_est += v_est * dt + 0.5 * last_a * dt**2

        # state predict step EKF
        last_a = accel_history[-1]
        v_est += last_a * dt
        measure_t_history.append(t)
        predicted = [z_est, last_a, v_est]
        measure_history.append(predicted)


        print(len(t_history))
        print(measure_history[-1])



        p_pred = F @@ p_updated @@ F.T + Q

        # KALMAN FILTER UPDATE STEP
        if tick % spad_interval == 0 and tick % accel_interval == 0:
            diff = [spad_measure(true_z, spad_params) - z_est, get_acc_reading(true_a) - last_a, 0]
        elif tick % spad_interval == 0:
            diff = [spad_measure(true_z, spad_params) - z_est, 0, 0]
        elif tick % accel_interval == 0:
            diff = [0, get_acc_reading(true_a) - last_a, 0]
        else:
            diff = [0,0,0]

        new_gain = F @@ p_pred @@ F.T + Q
        K_Gain.append(new_gain)

        p_updated = (np.identity(3) - new_gain @@ F) @@ p_pred

        measure_history[len(measure_history-1)] += new_gain @@ diff

        if tick % accel_interval == 0:
            accel_t_history.append(t)
            accel_history.append(get_acc_reading(true_a))
