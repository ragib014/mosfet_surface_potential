function output = solve_dg_poisson()
    % SOLVE_DG_POISSON 
    % Solves surface potential and charge for Double-Gate MOSFETs under:
    % 1. Classical (Taur) 
    % 2. Quantum (Log-Linear + Dark Space)
    % 3. Classical + Traps
    % 4. Quantum + Traps
    
    %% --- 1. PHYSICAL CONSTANTS & DEVICE PARAMETERS ---
    q       = 1.602e-19;    % Elementary charge (C)
    kB      = 1.38e-23;     % Boltzmann constant (J/K)
    eps0    = 8.854e-12;    % Vacuum permittivity (F/m)
    eps_si  = 11.7 * eps0;  % Silicon permittivity
    eps_ox  = 3.9 * eps0;   % Oxide permittivity
    h_bar   = 1.054e-34;    % Reduced Planck constant
    m0      = 9.11e-31;     % Electron rest mass
    
    % -- Device Geometry & Temp --
    T       = 300;          % Temperature (K)
    kT      = kB * T;
    Vt      = kT / q;       % Thermal Voltage
    Tsi     = 7e-9;         % Silicon Thickness (7 nm)
    EOT     = 1e-9;         % Equivalent Oxide Thickness (1 nm)
    Cox     = eps_ox / EOT; % Oxide Capacitance (F/m^2)
    Vfb     = 0.0;          % Flatband Voltage
    ni      = 1.5e10 * 1e6; % Intrinsic carrier conc (m^-3)
    
    % -- Quantum Parameters (Si 100) --
    % Ground state: Delta_2 valleys (Heavy mass in quantization direction)
    mz_conf = 0.91 * m0;    % Confinement effective mass
    md_dos  = 0.19 * m0;    % Density of states effective mass (transverse)
    gv      = 2;            % Valley degeneracy
    
    % First Subband Energy (Infinite Well Approx)
    E1      = (h_bar^2 * pi^2) / (2 * mz_conf * Tsi^2); 
    E1_eV   = E1 / q;
    
    % Effective 2D DOS (N_c)
    % D0 = gv * m_dos / (pi * h_bar^2)
    D0      = gv * md_dos / (pi * h_bar^2); 
    Nc_2D   = D0 * kT;      % Effective density (m^-2)

    % -- Interface Trap Parameters (Sech^2 Model) --
    Nit_peak = 2e12 * 1e4;  % Peak Trap Density (m^-2 eV^-1). 2e12 cm^-2
    Et_center= 0.0;         % Trap center relative to midgap (eV)
    sigma_E  = 0.1;         % Trap width (eV) (Disorder + Thermal)
    
    
    %% --- 2. SETUP SWEEP ---
    % We sweep the CENTER POTENTIAL (psi_c) and calculate everything else.
    % This avoids convergence issues in the subthreshold to inversion transition.
    psi_c_sweep = linspace(-0.2, 1.2, 100); 
    
    % Initialize Results Structures
    results = struct('Vg', [], 'Psis', [], 'Qg', [], 'name', '');
    cases = [1, 2, 3, 4];
    plot_styles = {'b-', 'r--', 'g-.', 'k-'};
    plot_names = {'Classical', 'Quantum', 'Classical+Traps', 'Quantum+Traps'};

    figure('Name', 'Physics Comparison', 'Color', 'w');
    subplot(2,1,1); hold on; grid on; xlabel('V_{GS} (V)'); ylabel('\psi_s (V)');
    title('Surface Potential vs Gate Voltage');
    
    subplot(2,1,2); hold on; grid on; xlabel('V_{GS} (V)'); ylabel('Q_{Gate} (\mu C/cm^2)');
    title('Total Gate Charge vs Gate Voltage');

    %% --- 3. SOLVER LOOP ---
    for c_idx = 1:4
        case_num = cases(c_idx);
        
        Vg_arr   = zeros(size(psi_c_sweep));
        Psis_arr = zeros(size(psi_c_sweep));
        Qg_arr   = zeros(size(psi_c_sweep));
        
        for i = 1:length(psi_c_sweep)
            pc = psi_c_sweep(i);
            
            % -- A. SOLVE SEMICONDUCTOR PHYSICS (Internal) --
            if case_num == 1 || case_num == 3
                % === CLASSICAL (TAUR FORMALISM) ===
                % Solve for beta: exp(q*pc/kT) = (2*eps*kT/(q^2*ni)) * (2*beta/Tsi)^2
                % Rearranged: beta^2 = Const * exp(pc/Vt)
                
                K_taur = (q^2 * ni) / (2 * eps_si * kT);
                beta_sq = (Tsi/2)^2 * K_taur * exp(pc/Vt);
                beta = sqrt(beta_sq);
                
                % Check limit for beta (beta < pi/2 for solution existence)
                if beta >= pi/2
                    beta = pi/2 - 1e-6; % Clamp to avoid singularity
                end
                
                % Calculate Surface Potential (Classical)
                % psi(x) = pc - 2Vt*ln(cos(beta*2x/Tsi))
                % at surface x = Tsi/2 -> cos(beta)
                ps = pc - 2*Vt * log(cos(beta));
                
                % Calculate Classical Mobile Charge
                % Q_semi = (4*eps*Vt/Tsi) * beta * tan(beta)
                Q_semi = (4 * eps_si * Vt / Tsi) * beta * tan(beta);
                
            else
                % === QUANTUM (LOG-LINEAR + DARK SPACE) ===
                % 1. Calculate Integrated Charge Density (N_inv)
                % Arg = (q*pc - E1)/kT. Note: pc is usually defined relative to E_int.
                % Here assuming pc is band bending.
                arg = (q*pc - E1) / kT;
                
                % N_inv = Nc * ln(1 + exp(arg))
                N_inv = Nc_2D * log(1 + exp(arg));
                Q_semi = q * N_inv;
                
                % 2. Calculate Quantum Potential Drop (Dark Space)
                % Delta_psi = (q * N_inv * Tsi / 8*eps) * (1 - 4/pi^2)
                geom_factor = (1 - 4/(pi^2));
                delta_psi_qm = (Q_semi * Tsi / (8 * eps_si)) * geom_factor;
                
                ps = pc + delta_psi_qm;
            end
            
            % -- B. SOLVE INTERFACE PHYSICS (Traps) --
            Q_it = 0;
            if case_num == 3 || case_num == 4
                % Tanh Trap Model
                % Charge = -q * N_peak * tanh((q*ps - E_center)/sigma)
                trap_arg = (q*ps - Et_center*q) / (sigma_E*q);
                Q_it = -q * Nit_peak * tanh(trap_arg);
            end
            
            % -- C. SOLVE BOUNDARY CONDITION (Gate Voltage) --
            % Vg = Vfb + ps + (Q_semi + Q_it_magnitude) / Cox
            % Note: Q_semi is electron charge magnitude (positive in formula above)
            % Q_it is usually negative for acceptor traps.
            % Total charge on gate Qg = + (Q_semi_electrons + Q_trapped_electrons)
            
            Q_total_gate = Q_semi + abs(Q_it); % Assuming Q_it is negative charge
            
            Vg = Vfb + ps + Q_total_gate / Cox;
            
            % Store
            Vg_arr(i)   = Vg;
            Psis_arr(i) = ps;
            Qg_arr(i)   = Q_total_gate * 1e4; % Convert to uC/cm^2? No, just C/m^2 * 1e4 -> C/cm^2? 
            % Let's keep SI: C/m^2. 
            % For plot, convert to uC/cm^2: 1 C/m^2 = 100 uC/cm^2
            Qg_arr(i)   = Q_total_gate * 100; 
        end
        
        % Plotting
        subplot(2,1,1); plot(Vg_arr, Psis_arr, plot_styles{c_idx}, 'DisplayName', plot_names{c_idx}, 'LineWidth', 1.5);
        subplot(2,1,2); plot(Vg_arr, Qg_arr,   plot_styles{c_idx}, 'DisplayName', plot_names{c_idx}, 'LineWidth', 1.5);
    end
    
    subplot(2,1,1); legend('Location','best'); xlim([0 1.0]);
    subplot(2,1,2); legend('Location','best'); xlim([0 1.0]); set(gca, 'YScale', 'log');
    
    output = 'Simulation Complete';
end