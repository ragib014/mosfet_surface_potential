function [V, x, y] = solve_dg_mosfet_poisson()
    % Solves the 2D non-linear Poisson equation for a double-gate (DG) MOSFET
    % using the Finite Difference Method (FDM) and Newton-Raphson iteration.
    %
    % This script simulates the top half of the device (oxide + half of Si)
    % due to symmetry.
    %
    % Domain:
    %   y = y_gate (t_si/2 + t_ox)  <-- Gate (Dirichlet)
    %   ---------------------------
    %   |         Oxide           |
    %   ---------------------------
    %   y = t_si/2                  <-- Si/SiO2 Interface
    %   |                         |
    %   |         Silicon         |
    %   |                         |
    %   y = 0                       <-- Channel Center (Neumann)
    %   ---------------------------
    %   x = 0     x = L_ch
    %   ^         ^
    % Source    Drain
    % (Dirichlet) (Dirichlet)
    %

    % 1. Constants
    q = 1.602e-19;      % Elementary charge (C)
    k = 1.380e-23;      % Boltzmann constant (J/K)
    T = 300;            % Temperature (K)
    eps0 = 8.854e-12;   % Permittivity of free space (F/m)
    eps_si_r = 11.7;    % Relative permittivity of Silicon
    eps_ox_r = 3.9;     % Relative permittivity of SiO2
    eps_si = eps_si_r * eps0;
    eps_ox = eps_ox_r * eps0;
    VT = k * T / q;     % Thermal voltage (V)
    ni = 1.0e16;        % Intrinsic carrier concentration in Si (m^-3)
                        % (Using 1e10 cm^-3 = 1e16 m^-3)

    % 2. Device Parameters (Distances in meters)
    L_ch = 60e-9;       % Channel length
    t_si = 10e-9;       % Silicon body thickness
    t_ox = 2e-9;        % Gate oxide thickness
    
    Na = 1e23;          % p-type body doping (m^-3) (1e17 cm^-3)
    Nd_sd = 1e26;       % n-type S/D doping (m^-3) (1e20 cm^-3)
    
    % 3. Applied Voltages
    Vgs = 1.0;          % Gate-Source voltage (V)
    Vds = 0.5;          % Drain-Source voltage (V)
    
    Vfb = -0.55;        % Flat-band voltage (V). 
                        % Depends on gate workfunction and doping. 
                        % This is a typical value for n+ poly gate on p-Si.
                        % For an ideal mid-gap gate, Vfb would be ~ -phi_Fp.
                        % Let's use a user-defined Vfb.

    % 4. Grid Setup
    Nx = 61;            % Number of grid points in x (channel length)
    Ny = 31;            % Number of grid points in y (thickness)
    N_total = Nx * Ny;  % Total number of nodes
    
    x = linspace(0, L_ch, Nx);
    y_domain_top = t_si/2 + t_ox;
    y = linspace(0, y_domain_top, Ny);
    
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    
    % Find the Si/SiO2 interface index
    % j_int is the first grid point IN THE OXIDE
    j_int = find(y >= t_si/2, 1);
    if isempty(j_int)
        error('Grid does not cover oxide layer.');
    end
    % Adjust: all j < j_int are Si, j >= j_int are Oxide
    % To be safe, let's say j_int is the interface point itself
    [~, j_int_idx] = min(abs(y - t_si/2));
    j_int = j_int_idx; 
    
    
    % 5. Helper Arrays
    
    % Permittivity array (eps_r at each grid node)
    eps_r = ones(Ny, Nx);
    eps_r(1:j_int, :) = eps_si_r;
    eps_r(j_int+1:Ny, :) = eps_ox_r;
    
    % Quasi-Fermi potentials
    % phi_p = 0 (grounded body)
    % phi_n varies linearly from 0 (Source) to Vds (Drain)
    phi_p = zeros(Ny, Nx);
    phi_n_vec = linspace(0, Vds, Nx);
    phi_n = repmat(phi_n_vec, Ny, 1);
    
    % 6. Boundary Conditions
    % We solve for the electrostatic potential psi.
    % V_bi is the built-in potential of the n+/p-body junction
    % This is the equilibrium potential of the n+ region relative to
    % the intrinsic level.
    V_bi_sd = VT * log(Nd_sd / ni);
    
    % Potential at the gate contact
    V_gate_eff = Vgs - Vfb;
    
    % Source (x=0) and Drain (x=L_ch) potentials
    % These are Dirichlet BCs
    V_source_bc = V_bi_sd + 0;      % Potential in n+ source
    V_drain_bc = V_bi_sd + Vds;     % Potential in n+ drain
    
    % 7. Initial Guess for V (Potential)
    V = zeros(Ny, Nx);
    
    % Linearly interpolate initial guess from boundaries
    for j = 1:Ny
        V(j, :) = linspace(V_source_bc, V_drain_bc, Nx);
    end
    for i = 1:Nx
        V(:, i) = linspace(V(1, i), V_gate_eff, Ny);
    end
    
    % 8. Iterative Solver (Newton-Raphson)
    max_iter = 100;
    tolerance = 1e-6;   % Convergence tolerance in Volts
    
    fprintf('Starting Newton-Raphson solver...\n');
    
    for iter = 1:max_iter
        V_old = V;
        
        % Create sparse matrix A and vector b
        A = spalloc(N_total, N_total, 5 * N_total);
        b = zeros(N_total, 1);
        
        % Loop over all grid points
        for j = 1:Ny  % y-direction
            for i = 1:Nx  % x-direction
                
                % 1D index for the current node (i, j)
                k = (j-1)*Nx + i;
                
                % --- Check for Boundary Conditions ---
                
                % Top Gate (j = Ny) - Dirichlet
                if j == Ny
                    A(k, k) = 1.0;
                    b(k) = V_gate_eff;
                    
                % Source (i = 1) - Dirichlet
                elseif i == 1
                    A(k, k) = 1.0;
                    b(k) = V_source_bc;

                % Drain (i = Nx) - Dirichlet
                elseif i == Nx
                    A(k, k) = 1.0;
                    b(k) = V_drain_bc;

                % Centerline (j = 1) - Neumann (d_psi/d_y = 0)
                elseif j == 1
                    % We use a "ghost node" at j=0, where V(i, 0) = V(i, 2)
                    % This modifies the standard FDM stencil
                    
                    % Get permittivity values (all silicon at j=1)
                    eps_x_p = (eps_r(j,i) + eps_r(j,i+1))/2 * eps0;
                    eps_x_m = (eps_r(j,i) + eps_r(j,i-1))/2 * eps0;
                    eps_y_p = (eps_r(j,i) + eps_r(j+1,i))/2 * eps0;
                    % eps_y_m uses ghost node, eps_r(0,i) = eps_r(2,i)
                    eps_y_m = eps_y_p; % By symmetry
                    
                    % Get charge density and its derivative
                    [rho, rho_p] = get_charge(V_old(j,i), phi_n(j,i), phi_p(j,i), Na, j, j_int, q, VT, ni);

                    % 1D indices for neighbors
                    k_xp = k + 1;
                    k_xm = k - 1;
                    k_yp = k + Nx; % Node (i, 2)
                    
                    % Assemble A matrix row
                    A(k, k_xp) = eps_x_p / dx^2;
                    A(k, k_xm) = eps_x_m / dx^2;
                    A(k, k_yp) = (eps_y_p + eps_y_m) / dy^2; % 2x term
                    
                    A(k, k) = - (eps_x_p + eps_x_m)/dx^2 ...
                            - (eps_y_p + eps_y_m)/dy^2 ...
                            + rho_p; % d(-rho)/dV = -rho_p
                            
                    % Assemble b vector
                    b(k) = -rho + rho_p * V_old(j,i); % -rho - (-rho_p * V_old)
                
                % --- Internal Nodes (Silicon or Oxide) ---
                else
                    % Get permittivity values at interfaces between nodes
                    eps_x_p = (eps_r(j,i) + eps_r(j,i+1))/2 * eps0;
                    eps_x_m = (eps_r(j,i) + eps_r(j,i-1))/2 * eps0;
                    eps_y_p = (eps_r(j,i) + eps_r(j+1,i))/2 * eps0;
                    eps_y_m = (eps_r(j,i) + eps_r(j-1,i))/2 * eps0;
                    
                    % Get charge density and its derivative
                    % This will be 0 in the oxide (j >= j_int)
                    [rho, rho_p] = get_charge(V_old(j,i), phi_n(j,i), phi_p(j,i), Na, j, j_int, q, VT, ni);

                    % 1D indices for neighbors
                    k_xp = k + 1;
                    k_xm = k - 1;
                    k_yp = k + Nx;
                    k_ym = k - Nx;
                    
                    % Assemble A matrix row
                    % FDM for nabla . (eps * nabla(V)) = -rho(V)
                    % Linearized: A*V_new = b
                    
                    A(k, k_xp) = eps_x_p / dx^2;
                    A(k, k_xm) = eps_x_m / dx^2;
                    A(k, k_yp) = eps_y_p / dy^2;
                    A(k, k_ym) = eps_y_m / dy^2;
                    
                    A(k, k) = - (eps_x_p + eps_x_m)/dx^2 ...
                            - (eps_y_p + eps_y_m)/dy^2 ...
                            + rho_p; % d(-rho)/dV = -rho_p

                    % Assemble b vector
                    % b = -rho(V_old) - d(-rho)/dV * V_old
                    b(k) = -rho + rho_p * V_old(j,i);
                end
            end % end i loop
        end % end j loop
        
        % Solve the linear system
        V_new_vec = A \ b;
        
        % Reshape 1D vector back to 2D matrix
        V = reshape(V_new_vec, [Nx, Ny])';
        
        % Check for convergence
        max_change = max(abs(V(:) - V_old(:)));
        fprintf('Iteration %d: Max potential change = %e V\n', iter, max_change);
        
        if max_change < tolerance
            fprintf('Solution converged in %d iterations.\n', iter);
            break;
        end
        
        if iter == max_iter
            fprintf('Warning: Solution did not converge after %d iterations.\n', max_iter);
        end
    end % end iteration loop

    % 9. Plot Results
    figure('Name', '2D Potential Profile (V)', 'NumberTitle', 'off');
    [X, Y] = meshgrid(x * 1e9, y * 1e9); % Convert to nm for plotting
    surf(X, Y, V);
    shading interp;
    xlabel('Channel Length (nm)');
    ylabel('Thickness (nm)');
    zlabel('Potential (V)');
    title(sprintf('2D Potential Profile (Vgs = %.1f V, Vds = %.1f V)', Vgs, Vds));
    colorbar;
    view(2); % Top-down 2D view
    
    % Plot vertical cutline at the middle of the channel
    figure('Name', 'Potential Cutline (Channel Center)', 'NumberTitle', 'off');
    mid_x_idx = round(Nx / 2);
    plot(y * 1e9, V(:, mid_x_idx));
    xlabel('Thickness (nm)');
    ylabel('Potential (V)');
    title(sprintf('Vertical Potential at x = %.1f nm', x(mid_x_idx)*1e9));
    grid on;
    
    % Plot horizontal cutline at the Si/SiO2 interface
    figure('Name', 'Potential Cutline (Si/SiO2 Interface)', 'NumberTitle', 'off');
    plot(x * 1e9, V(j_int, :));
    xlabel('Channel Length (nm)');
    ylabel('Potential (V)');
    title(sprintf('Horizontal Potential at Si/SiO2 Interface (y = %.1f nm)', y(j_int)*1e9));
    grid on;

end

function [rho, rho_p] = get_charge(V, phi_n, phi_p, Na, j, j_int, q, VT, ni)
    % Calculates charge density (rho) and its derivative (rho_p = d(rho)/dV)
    
    % j_int is the index of the Si/SiO2 interface
    % If j >= j_int, we are in the oxide (or at the interface)
    % We assume charge is zero at/in the oxide
    if j > j_int
        rho = 0;
        rho_p = 0;
        return;
    end
    
    % We are in Silicon
    n = ni * exp((V - phi_n) / VT);
    p = ni * exp((phi_p - V) / VT);
    
    % Charge density (C/m^3)
    % Assuming p-type, Na is acceptor (negative ion)
    rho = q * (p - n - Na);
    
    % Derivative of charge density w.r.t. potential V
    % d(p)/dV = -(1/VT) * p
    % d(n)/dV = (1/VT) * n
    rho_p = q * ( -(1/VT)*p - (1/VT)*n );
end