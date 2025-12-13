function [fs, fb] = solve_mosfet_potential(ug, ui, gamma, phik, vt)
% SOLVE_MOSFET_POTENTIAL Solves coupled Poisson equations for symmetric DG-MOSFET
%
%   [fs, fb] = solve_mosfet_potential(ug, ui, gamma, phik, vt)
%
%   Inputs:
%       ug    : Normalized Gate Voltage (Vgs - Vfb) / Vt? (Check units of derivation)
%               *Note: Based on Eq 1, ug = Vgs - Vfb.
%       ui    : Normalized Doping Potential (2*phi_b - Vbs)
%       gamma : Body factor (sqrt(2*q*eps*Na)/Cox)
%       phik  : Structure factor for Eq 2
%       vt    : Thermal voltage (kT/q)
%
%   Outputs:
%       fs    : Surface Potential
%       fb    : Center Potential

    % --- 1. Initial Guess ---
    % A simple heuristic: fb follows ug but is clamped in inversion
    if ug > ui
        fb_guess = ui; % Inversion region
    elseif ug < 0
        fb_guess = ug; % Accumulation
    else
        fb_guess = ug * 0.5; % Depletion
    end
    
    fb = fb_guess;
    
    % --- 2. Newton-Raphson Parameters ---
    max_iter = 50;
    tol = 1e-12;
    delta = 1e-8; % Perturbation for finite difference
    
    for k = 1:max_iter
        % A. Calculate Residual at current fb
        [R0, fs0] = calculate_error(fb, ug, ui, gamma, phik, vt);
        
        % Check convergence
        if abs(R0) < tol
            fs = fs0;
            return;
        end
        
        % B. Calculate Jacobian (Finite Difference)
        % We perturb fb to find the sensitivity of the entire system
        fb_pert = fb + delta;
        [R_pert, ~] = calculate_error(fb_pert, ug, ui, gamma, phik, vt);
        
        Jacobian = (R_pert - R0) / delta;
        
        % C. Damping / Update
        % Avoid large jumps if Jacobian is small (though sqrt fix helps this)
        step = R0 / Jacobian;
        
        % Limit step size to avoid numerical overflow in exponentials
        step = max(min(step, 0.5), -0.5); 
        
        fb = fb - step;
    end
    
    % If we reach here, convergence failed
    warning('Newton-Raphson did not converge within %d iterations.', max_iter);
    fs = fs0;
end

function [Error, fs] = calculate_error(fb, ug, ui, gamma, phik, vt)
    % --- Step 1: Calculate fs (Slave Variable) ---
    % Using Equation 2: fs is explicit if fb is known
    % f2(fs, fb) = fb + phik*[...] - fs = 0  =>  fs = fb + phik*[...]
    
    % Charge density term at center (normalized)
    % term = 1 - exp(-ui/vt) + exp(-ui/vt)*exp(fb/vt) - exp(-fb/vt)
    rho_center = 1 - exp(-ui/vt) + exp((fb - ui)/vt) - exp(-fb/vt);
    
    fs = fb + phik * rho_center;

    % --- Step 2: Calculate Integrated Charge (RHS of Eq 1) ---
    % The terms inside the bracket of Eq 1
    % Term 1: exp(-ui/vt) * (exp(fs/vt) - exp(fb/vt))
    term1 = exp(-ui/vt) * (exp(fs/vt) - exp(fb/vt));
    
    % Term 2: exp(-fs/vt) - exp(-fb/vt)
    term2 = exp(-fs/vt) - exp(-fb/vt);
    
    % Term 3: Linear doping term
    term3 = (fb - fs) * (1 - exp(-ui/vt));
    
    % Total RHS (Electric Field Squared / vt)
    % Note: Multiplied by vt as per image Eq 1
    RHS_squared = vt * (term1 + term2) + term3;
    
    % Clamp RHS to 0 to avoid numerical noise creating imaginary numbers
    if RHS_squared < 0
        RHS_squared = 0;
    end
    
    % --- Step 3: Calculate Residual (Regularized) ---
    % ORIGINAL (Squared) Form: (ug - fs)^2/gamma^2 - RHS = 0
    % This has the zero-derivative issue at flatband.
    
    % REGULARIZED (Linear) Form: (ug - fs)/gamma - sgn * sqrt(RHS) = 0
    % This keeps derivative non-zero.
    
    % Determine sign based on accumulation vs inversion
    % If ug > fs, we expect positive field (pointing into surface)
    field_sign = sign(ug - fs);
    
    % Calculate Error
    % E = (ug - fs) - gamma * sign(ug-fs) * sqrt(RHS_squared)
    % Simplifies to:
    Error = (ug - fs) - field_sign * gamma * sqrt(RHS_squared);
    
    % Note: If you prefer the squared residual for debugging:
    % Error = (ug - fs)^2 - gamma^2 * RHS_squared; 
    % (But strictly avoid this for the solver!)
end