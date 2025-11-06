/*
 * dg_mosfet_solver.c
 *
 * This program solves the 1D Poisson equation for a double-gate MOSFET
 * using the analytical model derived in our previous steps.
 *
 * It uses Halley's method to numerically solve the complex implicit
 * equation for the channel center potential (psi_c).
 *
 * F(psi_c) = 0
 *
 * Where F(psi_c) = LHS - RHS
 *
 * LHS = [Cox/eps_si * (Vg - Vfb - psi_s)]^2
 * RHS = 2q/eps_si * [Na*(psi_s - psi_c) + nc*Vt*(exp((psi_s - psi_c)/Vt) - 1)]
 *
 * And all dependent variables (psi_s, nc) are also functions of psi_c:
 * nc(psi_c) = ni * exp((psi_c - Vch) / Vt)
 * psi_s(psi_c) = psi_c + [q*(Na + nc)*tsi^2] / (8*eps_si)
 *
 * Halley's Method Iteration:
 * psi_step = - (2 * F * F') / (2 * F'^2 - F * F'')
 * psi_c_new = psi_c + psi_step
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h> // For exit()

// --- Physical Constants ---
// (These would typically be in a separate constants header)
const double Q = 1.602176634e-19;    // Elementary Charge (C)
const double K_B = 1.380649e-23;   // Boltzmann Constant (J/K)
const double EPS_0 = 8.8541878128e-12; // Vacuum Permittivity (F/m)
const double EPS_SI = 11.7 * EPS_0; // Silicon Permittivity
const double EPS_OX = 3.9 * EPS_0;  // Silicon Dioxide Permittivity

// --- Solver Configuration ---
#define MAX_ITER 100
#define CONVERGENCE_TOL 1e-9

/**
 * @struct lxMosMmosPre
 * @brief Stores all pre-calculated device parameters and constants.
 *
 * This struct is passed to the solver function.
 */
typedef struct {
    // Device Geometry
    double tox; // Oxide thickness (m)
    double tsi; // Silicon film thickness (m)

    // Doping and Material Properties
    double Na;  // Acceptor doping concentration (m^-3)
    double ni;  // Intrinsic carrier concentration (m^-3)
    double vfb; // Flat-band voltage (V)

    // Pre-calculated constants for efficiency
    double Vt;      // Thermal voltage (V)
    double Cox;     // Single-gate Oxide Capacitance (F/m^2)
    double q;       // Elementary charge
    double eps_si;  // Silicon permittivity
    double eps_ox;  // Oxide permittivity

} lxMosMmosPre;

/**
 * @struct MosfetCalculations
 * @brief Stores the results from the solver.
 */
typedef struct {
    double psi_c;       // Center potential (V)
    double psi_s;       // Surface potential (V)
    double Q_g_total;   // Total gate charge (from both gates) (C/m^2)
    double Q_b;         // Total bulk (depletion) charge (C/m^2)
    double Q_n;         // Total inversion charge (C/m^2)
    int iterations;     // Number of iterations to converge
    int converged;      // Flag (1 for success, 0 for failure)
} MosfetCalculations;


/**
 * @brief Evaluates F(psi_c), F'(psi_c), and F''(psi_c) for Halley's method.
 *
 * This is the core of the solver, calculating the function and its
 * first two analytical derivatives with respect to psi_c.
 *
 * This is a very complex calculation due to the chain rule.
 * Suffixes:
 * _p  -> first derivative (d/d_psi_c)
 * _pp -> second derivative (d^2/d_psi_c^2)
 */
void evaluate_F_Fp_Fpp(double psi_c, double Vg, double Vch, const lxMosMmosPre* pre,
                       double* F, double* Fp, double* Fpp)
{
    // --- Pre-calculations and constants from struct ---
    if (pre->Vt == 0) {
        fprintf(stderr, "Fatal Error: Thermal voltage Vt is zero.\n");
        exit(1);
    }
    double V_t_inv = 1.0 / pre->Vt;
    
    // Constants for LHS and RHS equations
    double K_L = pre->Cox / pre->eps_si; // [Cox/eps_si]
    double K_R = 2.0 * pre->q / pre->eps_si; // [2q/eps_si]
    
    // Constant for the delta_psi approximation
    // d_psi = [q*tsi^2 / (8*eps_si)] * (Na + nc)
    double K_dpsi = (pre->q * pre->tsi * pre->tsi) / (8.0 * pre->eps_si);

    // --- 1. Center Concentration (nc) and its derivatives ---
    double n_c = pre->ni * exp((psi_c - Vch) * V_t_inv);
    double n_c_p = n_c * V_t_inv;
    double n_c_pp = n_c_p * V_t_inv;

    // --- 2. Potential Drop (d_psi = psi_s - psi_c) and its derivatives ---
    double d_psi = K_dpsi * (pre->Na + n_c);
    double d_psi_p = K_dpsi * n_c_p;
    double d_psi_pp = K_dpsi * n_c_pp;

    // --- 3. Surface Potential (psi_s) and its derivatives ---
    double psi_s = psi_c + d_psi;
    double psi_s_p = 1.0 + d_psi_p;
    double psi_s_pp = d_psi_pp;

    // --- 4. LHS = [K_L * (Vg - Vfb - psi_s)]^2 ---
    double Vgeff = Vg - pre->vfb;
    double L_A = K_L * (Vgeff - psi_s); // Intermediate term for L
    
    double L = L_A * L_A;
    double L_p = 2.0 * L_A * (-K_L * psi_s_p); // Chain rule
    double L_pp = 2.0 * (-K_L * psi_s_p) * (-K_L * psi_s_p) + 2.0 * L_A * (-K_L * psi_s_pp);
    // Simplified:
    // L_pp = 2.0 * K_L * K_L * psi_s_p * psi_s_p - 2.0 * L_A * K_L * psi_s_pp;
    
    // --- 5. RHS = K_R * [ R_A + R_B ] ---
    
    // R_A = Na * d_psi
    double R_A = pre->Na * d_psi;
    double R_A_p = pre->Na * d_psi_p;
    double R_A_pp = pre->Na * d_psi_pp;

    // R_B = nc * Vt * (exp(d_psi / Vt) - 1)
    double E_L = exp(d_psi * V_t_inv); // exp(d_psi/Vt)
    double E_L_p = E_L * (d_psi_p * V_t_inv); // d/d_psi_c [exp(d_psi/Vt)]
    double E_L_pp = (E_L_p * d_psi_p * V_t_inv) + (E_L * d_psi_pp * V_t_inv); // d^2/d_psi_c^2 [exp(d_psi/Vt)]
    
    double R_B = n_c * pre->Vt * (E_L - 1.0);
    // Product rule: (a*b)' = a'b + ab'
    // a = n_c, b = Vt*(E_L-1)
    double R_B_p = n_c_p * pre->Vt * (E_L - 1.0) + n_c * pre->Vt * E_L_p;
    
    // Product rule: (a*b)'' = a''b + 2a'b' + ab''
    double R_B_pp = n_c_pp * pre->Vt * (E_L - 1.0) +  /* a''b */
                    2.0 * n_c_p * pre->Vt * E_L_p +    /* 2a'b' */
                    n_c * pre->Vt * E_L_pp;            /* ab'' */

    // Combine for RHS
    double R = K_R * (R_A + R_B);
    double R_p = K_R * (R_A_p + R_B_p);
    double R_pp = K_R * (R_A_pp + R_B_pp);

    // --- 6. Final Function F = L - R and its derivatives ---
    *F = L - R;
    *Fp = L_p - R_p;
    *Fpp = L_pp - R_pp;
}


/**
 * @brief Solves for all potentials and charges given Vg and Vch.
 *
 * @param Vg Gate Voltage (V)
 * @param Vch Channel Potential (V) - (i.e., Vd or Vs for this 1D slice)
 * @param pre Pointer to the pre-calculated device parameters
 * @return MosfetCalculations struct containing all results.
 */
MosfetCalculations solve_mosfet(double Vg, double Vch, const lxMosMmosPre* pre)
{
    MosfetCalculations result = {0};

    // --- 1. Initial Guess for psi_c ---
    // A good guess is often the channel potential, or slightly below
    // the effective gate voltage. We'll start with Vch.
    double psi_c = Vch;
    double F, Fp, Fpp, psi_c_step;

    for (int i = 0; i < MAX_ITER; i++) {
        // --- 2. Calculate F, F', and F'' at the current psi_c ---
        evaluate_F_Fp_Fpp(psi_c, Vg, Vch, pre, &F, &Fp, &Fpp);

        // --- 3. Calculate Halley's Method Step ---
        double numerator = 2.0 * F * Fp;
        double denominator = 2.0 * Fp * Fp - F * Fpp;

        if (fabs(denominator) < 1e-20) {
            // Denominator is near zero, unstable.
            // Fall back to Newton's method step.
            if (fabs(Fp) < 1e-20) {
                // F' is also zero, solver is stuck.
                psi_c_step = -F; // Take a small, desperate step
            } else {
                psi_c_step = -F / Fp; // Newton step
            }
        } else {
            // Regular Halley's step
            psi_c_step = -numerator / denominator;
        }

        psi_c += psi_c_step;

        // --- 4. Check for Convergence ---
        if (fabs(psi_c_step) < CONVERGENCE_TOL) {
            result.iterations = i + 1;
            result.converged = 1;
            break;
        }
    }

    if (!result.converged) {
        fprintf(stderr, "Warning: Solver did not converge for Vg=%.2f, Vch=%.2f\n", Vg, Vch);
        result.iterations = MAX_ITER;
    }

    // --- 5. Post-Calculation (after convergence) ---
    // Now that we have the final psi_c, calculate all output values.
    
    result.psi_c = psi_c;

    // Recalculate final values based on converged psi_c
    double n_c = pre->ni * exp((psi_c - Vch) / pre->Vt);
    double d_psi = (pre->q * pre->tsi * pre->tsi) / (8.0 * pre->eps_si) * (pre->Na + n_c);
    
    result.psi_s = psi_c + d_psi;

    // --- 6. Calculate Charges ---
    
    // Bulk Charge (exact, constant)
    result.Q_b = -pre->q * pre->Na * pre->tsi;

    // Total Gate Charge (from Gauss's Law at the gates)
    // Q_g_total = 2 * Q_g_single
    // Q_g_single = Cox * V_ox = Cox * (Vg - Vfb - psi_s)
    result.Q_g_total = 2.0 * pre->Cox * (Vg - pre->vfb - result.psi_s);

    // Inversion Charge (from charge balance)
    // Q_g_total + Q_n + Q_b = 0
    result.Q_n = -result.Q_g_total - result.Q_b;

    return result;
}


/**
 * @brief Main function to demonstrate the solver.
 */
int main() {
    // --- 1. Setup Device Parameters ---
    lxMosMmosPre myDevice;
    
    // Physical Parameters
    double T = 300.15; // Temperature (K)
    myDevice.tsi = 10e-9; // 10 nm silicon film
    myDevice.tox = 1.2e-9; // 1.2 nm oxide
    myDevice.Na = 1e22;    // 1e16 cm^-3 -> 1e22 m^-3 (Doping)
    myDevice.ni = 1e16;    // 1e10 cm^-3 -> 1e16 m^-3 (Intrinsic)
    myDevice.vfb = -0.3;   // Flat-band voltage

    // Store constants in the struct
    myDevice.q = Q;
    myDevice.eps_si = EPS_SI;
    myDevice.eps_ox = EPS_OX;

    // Pre-calculate common terms
    myDevice.Vt = (K_B * T) / Q;
    myDevice.Cox = myDevice.eps_ox / myDevice.tox;

    printf("--- Device Parameters ---\n");
    printf("Vt:  %.4f V\n", myDevice.Vt);
    printf("Cox: %.4e F/m^2\n", myDevice.Cox);
    printf("Tsi: %.2f nm\n", myDevice.tsi * 1e9);
    printf("Tox: %.2f nm\n", myDevice.tox * 1e9);
    printf("Na:  %.2e m^-3\n", myDevice.Na);
    printf("Vfb: %.2f V\n", myDevice.vfb);
    printf("-------------------------\n\n");

    // --- 2. Run Solver ---
    
    double Vg = 1.0;  // Gate voltage
    double Vch = 0.2; // Channel voltage (e.g., Vd = 0.2)

    printf("--- Running Solver for Vg=%.2fV, Vch=%.2fV ---\n", Vg, Vch);
    
    MosfetCalculations results = solve_mosfet(Vg, Vch, &myDevice);

    // --- 3. Print Results ---
    if (results.converged) {
        printf("Solver Converged in %d iterations.\n", results.iterations);
        printf("\n--- Potentials ---\n");
        printf("psi_c (Center): %.6f V\n", results.psi_c);
        printf("psi_s (Surface):  %.6f V\n", results.psi_s);
        
        printf("\n--- Charges (C/m^2) ---\n");
        printf("Q_b (Bulk):     %.6e C/m^2\n", results.Q_b);
        printf("Q_g (Total Gate): %.6e C/m^2\n", results.Q_g_total);
        printf("Q_n (Inversion):  %.6e C/m^2\n", results.Q_n);
    } else {
        printf("Solver FAILED to converge.\n");
    }

    return 0;
}
