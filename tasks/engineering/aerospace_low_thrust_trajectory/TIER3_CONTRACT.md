# Tier 3: Full-MEE/J2 minimum-fuel transfer

Solve the fixed-time variable-throttle transfer using full osculating modified
equinoctial elements (MEE), central gravity and J2 perturbation. The objective
is minimum fuel over 300 days (25920000 seconds), with final mass and longitude
free. The initial orbit and spacecraft constants are those in problem_spec.md.
The terminal orbit has p=42164000 m and f=g=h=k=0. The trajectory must remain
elliptic, outside Earth, and have positive mass.

Use Pontryagin's minimum principle with the SI objective -m(t_f), measured in
kilograms, and all seven state costates. The delivered state and costate
history must satisfy the full dynamics, adjoint equations, terminal conditions
and unregularized minimum-Hamiltonian conditions. The numerical solution
method is your choice. Astrodynamics and trajectory-optimization packages
are not allowed; Python, NumPy and SciPy are available.

## Control and output interface

The control q=[q_r,q_t,q_n] is a dimensionless thrust-fraction vector in the
radial, transverse and normal orbital frame. Its norm is between zero and one;
actual thrust is T_max*q. Coasting and partial or full throttle are allowed.

Retain the four original required filenames. Tier 3 arrays have these columns:

```text
tier3_trajectory.npy:
[t,p,f,g,h,k,L,m,lambda_p,lambda_f,lambda_g,lambda_h,lambda_k,lambda_L,lambda_m]
tier3_control.npy:
[t,q_r,q_t,q_n]
```

Arrays must be real and finite, with identical strictly increasing timestamps
from 0 to 25920000 and 1000 to 2000000 rows. Longitude L is unwrapped.
Stored control vectors have piecewise-linear interpolation without
renormalization; states and costates describe that control history.
There is no physical slew-rate limit.

The output sampling contract requires each time gap to be at most 3600 seconds,
each longitude increment to be positive and at most 0.2 radians, and adjacent
control vectors to differ by at most 0.05 in Euclidean norm. These define the
required serialization resolution, not the integrator's internal step size.

The tier3 JSON object contains:

```text
formulation: "full_mee_j2_minimum_fuel_v2"
method: description of the method actually used
final_sma_km, final_eccentricity, final_inclination_deg
dv_total_m_s, transfer_time_days, final_mass_kg, fuel_consumed_kg
initial_costates: seven SI costates in the array order above
constraint_violation_norm, shooting_converged
hamiltonian_initial, hamiltonian_final
```

Report inclination in degrees, semi-major axis in kilometres, delta-v in m/s,
mass and fuel in kilograms, time in days, and SI Hamiltonians in kg/s.
For these reported scalars, a=p/(1-f*f-g*g), e=hypot(f,g),
i=2*atan(hypot(h,k)), delta-v=v_e*log(m_0/m_f), and fuel=m_0-m_f.
shooting_converged must be true for the delivered solution and describes
convergence regardless of the numerical solution method, including collocation.

constraint_violation_norm is the Euclidean norm of the seven terminal
boundary residuals: relative p error against 42164000 m; f, g, h and k;
the dimensionless longitude-costate transversality residual; and the
dimensionless mass-costate transversality residual. Dimensionless costates
use the state scales [R_E,1,1,1,1,1,m_0] divided by the objective scale m_0.
Writing the dimensionless costates as ell, the residual vector is
[(p_f-42164000)/42164000, f_f, g_f, h_f, k_f, ell_L(t_f), ell_m(t_f)+1].
All reported scalars must describe the saved trajectory.
