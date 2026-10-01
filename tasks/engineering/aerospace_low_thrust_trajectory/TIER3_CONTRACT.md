# Tier 3: Full-MEE/J2 Minimum-Fuel Transfer, Version 2

This section replaces the original Tier 3 only. Tier 1 and Tier 2 are unchanged.
Solve the fixed-time, variable-throttle optimal-control problem below. Use the
osculating, non-averaged dynamics throughout the final solve and deliverables.
An averaged model may initialize a continuation method but is not a solution.

## Physical Problem

Use SI units, mu=3.986004418e14, R_E=6378137, J2=1.08263e-3,
T_max=0.5 N, v_e=29419.95 m/s, m_0=2000 kg, and t_f=25920000 s.
The initial state x=[p,f,g,h,k,L,m] is
[6678000,0,0,tan(28.5*pi/360),0,0,2000].
The terminal constraints are p=42164000 and f=g=h=k=0.
Final longitude and mass are free. Minimize -m(t_f), equivalently fuel consumed.
The trajectory must remain elliptic, outside Earth, with positive mass.

The RTN control q=[q_r,q_t,q_n] is a thrust fraction vector with norm rho in
[0,1]. Actual thrust is T_max*q. Coast (rho=0), partial throttle, and full
throttle are allowed. Fuel is not fixed by the transfer duration.

Define c=cos(L), s=sin(L), w=1+f*c+g*s, s2=1+h*h+k*k,
zeta=h*s-k*c, eta=h*c+k*s, r=p/w, and b=sqrt(p/mu).
The perturbing RTN acceleration a is the sum of T_max*q/m and

```text
a_J2_r = -1.5 * J2*mu*R_E^2/r^4 * (1-12*zeta^2/s2^2)
a_J2_t = -12  * J2*mu*R_E^2/r^4 * zeta*eta/s2^2
a_J2_n = -6   * J2*mu*R_E^2/r^4 * zeta*(1-h^2-k^2)/s2^2
```

Integrate in time, not an inconsistent mixture of time and longitude:

```text
p_dot = b * 2*p*a_t/w
f_dot = b * (a_r*s + a_t*((w+1)*c+f)/w - a_n*g*zeta/w)
g_dot = b * (-a_r*c + a_t*((w+1)*s+g)/w + a_n*f*zeta/w)
h_dot = b * a_n*s2*c/(2*w)
k_dot = b * a_n*s2*s/(2*w)
L_dot = sqrt(mu*p)*(w/p)^2 + b*zeta*a_n/w
m_dot = -T_max*rho/v_e
```

In particular, the normal-acceleration term in L_dot must not be dropped.
The six-row matrix B is the coefficient of a in these equations, including
its longitude row. These are the standard modified equinoctial equations;
see the [JPL MEE specification](https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf).

## Optimality Certificate

Let lambda=[lambda_p,lambda_f,lambda_g,lambda_h,lambda_k,lambda_L,lambda_m]
be the seven SI costates for the objective -m(t_f) in kg. Define H=lambda^T F.
Integrate all seven adjoints lambda_dot=-partial(H)/partial(x), holding the
control fixed in this partial derivative. Lambda_L is generally not zero
along the trajectory. Its terminal value, not its initial or entire history,
is zero. Terminal transversality also requires lambda_m(t_f)=-1.

With P=B^T*lambda[:6], the minimizing direction when thrusting is -P/|P|.
The coefficient of rho in the minimized Hamiltonian is
C=-T_max*|P|/m-T_max*lambda_m/v_e. Use full thrust when C<0 and coast
when C>0. At C=0 either is permitted; singular arcs require minimizing H.
There is no prescribed switching time, duty fraction, steering history, or
reference orbital phase. H need not be zero for a fixed-time problem.

Single shooting is not mandatory. Multiple shooting, boundary-value
collocation, direct-to-indirect initialization, and regularization/continuation
are permitted with NumPy/SciPy. The final certificate must satisfy the original
unregularized Hamiltonian conditions to the tolerances below. A converged
regularized solve alone does not certify the minimum-fuel problem.
No astrodynamics or trajectory-optimization packages are permitted.

## Files And Scalars

Retain the four original required filenames. Tier 3 changes to **15 columns**:

```text
tier3_trajectory.npy:
[t,p,f,g,h,k,L,m,lambda_p,lambda_f,lambda_g,lambda_h,lambda_k,lambda_L,lambda_m]
tier3_control.npy:
[t,q_r,q_t,q_n]
```

Arrays must be real, finite, and have identical strictly increasing timestamps
from 0 to 25920000, with 1000 to 2000000 rows. L is unwrapped. Between saved
rows, the control fraction VECTOR is interpolated linearly without subsequent
normalization. Insert sufficiently close distinct times around switches to
represent short continuous transitions. There is no physical slew-rate limit.
Saved states and costates must solve the equations under that interpolated
control, not under a different implicit steering law.

Resolve the output so every time gap is <=3600 s, every longitude increment
is in (0,0.2] rad, and every adjacent control-vector change has norm <=0.05.
These are certification mesh requirements, not prescribed integrator steps;
further refinement may be needed to pass the error estimate below.

The tier3 JSON object must contain:

```text
formulation: "full_mee_j2_minimum_fuel_v2"
method: description of the method actually used
final_sma_km, final_eccentricity, final_inclination_deg
dv_total_m_s, transfer_time_days, final_mass_kg, fuel_consumed_kg
initial_costates: seven SI costates in the array order above
constraint_violation_norm, shooting_converged
hamiltonian_initial, hamiltonian_final
```

The historical name shooting_converged is retained for solver convergence,
including collocation; it must be true but does not replace verification.
Compute a=p/(1-f*f-g*g), e=hypot(f,g), i=2*atan(hypot(h,k)),
delta-v=v_e*log(m_0/m_f), and fuel=m_0-m_f. Report inclination in degrees,
semi-major axis in km, time in days, and Hamiltonians in kg/s.

## Public Acceptance Rules

For dimensionless checks use D=diag(R_E,1,1,1,1,1,m_0), tau=t/t_f,
y=D^-1*x, ell=D*lambda/m_0, G=t_f*D^-1*F, and Hbar=ell^T*G.
The normalized adjoints are ell_prime=-partial(Hbar)/partial(y).

- Initial max normalized state error <=1e-9; time endpoints within 1e-8 s
  initially and 1e-6 s finally; control norm <=1+1e-10.
- Recompute the Euclidean norm of
  [(p_f-42164000)/42164000, f_f,g_f,h_f,k_f,ell_L(tf),ell_m(tf)+1].
  It must be <=1e-4. This implies the previous final-orbit tolerances and also
  checks both transversality conditions. Report this same norm within 1e-7.
- Every interval is checked, not a subset matched to a hidden trajectory.
  RK4 with one step and two half steps integrates the 14 normalized canonical
  variables under the linearly interpolated controls. The extrapolated result
  is Y_two_half+(Y_two_half-Y_one)/15. Sum absolute endpoint defects separately
  per component over all intervals. State component sums must be <=1e-4.
  Costate sums divided by max(1,max_rows(abs(ell_component))) must be <=1e-4.
  Apply the same scaling to the sums of abs((Y_two_half-Y_one)/15); each must
  be <=1e-5, otherwise refine the output mesh. This conservative estimate
  prevents unresolved output from being mistaken for a physical failure.
- Additionally, replay the saved piecewise-linear RTN control from the prescribed
  initial state for the full duration in independent inertial Cartesian
  central-gravity + J2 equations. Use DOP853 with rtol=3e-11,
  atol=3e-11*[R_E,R_E,R_E,sqrt(mu/R_E),sqrt(mu/R_E),sqrt(mu/R_E),m_0],
  and max_step=300 s. Stop integration at every saved control knot so an
  adaptive step never straddles a change in the control interpolation slope.
  Equivalent dimensionless Cartesian coordinates with scalar atol=3e-11 are
  used by the evaluator's SciPy DOP853 engine. The replay must remain outside
  Earth and reach t_f.
  Its recomputed five-element terminal norm (the state part of the boundary
  norm above) must be <=1e-4. At up to 2001 saved rows selected by evenly
  spaced integer indices including endpoints, the maximum position error
  divided by replay radius and velocity error divided by replay speed must
  each be <=1e-3; maximum mass error divided by m_0 must be <=1e-4.
  Local defects alone do not bound accumulated orbital-phase timing error.
- The recomputed Hbar range over the trajectory must be
  <=1e-4+0.01*max(abs(Hbar)). At each point compute
  gap=Hbar(y,ell,q)-min_{norm(v)<=1} Hbar(y,ell,v), including coast.
  Its maximum must be <=0.002, and its integral over tau must be <=1e-4.
  Evaluate at every row and every cubic-Hermite canonical midpoint, using
  endpoint canonical rates and midpoint linear control. Use Simpson integration
  for the gap integral.
  Report both endpoint SI Hamiltonians within 1e-10 kg/s.
- Scalars must match the saved terminal state within 0.001 km for SMA,
  1e-8 eccentricity, 1e-6 deg inclination, 1e-5 kg mass/fuel, and 0.001 m/s
  delta-v. Report exactly 300 days within 1e-10 days. Initial costates must
  match the array within 1e-8 in the normalized costate coordinates.
- Fuel must be no more than 1 kg above the independently computed, verified
  full-dynamics reference incumbent. This is a numerical quality benchmark,
  not a claim of a rigorous global optimum. Better feasible extremals are
  accepted. No absolute Edelbaum bound or reference control/history matching
  is used. First-order Pontryagin conditions alone are not a global proof.

The original orbit-averaged gold is invalid for this version. The evaluator
refuses a legacy reference rather than silently using its fuel value.
