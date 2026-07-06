"""
LTE bubble-wall stability: reduced 4x4 dispersion determinant.

  Unknowns (columns):   A_ac   -- core acoustic amplitude (f ~ xi^l branch, integrated)
                        A_vort -- core vorticity amplitude (EXACT closed form:
                                  f=0, g=xi^alpha, V=(alpha+2)/(l(l+1)) xi^alpha;
                                  regular at origin for Re(alpha) > 0)
                        B      -- shell amplitude: 1-parameter family fixed at the
                                  shock by the 3 perturbed RH conditions (TE,RE,PE),
                                  integrated inward to the wall (renormalized)
                        k_w    -- wall ripple amplitude
  Conditions (rows):    wall TE, wall RE, wall PE, wall ENT.

WARNING:: D has a spurious real zero at alpha = l-1 (indicial resonance:
acoustic and vorticity core columns become parallel there).
"""
import numpy as np
from scipy.integrate import solve_ivp

def _core_acoustic(bg, L, alpha, xi_min=1e-3, rtol=1e-10, atol=1e-13):
    """
    regular acoustic core solution at the wall, seeded with the
    acoustic  Frobenius branch  (s = alpha-(l-1)):
        f = s*xi^l,  g = (-l/w)*xi^(l-1),  V = (-1/w)*xi^(l-1)
    Finite at the resonance s=0, where the acoustic column becomes
    parallel to the vorticity column (spurious zero of D at alpha=l-1;

    ( recall there is also the vortical mode, but it is closed form so doesn't need to be integrated up)

    """
    w = float(bg['wm'])
    xi_w = bg['xi_w']
    l = float(L)
    s = alpha - (l - 1.0)

    # RHS uses the interior-at-rest core state (w=wm, v=0, g=1)
    def rhs(xi, Y):
        y = Y[:3] + 1j*Y[3:]
        f, g, V = y
        Vp = (alpha/xi)*V + f/(xi**2*w)
        gp = (3*alpha*f + 3*xi*w*alpha*g + (2*w/xi)*g - (w/xi)*l*(l+1)*V) / (w*(3*xi**2 - 1.0))
        fp = xi*w*gp - w*alpha*g
        d = np.array([fp, gp, Vp])
        return np.concatenate([d.real, d.imag])

    # seed the Frobenius branch at xi_min (s-SCALED, so finite at s=0); these give the initial
    # conditions for the ODE integration from xi_min to xi_w
    y0 = np.array([s*xi_min**l, (-l/w)*xi_min**(l-1), (-1.0/w)*xi_min**(l-1)], dtype=complex)
    Y0 = np.concatenate([y0.real, y0.imag])
    sol = solve_ivp(rhs, (xi_min, xi_w), Y0, t_eval=[xi_w], rtol=rtol, atol=atol, method='RK45')
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:3, -1] + 1j*sol.y[3:, -1]

EPS_BG = 1e-7   # one-sided offset for evaluating shell splines at surfaces#


def _shell_rhs(xi, y, bg, L, alpha):
    """Shell bulk ODE (f',g',V') from the VALIDATED equations
    (xi OUTSIDE the time-derivative; translation-mode residuals ~1e-9)."""
    f, g, V = y
    S = _state(bg, xi)
    v, w, gm = S['v'], S['w'], S['g']
    vp, wp, gp = S['vp'], S['wp'], S['gp']
    l = float(L)
    T1 = 4*gm**2 - 1;            T1p = 8*gm*gp
    T2 = 2*w*gm**4*v;            T2p = 2*(wp*gm**4*v + 4*w*gm**3*gp*v + w*gm**4*vp)
    T3 = 4*gm**2*v;              T3p = 8*gm*gp*v + 4*gm**2*vp
    T4 = gm**2*w*(2*gm**2*v**2+1)
    T4p = (2*gm*gp*w + gm**2*wp)*(2*gm**2*v**2+1) + gm**2*w*(4*gm*gp*v**2 + 4*gm**2*v*vp)
    R3 = 4*gm**2*v**2 + 1;       R3p = 8*gm*gp*v**2 + 8*gm**2*v*vp
    R4 = T2;                     R4p = T2p
    T5 = w*gm**2;                R7 = w*gm**2*v
    F00 = T1*f + T2*g; F0r = T3*f + T4*g; Frr = R3*f + R4*g
    AT = T3 - xi*T1;  BT = T4 - xi*T2
    AR = R3 - xi*T3;  BR = R4 - xi*T4

    # timelike and radial bulk equations
    RT = (-alpha*F00 - (T3p - xi*T1p)*f - (T4p - xi*T2p)*g
          - (2.0/xi)*F0r + (l*(l+1)/xi)*T5*V)
    RR = (-alpha*F0r - (R3p - xi*T3p)*f - (R4p - xi*T4p)*g
          - (2.0/xi)*Frr + 2.0*f/xi + (l*(l+1)/xi)*R7*V)
    det = AT*BR - BT*AR
    fp = (BR*RT - BT*RR)/det
    gp_ = (AT*RR - AR*RT)/det
    wg2 = w*gm**2; wg2p = wp*gm**2 + 2*w*gm*gp
    Pa = wg2
    Pb = (wg2p*v + wg2*vp) - xi*wg2p + 3.0*wg2*v/xi
    Pc = wg2*(v - xi)

    # polar bulk equation
    Vp = -((alpha*Pa + Pb)*V + f/xi)/Pc

    # return the derivatives of the shell variables (f,g,V) at xi
    return [fp, gp_, Vp]



# ----------------------------------------------------------------------
# background state helpers
# ----------------------------------------------------------------------
def _state(bg, x):
    """Shell-side background state at xi=x (scalar), one-sided into the shell."""
    xa = np.array([x])
    v = float(bg['v0'](xa)[0]);  w = float(bg['w0'](xa)[0])
    vp = float(bg['v0p'](xa)[0]); wp = float(bg['w0p'](xa)[0])
    p = float(bg['p0'](xa)[0]);  s = float(bg['s0'](xa)[0])
    T = float(bg['T0'](xa)[0])
    g = 1.0/np.sqrt(1.0 - v*v);  gp = g**3 * v * vp
    # s' and p' by finite difference on the provided splines; ONE-SIDED near the
    # wall and the shock, where bg's piecewise T0/p0 definitions switch branch
    h = 1e-6
    def _d(F):
        if x - bg['xi_w'] < 3*h:      # forward, into the shell
            return float((-3*F(np.array([x]))[0] + 4*F(np.array([x+h]))[0]
                          - F(np.array([x+2*h]))[0])/(2*h))
        if bg['xi_sh'] - x < 3*h:     # backward, into the shell
            return float((3*F(np.array([x]))[0] - 4*F(np.array([x-h]))[0]
                          + F(np.array([x-2*h]))[0])/(2*h))
        return float((F(np.array([x+h]))[0] - F(np.array([x-h]))[0])/(2*h))
    sp = _d(bg['s0']); pp = _d(bg['p0'])
    return dict(v=v, w=w, p=p, s=s, T=T, g=g, vp=vp, wp=wp, gp=gp, sp=sp, pp=pp)

def _state_core(bg):
    """Broken-phase at-rest core state at the wall (all xi-derivatives zero)."""
    wm = float(bg['wm']); Tm = float(bg['Tm'])
    return dict(v=0.0, w=wm, p=wm/4.0, s=wm/Tm, T=Tm,
                g=1.0, vp=0.0, wp=0.0, gp=0.0, sp=0.0, pp=0.0)

def _state_ahead_of_shock(bg):
    """
    Undisturbed state ahead of the shock, from RH using the shell side
    (avoids relying on splines beyond the shock).  eps recovered from p0.
    Basically we get BAG EOS, untouched by perturbations
    """
    S = _state(bg, bg['xi_sh'] - EPS_BG)
    xi = bg['xi_sh']
    eps = S['w']/4.0 - S['p']                       # p = w/4 - eps (symmetric phase)
    pN = S['p'] + S['w']*S['g']**2*S['v']*(S['v'] - xi)   # momentum-flux continuity
    wN = 4.0*(pN + eps)
    return dict(v=0.0, w=wN, p=pN, s=None, T=None,
                g=1.0, vp=0.0, wp=0.0, gp=0.0, sp=0.0, pp=0.0), S


# ----------------------------------------------------------------------
# junction template
#   row(K):  sum_fields  kap_+.U_+ - kap_-.U_-   +   k * (ck_+ - ck_-)  = 0
#   kap: coefficients of (f, g, V) in  -xi*dT^{0nu} + dT^{r nu}   (dw=4f, dp=f)
#   ck = alpha*mu_a + mu_0 :
#     mu_a = -X0        (X0 = time part of the zeroth-order flux for condition K)
#     mu_0 = -X0 + [Xr' - xi*X0']   (background-derivative Taylor terms)
# ----------------------------------------------------------------------
def _kap(S, K, xi):
    v, w, g, s = S['v'], S['w'], S['g'], S['s']
    if K == 'TE':
        return np.array([-xi*(4*g*g - 1) + 4*g*g*v,
                         -xi*(2*w*g**4*v) + g*g*w*(2*g*g*v*v + 1),
                         0.0])
    if K == 'RE':
        return np.array([-xi*(4*g*g*v) + (4*g*g*v*v + 1),
                         -xi*(g*g*w*(2*g*g*v*v + 1)) + 2*w*g**4*v,
                         0.0])
    if K == 'PE':
        return np.array([0.0, 0.0, (v - xi)*w*g*g])
    if K == 'ENT':
        return np.array([(3*g*s/w)*(v - xi),
                         s*g**3*(1 - xi*v),
                         0.0])
    raise ValueError(K)

def _mu(S, K, xi):
    """(mu_a, mu_0) for the k-coupling of condition K on one side."""
    v, w, p, s, g = S['v'], S['w'], S['p'], S['s'], S['g']
    vp, wp, gp, sp, pp = S['vp'], S['wp'], S['gp'], S['sp'], S['pp']
    wg2 = w*g*g
    dwg2 = wp*g*g + 2*w*g*gp
    T00 = wg2 - p;         dT00 = dwg2 - pp
    T0r = wg2*v;           dT0r = dwg2*v + wg2*vp
    Trr = wg2*v*v + p;     dTrr = dwg2*v*v + 2*wg2*v*vp + pp
    if K == 'TE':
        return -T00, -T00 + (dT0r - xi*dT00)
    if K == 'RE':
        return -T0r, -T0r + (dTrr - xi*dT0r)
    if K == 'PE':
        return 0.0, -p/xi
    if K == 'ENT':
        J0 = g*s;              dJ0 = gp*s + g*sp
        Jr = g*v*s;            dJr = (gp*v + g*vp)*s + g*v*sp
        return -J0, -J0 + (dJr - xi*dJ0)
    raise ValueError(K)


# ----------------------------------------------------------------------
# shock: admissible 1-parameter family (null vector of the 3x4 system)
# remember: 3x4 since there are 3 unknowns (f, g, V) on the shell side and 1 unknown (k_sh) for the shock ripple
#           but only 3 conditions (TE, RE, PE) at the shock, so we have a 3x4 system to solve for the null vector
# unknowns (f, g, V, k_sh) on the shell side; ahead unperturbed constant state
# ----------------------------------------------------------------------
def shock_null_vector(bg, alpha):
    xi = bg['xi_sh']
    ahead, S = _state_ahead_of_shock(bg)
    rows = []
    for K in ('TE', 'RE', 'PE'):
        kapm = _kap(S, K, xi)
        ma_m, m0_m = _mu(S, K, xi)
        ma_p, m0_p = _mu(ahead, K, xi) if K != 'ENT' else (0., 0.)
        ck = alpha*(ma_p - ma_m) + (m0_p - m0_m)
        rows.append([-kapm[0], -kapm[1], -kapm[2], ck])
    A = np.array(rows, dtype=complex)
    # smallest right-singular vector = admissible direction
    _, sv, Vh = np.linalg.svd(A)
    null = Vh[-1].conj()
    return null, sv    # null = (f, g, V, k_sh) at the shock; sv for rank diagnostics


# ----------------------------------------------------------------------
# shell: integrate (f,g,V) inward from the core ---> shock, with renormalization
# ----------------------------------------------------------------------
def shell_arrival(bg, L, alpha, y_shock, n_seg=8, rtol=1e-9, atol=1e-12):
    """
    Propagate the complex 3-vector from xi_sh to xi_w.  Only the arrival
    DIRECTION matters; segments are rescaled by 1/max|y| (real, positive).
    """
    xi_w, xi_sh = bg['xi_w'] + EPS_BG, bg['xi_sh'] - EPS_BG
    nodes = np.linspace(xi_sh, xi_w, n_seg + 1)
    y = np.asarray(y_shock, dtype=complex)
    y = y / max(np.max(np.abs(y)), 1e-300)
    def rrhs(t, Y):
        yc = Y[:3] + 1j*Y[3:]
        d = np.asarray(_shell_rhs(t, yc, bg, L, alpha), dtype=complex)
        return np.concatenate([d.real, d.imag])
    for a, b in zip(nodes[:-1], nodes[1:]):
        Y0 = np.concatenate([y.real, y.imag])
        sol = solve_ivp(rrhs, (a, b), Y0, method='RK45', rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'shell integration failed on [{a},{b}]: {sol.message}')
        y = sol.y[:3, -1] + 1j*sol.y[3:, -1]
        y = y / max(np.max(np.abs(y)), 1e-300)
    return y     # (f, g, V) at the wall, shell side, unit-normalized


# ----------------------------------------------------------------------
# core columns
# ----------------------------------------------------------------------
def core_columns(bg, L, alpha):
    ac = _core_acoustic(bg, L, alpha)                 # acoustic (integrated)
    ac = ac / max(np.max(np.abs(ac)), 1e-300)
    xw = bg['xi_w']

    # vortical mode is closed form, don't need to integrate
    vort = np.array([0.0,
                     xw**alpha,
                     (alpha + 2.0)/(L*(L + 1.0)) * xw**alpha], dtype=complex)
    vort = vort / max(np.max(np.abs(vort)), 1e-300)
    # return the core columns (acoustic, vortical) at the wall, unit-normalized

    # structure: ac = (f, g, V) for the acoustic branch; vort = (f, g, V) for the vortical branch
    return ac, vort



# ----------------------------------------------------------------------
# the 4x4 wall system and the dispersion determinant
# ----------------------------------------------------------------------
def wall_matrix(bg, L, alpha, n_seg=8, rtol=1e-9, sub = False):
    # 4 unknowns: core acoustic amplitude, core vorticity amplitude,
    # shell amplitude, wall ripple amplitude

    xi = bg['xi_w']

    #Sp = plus state (just outside the wall, shell side);
    # Sm = minus state (just inside the wall, core side)
    Sp = _state(bg, xi + EPS_BG)      # shell (+) side of the wall
    Sm = _state_core(bg)              # core  (-) side

    # get the core columns (acoustic, vortical) = (f_ac, g_ac, V_ac), (f_vort, g_vort, V_vort) at the wall
    ac, vort = core_columns(bg, L, alpha)

    # get the null vector of the shock system (f, g, V, k_sh)
    # and then integrate inward to the wall (f, g, V) on the shell side
    null, sv = shock_null_vector(bg, alpha)
    sh = shell_arrival(bg, L, alpha, null[:3], n_seg=n_seg, rtol=rtol)

    def row(K):
        kp, km = _kap(Sp, K, xi), _kap(Sm, K, xi)
        ma_p, m0_p = _mu(Sp, K, xi)
        ma_m, m0_m = _mu(Sm, K, xi)
        ck = alpha*(ma_p - ma_m) + (m0_p - m0_m)
        return np.array([-(km @ ac), -(km @ vort), (kp @ sh), ck], dtype=complex)

    rTE, rRE, rPE, rENT = row('TE'), row('RE'), row('PE'), row('ENT')
    if sub:
        lam_p = Sp['g']*Sp['T']
        lam_m = Sm['g']*Sm['T']
        lam = 0.5*(lam_p + lam_m)                           # LTE: these agree; checked in selftest
        # given the ENT and TE rows are near-parallel, replace the TE row with a linear combination
        # of TE and ENT that is orthogonal to ENT
        M = np.vstack([rTE - lam*rENT, rRE, rPE, rENT])
    else:
        M = np.vstack([rTE, rRE, rPE, rENT])
    info = dict(shock_sv=sv,
                shell_arrival=sh, core_ac=ac, core_vort=vort, shock_null=null)
    return M, info

def dispersion(bg, L, alpha, **kw):
    """
    D(alpha): zeros are the eigenvalues.  Rows/columns are rescaled by
    positive reals (does not move zeros, does not affect winding numbers).
    the scaling is done to avoid overflow/underflow in the determinant .
    """
    M, info = wall_matrix(bg, L, alpha, **kw)

    # rs = max row abs, cs = max col abs; divide each row by rs, each col by cs
    rs = np.max(np.abs(M), axis=1); rs[rs == 0] = 1.0
    M = M / rs[:, None]
    cs = np.max(np.abs(M), axis=0); cs[cs == 0] = 1.0
    M = M / cs[None, :]
    return np.linalg.det(M), info


# ----------------------------------------------------------------------
# root-finding utilities
# ----------------------------------------------------------------------
def scan_real(bg, L, alphas, **kw):
    return np.array([dispersion(bg, L, a + 0j, **kw)[0] for a in alphas])

def winding(bg, L, z0, z1, n=200, **kw):
    """number of zeros of D inside the rectangle with corners z0, z1
    (cauchy's argument principle; D assumed nonzero on the contour).
    rememeber cauchy's arg principle: if $D$ has $N_z$ zeros and $N_p$ poles
      inside the contour, the total phase accumulated is exactly $2\pi(N_z - N_p)$
    """

    # build the rectangle contour in the complex plane, clockwise from z0 to z1
    x0, x1 = sorted((z0.real, z1.real)); y0, y1 = sorted((z0.imag, z1.imag))
    pts = np.concatenate([
        x0 + np.linspace(0, 1, n)*(x1 - x0) + 1j*y0,
        x1 + 1j*(y0 + np.linspace(0, 1, n)*(y1 - y0)),
        x1 - np.linspace(0, 1, n)*(x1 - x0) + 1j*y1,
        x0 + 1j*(y1 - np.linspace(0, 1, n)*(y1 - y0)),
    ])
    # evaluate D (the matching determinant) at each point along the contour
    D = np.array([dispersion(bg, L, z, **kw)[0] for z in pts])

    # compute the phase change along the contour
    """
    D     = [D0,  D1,  D2,  D3, ..., D_{n-1}]
    D[1:] = [D1,  D2,  D3,  D4, ..., D_{n-1}]
    D[:-1]= [D0,  D1,  D2,  D3, ..., D_{n-2}]
    so D[1:]/D[:-1] = [D1/D0, D2/D1, D3/D2, ..., D_{n-1}/D_{n-2}]
    i.e. each gives the increment in phase from one point to the next along the contour.
    The sum of these increments gives the total phase change around the contour,
    which is related to the number of zeros inside by cauchy's arg principle.
    """
    dphi = np.angle(D[1:]/D[:-1]) # this is an array of phase increments along the contour
    if np.max(np.abs(dphi)) > 2.5:
        print('winding: phase step >2.5 rad; increase n for reliability')

    #  sum the phase increments and divide by 2pi to get the winding number (number of zeros inside)
    return int(np.round(np.sum(dphi)/(2*np.pi)))

def refine(bg, L, alpha0, tol=1e-10, maxit=60, h0=1e-3, **kw):
    """Complex secant iteration on D(alpha)."""
    z0, z1 = alpha0, alpha0 + h0
    f0 = dispersion(bg, L, z0, **kw)[0]
    f1 = dispersion(bg, L, z1, **kw)[0]
    for _ in range(maxit):
        if f1 == f0:
            break
        z2 = z1 - f1*(z1 - z0)/(f1 - f0)
        z0, f0, z1 = z1, f1, z2
        f1 = dispersion(bg, L, z1, **kw)[0]
        if abs(z1 - z0) < tol*(1 + abs(z1)):
            break
    return z1, abs(f1)


# ----------------------------------------------------------------------
# diagnostics / validation
# ----------------------------------------------------------------------
def background_checks(bg):
    out = {}
    xi = bg['xi_sh']
    ahead, S = _state_ahead_of_shock(bg)
    # energy flux continuity at the shock ( momentum was used to build wN)
    Em = -xi*(S['w']*S['g']**2 - S['p']) + S['w']*S['g']**2*S['v']
    Ep = -xi*(ahead['w'] - ahead['p'])
    out['shock_Eflux'] = abs(Ep - Em)/max(abs(Em), 1e-30)
    xi = bg['xi_w']
    Sp = _state(bg, xi + EPS_BG); Sm = _state_core(bg)
    for name, Xp, Xm in [
        ('wall_Eflux', -xi*(Sp['w']*Sp['g']**2 - Sp['p']) + Sp['w']*Sp['g']**2*Sp['v'],
                       -xi*(Sm['w'] - Sm['p'])),
        ('wall_Mflux', -xi*Sp['w']*Sp['g']**2*Sp['v'] + Sp['w']*Sp['g']**2*Sp['v']**2 + Sp['p'],
                       Sm['p']),
        ('wall_Sflux', Sp['g']*Sp['s']*(Sp['v'] - xi), Sm['s']*(0.0 - xi)),
        ('wall_gammaT', Sp['g']*Sp['T'], Sm['T']),
    ]:
        out[name] = abs(Xp - Xm)/max(abs(Xm), 1e-30)
    return out

def selftest(bg, alpha_probe=(0.3 + 0.2j), sub = False):
    print('--- background junction residuals (want << 1) ---')
    for k, r in background_checks(bg).items():
        print(f'   {k:14s} {r:.3e}')
    print('--- shock system rank at alpha_probe (want sv[2] >> sv[3]-ish scale) ---')
    null, sv = shock_null_vector(bg, alpha_probe)
    print('   singular values:', np.array2string(sv, precision=3))
    print('   null (f,g,V,k_sh):', np.array2string(null, precision=3))
    print('--- LTE invariant [gamma_rel * T] at the wall (wall-frame gamma!) ---')
    xiw = bg['xi_w']
    Sp = _state(bg, xiw + EPS_BG); Sm = _state_core(bg)
    mu_p = (xiw - Sp['v'])/(1.0 - xiw*Sp['v'])
    gr_p = 1/np.sqrt(1-mu_p**2); gr_m = 1/np.sqrt(1-xiw**2)
    print(f"   plus={gr_p*Sp['T']:.8f}  minus={gr_m*Sm['T']:.8f}   (LTE => equal;"
          f" NB lab-frame gamma*T is NOT continuous)")
    M, info = wall_matrix(bg, 2, alpha_probe, sub = sub)
    print('--- l=1 translation mode: D(-1) should be ~0 ---')
    Dm1, _ = dispersion(bg, 1, -1.0 + 0j, sub = sub)
    ring = [dispersion(bg, 1, -1.0 + 0.05*np.exp(2j*np.pi*t), sub = sub)[0]
            for t in np.linspace(0, 1, 8, endpoint=False)]
    med = np.median(np.abs(ring))
    print(f'   |D(-1)| = {abs(Dm1):.3e}   median|D| on r=0.05 ring = {med:.3e}   '
          f'ratio = {abs(Dm1)/med:.3e}')
    w = winding(bg, 1, (-1.15 - 0.1j), (-0.85 + 0.1j), n=120, sub = sub)
    print(f'   winding number around alpha=-1 (l=1): {w}   (expect 1)')
    return dict(D_minus1=Dm1, ring_median=med, winding=w)

def shell_conditioning(bg, n=200):
    """Profile of the (f',g')-elimination determinant across the shell.
        near-zero anywhere flags a near-sonic layer."""
    xs = np.linspace(bg['xi_w'] + EPS_BG, bg['xi_sh'] - EPS_BG, n)
    out = []
    for x in xs:
        S = _state(bg, x)
        v, w, g = S['v'], S['w'], S['g']
        T1 = 4*g*g - 1; T2 = 2*w*g**4*v; T3 = 4*g*g*v; T4 = g*g*w*(2*g*g*v*v + 1)
        R1 = T3; R2 = T4; R3 = 4*g*g*v*v + 1; R4 = T2
        AT = T3 - x*T1; BT = T4 - x*T2; AR = R3 - x*R1; BR = R4 - x*R2
        out.append(AT*BR - BT*AR)
    return xs, np.array(out)



# ----------------------------------------------------------------------
# eigenvector reconstruction and inspection
# ----------------------------------------------------------------------
def eigenvector(bg, L, alpha, n_core=200, n_shell=200, rtol=1e-9, sub = False):
    """Reconstruct the eigenfunction at a root alpha of D.

    Returns dict with: amplitudes (A_ac, A_vort, B, k_w, k_sh), smallest
    singular value of the wall matrix (should be ~0 at a root), dense radial
    profiles xi_core, U_core (n,3), xi_shell, U_shell (n,3), and the wall-row
    residuals of the reconstructed vector (consistency check).
    Fields are (f, g, V) amplitudes; normalization: max |amplitude| = 1.
    """
    M, info = wall_matrix(bg, L, alpha, rtol=rtol, sub = sub)
    _, sv, Vh = np.linalg.svd(M)
    amp = Vh[-1].conj()                      # (A_ac, A_vort, B, k_w)
    amp = amp / amp[np.argmax(np.abs(amp))]
    A_ac, A_vort, B, k_w = amp
    resid = np.abs(M @ amp)

    # core profiles
    xi_c = np.linspace(1e-3, bg['xi_w'], n_core)
    ac_prof = _core_acoustic_dense(bg, L, alpha, xi_c, rtol=rtol)
    ac_prof = ac_prof / max(np.max(np.abs(ac_prof[-1])), 1e-300)   # match core_columns norm
    vort_prof = np.stack([np.zeros_like(xi_c, dtype=complex),
                          xi_c**alpha,
                          (alpha+2.0)/(L*(L+1.0)) * xi_c**alpha], axis=1)
    vort_prof = vort_prof / max(np.max(np.abs(vort_prof[-1])), 1e-300)
    U_core = A_ac*ac_prof + A_vort*vort_prof

    # shell profile: integrate from the shock null vector WITHOUT renormalizing
    null, _ = shock_null_vector(bg, alpha)
    xi_s = np.linspace(bg['xi_sh'] - EPS_BG, bg['xi_w'] + EPS_BG, n_shell)
    y = null[:3].astype(complex)
    def rrhs(t, Y):
        yc = Y[:3] + 1j*Y[3:]
        d = np.asarray(_shell_rhs(t, yc, bg, L, alpha), dtype=complex)
        return np.concatenate([d.real, d.imag])
    sol = solve_ivp(rrhs, (xi_s[0], xi_s[-1]),
                    np.concatenate([y.real, y.imag]),
                    t_eval=xi_s, rtol=rtol, atol=1e-13)
    traj = (sol.y[:3] + 1j*sol.y[3:]).T                       # (n,3)
    # rescale so its wall value matches the (unit-normalized) column used in M
    wall_val = traj[-1]
    scale = 1.0/max(np.max(np.abs(wall_val)), 1e-300)
    U_shell = B * scale * traj[::-1]                          # ascending xi
    k_sh = B * scale * null[3]
    return dict(alpha=alpha, A_ac=A_ac, A_vort=A_vort, B=B, k_w=k_w, k_sh=k_sh,
                sv_min=sv[-1], sv=sv, wall_residual=resid,
                xi_core=xi_c, U_core=U_core,
                xi_shell=xi_s[::-1], U_shell=U_shell)

def _core_acoustic_dense(bg, L, alpha, xi_eval, xi_min=1e-3, rtol=1e-10, atol=1e-13):
    w = float(bg['wm']); l = float(L); s = alpha - (l - 1.0)
    def rhs(xi, Y):
        y = Y[:3] + 1j*Y[3:]
        f, g, V = y
        Vp = (alpha/xi)*V + f/(xi**2*w)
        gp = (3*alpha*f + 3*xi*w*alpha*g + (2*w/xi)*g - (w/xi)*l*(l+1)*V)/(w*(3*xi**2-1.0))
        fp = xi*w*gp - w*alpha*g
        d = np.array([fp, gp, Vp])
        return np.concatenate([d.real, d.imag])
    y0 = np.array([s*xi_min**l, (-l/w)*xi_min**(l-1), (-1.0/w)*xi_min**(l-1)], dtype=complex)
    sol = solve_ivp(rhs, (xi_min, xi_eval[-1]), np.concatenate([y0.real, y0.imag]),
                    t_eval=np.clip(xi_eval, xi_min, None), rtol=rtol, atol=atol)
    return (sol.y[:3] + 1j*sol.y[3:]).T

def dmap(bg, L, re_range, im_range, nr=30, ni=20, **kw):
    """log10|D| heat map over a complex-alpha rectangle (eigenvalues = dark
    spots; the analog of the QZ spectrum scatter, but artifact-free)."""
    res = np.linspace(*re_range, nr); ims = np.linspace(*im_range, ni)
    Z = np.zeros((ni, nr))
    for i, b in enumerate(ims):
        for j, a in enumerate(res):
            Z[i, j] = np.log10(abs(dispersion(bg, L, a + 1j*b, **kw)[0]) + 1e-300)
    return res, ims, Z
