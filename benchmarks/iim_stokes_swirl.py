"""RM-IIM stage 2b: a Stokes solve with a tangential traction on a circle (exact solution).

Swirl u = v(r) e_theta about C, periodic unit box: rigid rotation v = a r inside r < R, and
outside a C^2 profile decaying to rest at r = R2 < 0.5 with a different slope at R. The velocity
is continuous, its radial derivative jumps at R, the pressure is constant. The exact Stokes
force is the smooth body force f = -mu (v'' + v'/r - v/r^2) e_theta on each side (zero inside)
plus a tangential line force f_G = -mu [v'] e_theta on the circle.

Discrete steady Stokes on the periodic MAC grid: each velocity component from the FFT Poisson
problem  mu Lap u = -f, then the exact projection.
  smeared : f_G spread with the smoothed delta of half-width 2h (the current one-fluid approach)
  iim     : no line force; the jumps of du/dx_i (and optionally d2u/dx_i^2) across the circle
            enter as immersed-interface corrections of the Laplacian (pyRMT.iim)

Usage: python benchmarks/iim_stokes_swirl.py
"""
import os, sys
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.mac import poisson_eigs_periodic, solve_poisson_periodic, project_per
from pyRMT.iim import laplacian_jump_correction, divergence_jump_correction_per, project_per_with_jumps

MU, A, R, R2, K, C = 1.0, 1.0, 0.2, 0.42, 4.0, (0.5, 0.5)
CORRECT_DIV = "--plain-div" not in sys.argv       # jump-corrected divergence in the projection

# --- exact solution, per side, as sympy expressions in Cartesian coordinates ----------------
x, y = sp.symbols("x y", real=True)
xp, yp = x - C[0], y - C[1]; r = sp.sqrt(xp ** 2 + yp ** 2)
v_in = A * r
v_out = A * R * ((R2 - r) / (R2 - R)) ** 3 * (1 + K * (r - R))
SIDES = {}
for name, vr in (("in", v_in), ("out", v_out)):
    ux = -vr * yp / r; uy = vr * xp / r
    fx = -MU * (sp.diff(ux, x, 2) + sp.diff(ux, y, 2)); fy = -MU * (sp.diff(uy, x, 2) + sp.diff(uy, y, 2))
    d = {"ux": ux, "uy": uy, "fx": fx, "fy": fy}
    for c in ("ux", "uy"):
        d[c + "_x"] = sp.diff(d[c], x); d[c + "_y"] = sp.diff(d[c], y)
        d[c + "_xx"] = sp.diff(d[c], x, 2); d[c + "_yy"] = sp.diff(d[c], y, 2)
    SIDES[name] = {k: sp.lambdify((x, y), e, "numpy") for k, e in d.items()}
dvjump = float(sp.diff(v_out, sp.Symbol("rr")).subs({}) if False else
               (sp.diff(v_out.subs(r, sp.Symbol("s")), sp.Symbol("s")) - A).subs(sp.Symbol("s"), R))


def ev(key, X, Y, phi):
    """Evaluate on the side each point lies on; zero beyond R2 (fluid at rest)."""
    rr = np.hypot(X - C[0], Y - C[1]) + 1e-300
    out = np.where(phi <= 0, SIDES["in"][key](X, Y) * np.ones_like(X), SIDES["out"][key](X, Y) * np.ones_like(X))
    return np.where(rr < R2, out, 0.0)


def jump(key, X, Y):
    return SIDES["out"][key](X, Y) * np.ones_like(X) - SIDES["in"][key](X, Y) * np.ones_like(X)


def solve(N, method, second=True):
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; xf = np.arange(N) * h
    XU, YU = np.meshgrid(xf, xc); XV, YV = np.meshgrid(xc, xf)          # u: x-faces, v: y-faces
    pu = np.hypot(XU - C[0], YU - C[1]) - R; pv = np.hypot(XV - C[0], YV - C[1]) - R
    eig = poisson_eigs_periodic(N, N, h, h)
    comps = []
    for c, X, Y, ph in (("ux", XU, YU, pu), ("uy", XV, YV, pv)):
        f = ev("f" + c[1], X, Y, ph)
        if method == "smeared":                    # line force spread over |phi| < 2h
            eps = 2 * h; dl = np.where(np.abs(ph) < eps, (1 + np.cos(np.pi * ph / eps)) / (2 * eps), 0.0)
            th = np.arctan2(Y - C[1], X - C[0]); et = -np.sin(th) if c == "ux" else np.cos(th)
            f = f + (-MU * dvjump) * et * dl
            rhs = -f / MU
        else:
            Cc = laplacian_jump_correction(ph, jump(c + "_x", X, Y), jump(c + "_y", X, Y), h, h,
                                           gxx=jump(c + "_xx", X, Y) if second else None,
                                           gyy=jump(c + "_yy", X, Y) if second else None)
            rhs = -f / MU - Cc
        comps.append(solve_poisson_periodic(rhs - rhs.mean(), eig))
    if method == "smeared" or not CORRECT_DIV:
        u, v, _ = project_per(comps[0], comps[1], h, h, 1.0, 1.0, eig)
    else:                                          # divergence with the same jump corrections
        dc = divergence_jump_correction_per(pu, jump("ux_x", XU, YU), pv, jump("uy_y", XV, YV), h, h)
        u, v, _ = project_per_with_jumps(comps[0], comps[1], h, h, eig, dc)
    eu = u - ev("ux", XU, YU, pu); evv = v - ev("uy", XV, YV, pv)
    e = np.concatenate([eu.ravel(), evv.ravel()])
    return np.abs(e).max(), np.sqrt(np.mean(e ** 2))


def main():
    print(f"tangential traction jump mu[v'] = {MU * dvjump:.3f}")
    for lab, m, sec in (("smeared (current)", "smeared", False), ("IIM, 1st-derivative jumps", "iim", False),
                        ("IIM, 1st+2nd-derivative jumps", "iim", True)):
        print(lab); prev = None
        for N in (32, 64, 128, 256, 512):
            e = solve(N, m, sec)
            o = "" if prev is None else "  orders max %.2f  L2 %.2f" % (np.log2(prev[0] / e[0]), np.log2(prev[1] / e[1]))
            print(f"  N={N:4d}  max {e[0]:.2e}  L2 {e[1]:.2e}{o}"); prev = e


if __name__ == "__main__":
    main()
