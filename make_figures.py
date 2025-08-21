#!/usr/bin/env python3
# make_figures.py — Reproduce all figures for BESEVIC (2025)
# Requirements: numpy, matplotlib, (optional) pandas for tables

import numpy as np
import matplotlib.pyplot as plt

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------- 1) Spheres: Brown–York convergence ----------
def fig_sphere_convergence(M=1.0):
    R_vals = np.linspace(3.1, 200, 400)
    M_by = R_vals * (1 - np.sqrt(1 - 2*M/R_vals))
    rel_err = np.abs(M_by - M)/M
    plt.figure(figsize=(6.2, 4.2))
    plt.loglog(R_vals, rel_err, linestyle='-')
    plt.xlabel("Rayon $R$")
    plt.ylabel("Erreur relative $|E_{\\rm BY}(R)-M|/M$")
    plt.title("Convergence Brown--York \\to ADM (Schwarzschild, $M=1$)")
    plt.grid(True, which="both")
    savefig("fig_error_vs_radius_improved.pdf")

# ---------- 2) Ellipsoids: smoothed proxy curve ----------
def fig_ellipsoids_smooth(a_base=15.0, M=1.0):
    qs = np.linspace(0.7, 1.3, 61)
    def rel_err_proxy(q, beta=0.0):
        r_eff = (a_base*a_base*(q*a_base))**(1.0/3.0)
        M_est = r_eff * (1 - np.sqrt(1 - 2*M/r_eff))
        anis = beta * 0.02 * (q - 1.0)**2
        return np.abs((M_est + anis) - M)
    plt.figure(figsize=(6.2, 4.2))
    for beta in [0.0, 0.25, 0.5]:
        errs = np.array([rel_err_proxy(q, beta=beta) for q in qs])
        plt.plot(qs, errs, label=f"$\\beta={beta}$")
    plt.xlabel("Rapport d'aspect $b/a$")
    plt.ylabel("Erreur absolue $|M_{\\rm est}-1|$")
    plt.title("Ellipsoïdes : stabilité vs forme (modèle lissé)")
    plt.grid(True)
    plt.legend()
    savefig("fig_relerr_vs_aspect_improved.pdf")

# ---------- 3) Kerr: refined k0 via isometric embedding ----------
def kerr_sigma_components(R, M, a, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    Sigma = R*R + a*a*ct*ct
    Delta = R*R - 2*M*R + a*a
    A = (R*R + a*a)**2 - a*a*Delta*st*st
    sigma_thth = Sigma
    sigma_phph = (A * st*st) / Sigma
    return sigma_thth, sigma_phph, Sigma, Delta, A

def kerr_k_physical(R, M, a, theta):
    ct = np.cos(theta); st = np.sin(theta)
    Sigma = R*R + a*a*ct*ct
    Delta = R*R - 2*M*R + a*a
    A = (R*R + a*a)**2 - a*a*Delta*st*st
    Delta_p = 2*R - 2*M
    Sigma_p = 2*R
    A_p = 4*R*(R*R + a*a) - a*a*Delta_p*st*st
    F = A * Delta / Sigma
    F_p = (A_p * Delta + A * Delta_p) / Sigma - (A * Delta) * Sigma_p / (Sigma * Sigma)
    k = 0.5 * F_p / ( A * np.sqrt(Delta / Sigma) )
    sqrt_sigma = np.sqrt(A) * st
    return k, sqrt_sigma

def embed_isometric_euclid(R, M, a, thetas):
    sigma_thth, sigma_phph, _, _, _ = kerr_sigma_components(R, M, a, thetas)
    Remb = np.sqrt(sigma_phph)
    dθ = thetas[1] - thetas[0]
    Rp = np.zeros_like(Remb)
    Rp[1:-1] = (Remb[2:] - Remb[:-2]) / (2*dθ)
    Rp[0] = (Remb[1] - Remb[0]) / dθ
    Rp[-1] = (Remb[-1] - Remb[-2]) / dθ
    Zp_sq = sigma_thth - Rp**2
    Zp_sq = np.maximum(Zp_sq, 0.0)
    Zp = -np.sqrt(Zp_sq)
    Z = np.zeros_like(Remb)
    Z[1:] = np.cumsum(0.5*(Zp[1:] + Zp[:-1]) * dθ)
    return Remb, Z, Rp, Zp

def k0_from_embedding(Remb, Z, Rp, Zp, thetas):
    cosφ, sinφ = 1.0, 0.0
    H_mean = np.zeros_like(Remb)
    dθ = thetas[1] - thetas[0]
    Rpp = np.zeros_like(Remb); Rpp[1:-1] = (Rp[2:] - Rp[:-2])/(2*dθ); Rpp[0]=(Rp[1]-Rp[0])/dθ; Rpp[-1]=(Rp[-1]-Rp[-2])/dθ
    Zpp = np.zeros_like(Z);    Zpp[1:-1] = (Zp[2:] - Zp[:-2])/(2*dθ); Zpp[0]=(Zp[1]-Zp[0])/dθ; Zpp[-1]=(Zp[-1]-Zp[-2])/dθ
    for i in range(len(thetas)):
        Rv, Rpv, Rppv, Zpv, Zppv = Remb[i], Rp[i], Rpp[i], Zp[i], Zpp[i]
        Xθ = np.array([Rpv, 0.0, Zpv])
        Xφ = np.array([0.0, Rv, 0.0])
        Xθθ = np.array([Rppv, 0.0, Zppv])
        Xθφ = np.array([0.0, Rpv, 0.0])
        Xφφ = np.array([-Rv, 0.0, 0.0])
        E = np.dot(Xθ, Xθ); F = np.dot(Xθ, Xφ); G = np.dot(Xφ, Xφ)
        N = np.cross(Xθ, Xφ); Nn = np.linalg.norm(N)
        n_hat = N / Nn
        e = np.dot(n_hat, Xθθ); f = np.dot(n_hat, Xθφ); g = np.dot(n_hat, Xφφ)
        H_mean[i] = (e*G - 2*f*F + g*E) / (2*(E*G - F*F))
    k0_trace = 2*H_mean
    return k0_trace

def fig_kerr_embedding(R=200.0, M=1.0):
    a_vals = np.linspace(0.0, 1.2, 7)
    E_vals = []; abs_err = []
    for a in a_vals:
        thetas = np.linspace(1e-5, np.pi-1e-5, 1200)
        k_phys = np.zeros_like(thetas); sqrt_sigma = np.zeros_like(thetas)
        for i, th in enumerate(thetas):
            k_, s_ = kerr_k_physical(R, M, a, th)
            k_phys[i] = k_; sqrt_sigma[i] = s_
        Remb, Z, Rp, Zp = embed_isometric_euclid(R, M, a, thetas)
        k0_theta = k0_from_embedding(Remb, Z, Rp, Zp, thetas)
        integrand = (k0_theta - k_phys) * sqrt_sigma
        E_BY = (1/(8*np.pi)) * 2*np.pi * np.trapz(integrand, thetas)
        E_vals.append(E_BY); abs_err.append(abs(E_BY - M))
    a_vals = np.array(a_vals); abs_err = np.array(abs_err)
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(a_vals, abs_err, marker='o')
    plt.xlabel("Rapport $a/M$"); plt.ylabel("Erreur absolue $|E_{\\rm BY}(R)-M|$")
    plt.title("Kerr (BL, $R=200M$) : référence $k_0(\\theta)$ par embedding")
    plt.grid(True)
    savefig("fig_kerr_embedding_refined.pdf")

# ---------- 4) TOV: full integration (constant density) ----------
def integrate_TOV_const_density(rho0=8e-5, p_c=1e-3, r_stop=50.0, dr=1e-3):
    r_list=[dr]; m_list=[(4.0/3.0)*np.pi*rho0*dr**3]; p_list=[p_c]
    def dm_dr(r,m,p): return 4.0*np.pi*r*r*rho0
    def dp_dr(r,m,p): return - (rho0+p)*(m+4.0*np.pi*r**3*p) / ( r*(r-2*m) )
    r=dr; m=m_list[-1]; p=p_list[-1]
    while r<r_stop and p>0:
        k1m=dm_dr(r,m,p); k1p=dp_dr(r,m,p)
        k2m=dm_dr(r+0.5*dr, m+0.5*dr*k1m, p+0.5*dr*k1p)
        k2p=dp_dr(r+0.5*dr, m+0.5*dr*k1m, p+0.5*dr*k1p)
        k3m=dm_dr(r+0.5*dr, m+0.5*dr*k2m, p+0.5*dr*k2p)
        k3p=dp_dr(r+0.5*dr, m+0.5*dr*k2m, p+0.5*dr*k2p)
        k4m=dm_dr(r+dr, m+dr*k3m, p+dr*k3p)
        k4p=dp_dr(r+dr, m+dr*k3m, p+dr*k3p)
        m += (dr/6.0)*(k1m+2*k2m+2*k3m+k4m)
        p += (dr/6.0)*(k1p+2*k2p+2*k3p+k4p)
        r += dr
        r_list.append(r); m_list.append(m); p_list.append(p)
        if p<=0: break
    return np.array(r_list), np.array(m_list), np.array(p_list)

def fig_tov_full():
    r,m,p = integrate_TOV_const_density()
    inside = np.maximum(1.0 - 2.0*m/r, 1e-14)
    E_BY = r * (1.0 - np.sqrt(inside))
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(r, m, label="$m(r)$")
    plt.plot(r, E_BY, linestyle='--', label="$E_{\\rm BY}(r)$")
    plt.xlabel("Rayon $r$"); plt.ylabel("Masse")
    plt.title("TOV (densité constante) : $m(r)$ vs $E_{\\rm BY}(r)$")
    plt.grid(True); plt.legend()
    savefig("fig_tov_full.pdf")

# ---------- 5) Extra dimension effect ----------
def fig_extra_dimension():
    a_base=15.0; M=1.0
    R_extra = np.logspace(0, 3, 400)
    R = a_base
    M_4D = R * (1 - np.sqrt(1 - 2*M/R))
    err4 = np.abs(M_4D - M)
    M_extra = 1.0 / R_extra
    total_err = np.abs(M_4D + M_extra - M)
    plt.figure(figsize=(6.2, 4.2))
    plt.loglog(R_extra, total_err, label="Erreur 4D+1D")
    plt.axhline(err4, linestyle='--', label="Erreur 4D seule")
    plt.xlabel("Rayon extra $R_{\\rm extra}$"); plt.ylabel("Erreur absolue")
    plt.title("Effet d'une dimension supplémentaire 1D")
    plt.grid(True, which="both"); plt.legend()
    savefig("fig_extra_dimension_effect_improved.pdf")

if __name__ == "__main__":
    fig_sphere_convergence()
    fig_ellipsoids_smooth()
    fig_kerr_embedding()
    fig_tov_full()
    fig_extra_dimension()
    print("All figures generated.")
