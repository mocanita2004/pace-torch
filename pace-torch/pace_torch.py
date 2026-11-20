# -*- coding: utf-8 -*-
r"""
pace_torch.py — Reimplementazione differenziabile del modello PACE in PyTorch
=============================================================================

Paper di riferimento (citato sotto come "[PACE]"):
  G. Palma, S. Monti, A. Buonanno, R. Pacelli, L. Cella,
  "PACE: A Probabilistic Atlas for Normal Tissue Complication Estimation in
  Radiation Oncology", Frontiers in Oncology 9:130 (2019).

Di seguito un riassunto del modello e di come è implementato.

-------------------------------------------------------------------------------
IL MODELLO IN BREVE ([PACE], "Model Design", Figure 1-3, Equazioni 1-3)
-------------------------------------------------------------------------------
  FASE 1 (Fig. 2): le mappe di dose degli N pazienti sono deformate su
    un'anatomia comune. Per OGNI voxel j si fa una regressione logistica
    dell'esito O_i (0/1) sulla dose locale D_i(x_j) e sulle covariate (età):
        P_j(D, V) = sigma(a_j + b_j D + c_j V),     sigma(z) = 1/(1+e^-z)
  FASE 2 (Fig. 3): per un paziente, voxel per voxel:
        P(x_j) = probabilità predetta  ("P-map")
        W(x_j) = 1 / ampiezza del CI al 95% di P(x_j)  ("W-map"),
                 W = 0 dove b_j < 0 (odds ratio della dose < 1)
  FASE 3: riassunto alla Lyman-Kutcher-Burman:
        gEUp = [ sum_j P^(1/nu) W / sum_j W ]^nu                  (Eq. 3)
        t    = (gEUp - Tp50) / (mu Tp50)                            (Eq. 2)
        PACE = (1 + erf(t / sqrt 2)) / 2 = Phi(t)                   (Eq. 1)
    con nu (effetto volume), mu (pendenza), Tp50 stimati per massima
    verosimiglianza.

-------------------------------------------------------------------------------
LA PROGRAMMAZIONE DIFFERENZIABILE IN PYTORCH
-------------------------------------------------------------------------------
Addestrare = minimizzare la log-verosimiglianza negativa
    L(theta) = - sum_i [ O_i log PACE_i + (1 - O_i) log(1 - PACE_i) ].
Serve il gradiente dL/dtheta. PyTorch lo calcola con la differenziazione
automatica (autodiff): mentre il codice gira, PyTorch registra in un
"grafo computazionale" tutte le operazioni elementari fatte sui tensori con
requires_grad=True; chiamando loss.backward() percorre il grafo all'indietro
applicando la regola della catena e deposita in theta.grad la derivata
ESATTA di loss rispetto a theta. È lo stile "imperativo" di PyTorch.

PyTorch offre anche uno stile "funzionale", identico a quello di JAX, nel
modulo torch.func:
    torch.func.grad(f)     -> funzione che calcola il gradiente di f
    torch.func.jacrev(f)   -> jacobiana di f (Hessiana = jacrev(grad(f)))
    torch.func.vmap(f)     -> applica f su un asse "batch" (tanti voxel)
                              senza cicli for espliciti
Qui usiamo entrambi gli stili:
  - FASE 1 (Newton per ogni voxel): torch.func.grad / jacrev / vmap
  - FASE 3 (Adam): lo stile classico con .backward() e torch.optim.Adam
  - Newton e incertezze sui 3 parametri: Hessiana con torch.func

VERSIONE SOLO PER I DATI IN SILICO: la lettura dei dati clinici (.mat,
CSV degli esiti) e il Leave-One-Out sono stati tolti.

Uso:
  pip install torch scipy numpy matplotlib
  python pace_torch.py                       # 400 mappe sintetiche, n = 0.5
  python pace_torch.py --n-synthetic 200 --n-volume 0.25
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import grad, jacrev, vmap

from pace_common import (Cohort, make_synthetic_cohort,
                         classification_metrics, print_metrics, to_volume)

# Doppia precisione: somme su ~10^5 voxel e potenze P^(1/nu) con nu ~ 0.05
# richiedono precisione.
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z95 = 1.959963984540054   # quantile della gaussiana per un CI al 95%


def T(x):
    """Converte un array NumPy in un tensore PyTorch (float64) sul device."""
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=DEVICE)


# =============================================================================
# FASE 1 — un GLM logistico per ogni voxel  ([PACE] Figura 2)
# =============================================================================
# Predittori del paziente i nel voxel j:  z_ij = (1, D_i(x_j)/s, V_i1, ..., V_iK)
# Modello:  P_ij = sigma(z_ij . beta_j)
# -log L_j = sum_i [ log(1 + e^eta_ij) - O_i eta_ij ],   eta = z . beta

def voxel_nll(beta, Z, y, ridge):
    """-log L per UN voxel (beta: (p,), Z: (N,p), y: (N,), ridge: (p,)).

    F.softplus(eta) = log(1 + e^eta) calcolato in modo stabile.
    La piccola penalità 0.5*ridge*beta^2 NON è nel paper: evita coefficienti
    infiniti nei voxel con dati "perfettamente separabili" (pochi eventi).
    Non si penalizza l'intercetta (ridge[0] = 0).
    """
    eta = Z @ beta
    return torch.sum(F.softplus(eta) - y * eta) + 0.5 * torch.sum(ridge * beta ** 2)


# torch.func.grad: da una funzione scalare otteniamo la funzione che ne
# calcola il gradiente (p,) rispetto a beta. L'Hessiana (p,p) è la
# "jacobiana del gradiente": jacrev(grad(f)).
# (Esiste anche torch.func.hessian, che usa la modalità "forward" per il
# secondo livello di derivate: è più veloce ma non tutte le funzioni di
# PyTorch la supportano; derivare due volte "all'indietro" è più robusto.)
def hessian(f):
    return jacrev(grad(f))


voxel_grad = grad(voxel_nll)
voxel_hess = hessian(voxel_nll)

# vmap le "vettorizza" su un asse batch: in_dims=(0, 0, None, None) vuol
# dire "beta e Z hanno un voxel per riga (asse 0); y e ridge sono condivisi".
voxel_grad_batch = vmap(voxel_grad, in_dims=(0, 0, None, None))
voxel_hess_batch = vmap(voxel_hess, in_dims=(0, 0, None, None))


def fit_voxel_chunk(D_chunk, V, y, ridge, max_step, n_iter):
    """Regressioni logistiche di m voxel in parallelo col metodo di Newton.

    D_chunk: (N, m) dosi (già divise per dose_scale); V: (N, K); y: (N,).
    Newton: beta <- beta - H^{-1} g  (salto al minimo della parabola che
    approssima -log L). Alla fine Cov(beta) ~ H^{-1} (informazione di Fisher).
    Restituisce beta (m, p) e cov (m, p, p).
    """
    N, m = D_chunk.shape
    K = V.shape[1]
    Z = torch.cat([torch.ones(m, N, 1, device=DEVICE),
                   D_chunk.T.unsqueeze(-1),
                   V.unsqueeze(0).expand(m, N, K)], dim=-1)          # (m, N, p)
    beta = torch.zeros(m, Z.shape[-1], device=DEVICE)
    for _ in range(n_iter):
        g = voxel_grad_batch(beta, Z, y, ridge)                        # (m, p)
        H = voxel_hess_batch(beta, Z, y, ridge)                        # (m, p, p)
        step = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)   # H step = g, per ogni voxel
        beta = beta - step.clamp(-max_step, max_step)
    cov = torch.linalg.inv(voxel_hess_batch(beta, Z, y, ridge))
    return beta, cov


# =============================================================================
# FASE 2 — P-map e W-map  ([PACE] Figura 3)
# =============================================================================
def pw_maps_chunk(beta, cov, D_chunk, V):
    """log P e log W (n, m) per n pazienti e m voxel.

    Usiamo i logaritmi perché P^(1/nu) con nu ~ 0.05 è P^20: senza log
    avremmo numeri ~1e-40 che diventano 0 (underflow).
    """
    n, m = D_chunk.shape
    K = V.shape[1]
    Z = torch.cat([torch.ones(n, m, 1, device=DEVICE), D_chunk.unsqueeze(-1),
                   V.unsqueeze(1).expand(n, m, K)], dim=-1)               # (n, m, p)
    eta = torch.einsum("nmp,mp->nm", Z, beta)
    # Metodo delta: Var(z . beta) = z^T Cov z
    se = torch.sqrt(torch.einsum("nmp,mpq,nmq->nm", Z, cov, Z))
    # Ampiezza del CI al 95% su P: sigma(a) - sigma(b), a,b = eta +- 1.96 se,
    # calcolata in modo stabile con  sigma(a)-sigma(b) = sigma(a) sigma(-b) (1 - e^(b-a))
    a, b = eta + Z95 * se, eta - Z95 * se
    log_ci = F.logsigmoid(a) + F.logsigmoid(-b) + torch.log(-torch.expm1(-2.0 * Z95 * se))
    log_P = F.logsigmoid(eta)
    # W = 1/CI, e W = 0 (log W = -inf) dove l'odds ratio della dose è < 1
    log_W = torch.where(beta[None, :, 1] > 0, -log_ci, torch.full_like(log_ci, -math.inf))
    return log_P, log_W


# =============================================================================
# FASE 3 — gEUp e probabilità PACE  ([PACE] Equazioni 1-3)
# =============================================================================
# Ottimizziamo theta in R^3 senza vincoli e lo trasformiamo in
#   nu, mu in (1e-3, 1e2)  (sigmoide in scala logaritmica)  e  Tp50 in (0, 1)  (sigmoide)
NU_RANGE = (1e-3, 1e2)
MU_RANGE = (1e-3, 1e2)


def _to_range(theta, lo, hi):
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(theta))


def _from_range(x, lo, hi):
    u = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
    u = np.clip(u, 1e-9, 1 - 1e-9)
    return np.log(u / (1 - u))


def unpack(theta):
    """theta (3,) non vincolato -> (nu, mu, Tp50)."""
    return (_to_range(theta[0], *NU_RANGE), _to_range(theta[1], *MU_RANGE),
            torch.sigmoid(theta[2]))


def pack(nu, mu, tp50):
    """(nu, mu, Tp50) -> theta (inversa di unpack)."""
    tp50 = float(np.clip(tp50, 1e-6, 1 - 1e-6))
    return T([_from_range(nu, *NU_RANGE), _from_range(mu, *MU_RANGE), np.log(tp50 / (1 - tp50))])


def log_geup(nu, log_P, log_W):
    """log gEUp (Eq. 3):  nu * [logsumexp(log P/nu + log W) - logsumexp(log W)].
    I voxel con W = 0 (log W = -inf) non contribuiscono."""
    return nu * (torch.logsumexp(log_P / nu + log_W, dim=1) - torch.logsumexp(log_W, dim=1))


def t_stat(theta, log_P, log_W):
    """t dell'Eq. 2."""
    nu, mu, tp50 = unpack(theta)
    geup = torch.exp(log_geup(nu, log_P, log_W))
    return (geup - tp50) / (mu * tp50)


def pace_nll(theta, log_P, log_W, y):
    """-log L(nu, mu, Tp50).  PACE = Phi(t), 1 - Phi(t) = Phi(-t);
    torch.special.log_ndtr(t) = log Phi(t) in modo numericamente stabile."""
    t = t_stat(theta, log_P, log_W)
    return -torch.sum(y * torch.special.log_ndtr(t) + (1.0 - y) * torch.special.log_ndtr(-t))


def run_adam(theta0, log_P, log_W, y, n_steps, lr):
    """Minimizza pace_nll con Adam (stile "imperativo" di PyTorch).

    Il ciclo tipico di addestramento in PyTorch è sempre questo:
       1. opt.zero_grad()   azzera i gradienti del passo precedente
       2. loss = f(theta)   calcolo "in avanti" (PyTorch registra il grafo)
       3. loss.backward()   autodiff all'indietro: riempie theta.grad
       4. opt.step()        aggiorna theta usando theta.grad
    """
    theta = theta0.clone().requires_grad_(True)   # "traccia le operazioni su theta"
    opt = torch.optim.Adam([theta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)  # lr -> 0
    for _ in range(n_steps):
        opt.zero_grad()
        loss = pace_nll(theta, log_P, log_W, y)
        loss.backward()
        opt.step()
        sched.step()
    return theta.detach()   # detach: "stacca" theta dal grafo, ora è un numero e basta


def nll_grad_hess(theta, log_P, log_W, y):
    """Valore, gradiente (3,) e Hessiana (3,3) di -log L con torch.func."""
    f = lambda th: pace_nll(th, log_P, log_W, y)
    return f(theta), grad(f)(theta), hessian(f)(theta)


def newton_minimize(theta, log_P, log_W, y, free=(True, True, True), n_iter=50, tol=1e-10):
    """Newton "con salvagente" (Levenberg-Marquardt + backtracking) sui
    parametri liberi `free`; gli altri restano fissi (profilo di
    verosimiglianza)."""
    theta = theta.detach().cpu().numpy().astype(float).copy()
    idx = np.flatnonzero(free)

    def vgh(th):
        f, g, H = nll_grad_hess(T(th), log_P, log_W, y)
        return float(f), g.detach().cpu().numpy(), H.detach().cpu().numpy()

    def val(th):
        with torch.no_grad():   # qui non servono gradienti: niente grafo
            return float(pace_nll(T(th), log_P, log_W, y))

    f, g, H = vgh(theta)
    for _ in range(n_iter):
        gs, Hs = g[idx], H[np.ix_(idx, idx)]
        lam_min = np.linalg.eigvalsh(Hs).min()
        if lam_min <= 1e-8:                          # H non definita positiva
            Hs = Hs + (1e-3 - lam_min) * np.eye(len(idx))
        step = np.linalg.solve(Hs, gs)
        a, accepted = 1.0, False
        while a > 1e-6:                               # backtracking
            th_new = theta.copy()
            th_new[idx] -= a * step
            f_new = val(th_new)
            if np.isfinite(f_new) and f_new <= f:
                accepted = True
                break
            a *= 0.5
        if not accepted:
            break
        improvement = f - f_new
        theta = th_new
        f, g, H = vgh(theta)
        if improvement < tol:
            break
    return T(theta), f


# =============================================================================
# Il modello completo
# =============================================================================
class PACE:
    """Modello PACE: fit (addestramento) e predict_proba (predizione)."""

    def __init__(self, ridge=1e-2, n_newton=30, max_step=5.0, chunk=20000,
                 n_adam=500, lr=0.05, nu_inits=(0.05, 0.1, 0.3, 1.0), verbose=True):
        self.ridge, self.n_newton, self.max_step = ridge, n_newton, max_step
        self.chunk, self.n_adam, self.lr = chunk, n_adam, lr
        self.nu_inits, self.verbose = nu_inits, verbose

    # ---------------- FASE 1 ----------------
    def fit_voxel_glms(self, coh: Cohort):
        D = T(coh.dose) / coh.dose_scale
        V, y = T(coh.cov), T(coh.y)
        p = 2 + V.shape[1]
        ridge = torch.full((p,), self.ridge, device=DEVICE)
        ridge[0] = 0.0                                     # niente penalità sull'intercetta
        M = D.shape[1]
        c = min(self.chunk, M)                             # blocchi di voxel (memoria)
        betas, covs = [], []
        t0 = time.time()
        for start in range(0, M, c):
            b, C = fit_voxel_chunk(D[:, start:start + c], V, y, ridge, self.max_step, self.n_newton)
            betas.append(b.cpu().numpy()); covs.append(C.cpu().numpy())
        self.beta = np.concatenate(betas)        # (M, p)
        self.cov_beta = np.concatenate(covs)     # (M, p, p)
        self.dose_scale = coh.dose_scale
        if self.verbose:
            print(f"[fase 1] {M} GLM stimati in {time.time() - t0:.1f} s; "
                  f"voxel con odds ratio della dose > 1: {np.mean(self.beta[:, 1] > 0):.1%}")
        return self

    # ---------------- FASE 2 ----------------
    @torch.no_grad()    # le mappe sono "dati" per la fase 3: non serve il grafo
    def maps(self, coh: Cohort):
        D = T(coh.dose) / self.dose_scale
        V = T(coh.cov)
        M = D.shape[1]
        c = min(self.chunk, M)
        lP, lW = [], []
        for start in range(0, M, c):
            sl = slice(start, start + c)
            P_, W_ = pw_maps_chunk(T(self.beta[sl]), T(self.cov_beta[sl]), D[:, sl], V)
            lP.append(P_); lW.append(W_)
        log_P, log_W = torch.cat(lP, dim=1), torch.cat(lW, dim=1)
        if bool(torch.any(torch.all(torch.isinf(log_W), dim=1))):
            raise RuntimeError("Un paziente ha W = 0 in tutti i voxel: la gEUp non è definita.")
        return log_P, log_W

    # ---------------- FASE 3 ----------------
    def fit_global(self, log_P, log_W, y, theta_init=None):
        y = T(y)
        if theta_init is not None:
            starts = [theta_init.clone()]
        else:
            starts = []
            for nu0 in self.nu_inits:
                with torch.no_grad():
                    g = torch.exp(log_geup(nu0, log_P, log_W)).cpu().numpy()
                tp0 = float(np.clip(np.median(g), 1e-4, 1 - 1e-4))
                mu0 = float(np.clip(np.std(g) / tp0, 2e-3, 50.0))
                starts.append(pack(nu0, mu0, tp0))
        best = None
        for th0 in starts:
            theta = run_adam(th0, log_P, log_W, y, self.n_adam, self.lr)
            with torch.no_grad():
                nll = float(pace_nll(theta, log_P, log_W, y))
            if self.verbose:
                nu, mu, tp = (float(x) for x in unpack(theta))
                print(f"[fase 3] Adam da nu0={float(unpack(th0)[0]):.3g} -> nu={nu:.3g} mu={mu:.3g} "
                      f"Tp50={tp:.3g}  -logL={nll:.4f}")
            if best is None or nll < best[1]:
                best = (theta, nll)
        self.theta, self.nll_min = newton_minimize(best[0], log_P, log_W, y)
        if self.verbose:
            print(f"[fase 3] dopo Newton: {self.params}  -logL={self.nll_min:.4f}")
        if self.params["nu"] > 0.5 * NU_RANGE[1]:
            print("  nota: nu è al bordo superiore. Per nu -> infinito la gEUp tende alla media\n"
                  "  geometrica pesata delle P e la verosimiglianza diventa piatta in nu:\n"
                  "  le predizioni sono valide, ma l'incertezza su nu non è stimabile.")
        return self

    def fit(self, coh: Cohort):
        """Fase 1 + fase 2 + fase 3 (Figura 1, "Model Training")."""
        self.fit_voxel_glms(coh)
        log_P, log_W = self.maps(coh)
        self.fit_global(log_P, log_W, coh.y)
        self._train_maps = (log_P, log_W, T(coh.y))
        return self

    @torch.no_grad()
    def predict_proba(self, coh: Cohort):
        """Probabilità PACE (Figura 1, "Model Application")."""
        log_P, log_W = self.maps(coh)
        return torch.special.ndtr(t_stat(self.theta, log_P, log_W)).cpu().numpy()

    @property
    def params(self):
        nu, mu, tp = unpack(self.theta)
        return {"nu": float(nu), "mu": float(mu), "Tp50": float(tp)}

    # ---------------- Incertezze ----------------
    def wald_ci(self):
        """CI al 95% da Cov(theta) ~ H^{-1}, con H Hessiana di -log L
        (calcolata con torch.func), poi trasformato nei parametri fisici."""
        log_P, log_W, y = self._train_maps
        _, _, H = nll_grad_hess(self.theta, log_P, log_W, y)
        se = np.sqrt(np.diag(np.linalg.inv(H.detach().cpu().numpy())))
        th = self.theta.cpu().numpy()
        out = {}
        for k, name in enumerate(["nu", "mu", "Tp50"]):
            lo, hi = th.copy(), th.copy()
            lo[k] -= Z95 * se[k]; hi[k] += Z95 * se[k]
            out[name] = tuple(sorted((float(unpack(T(lo))[k]), float(unpack(T(hi))[k]))))
        return out

    def profile_ci(self, n_grid=41, span=4.0):
        """CI al 95% col profilo di verosimiglianza (Wilks), come nel paper:
        si fissa un parametro, si ri-ottimizzano gli altri due, e si tengono
        i valori con 2 * Delta(-log L) <= 3.84 (chi^2 con 1 grado di libertà).

        La griglia su cui si fissa il parametro è centrata nella stima e va da
        -span a +span errori standard di Wald (in theta): così il passo si
        adatta alla larghezza vera dell'intervallo. Gli estremi del CI sono
        trovati interpolando linearmente dove 2 * Delta(-log L) = 3.84,
        invece di prendere il punto di griglia più vicino."""
        log_P, log_W, y = self._train_maps
        th_hat = self.theta.cpu().numpy()
        _, _, H = nll_grad_hess(self.theta, log_P, log_W, y)
        se = np.sqrt(np.diag(np.linalg.inv(H.detach().cpu().numpy())))
        soglia = 3.841
        out = {}
        for k, name in enumerate(["nu", "mu", "Tp50"]):
            free = [j != k for j in range(3)]
            offsets = np.linspace(-span, span, n_grid) * se[k]
            prof = np.full(n_grid, np.nan)
            c = n_grid // 2
            for direction in (range(c, n_grid), range(c, -1, -1)):
                th = th_hat.copy()
                for i in direction:
                    th[k] = th_hat[k] + offsets[i]
                    th_opt, f = newton_minimize(T(th), log_P, log_W, y, free=free, n_iter=30)
                    th = th_opt.cpu().numpy()
                    prof[i] = f
            dev = 2 * (prof - min(np.nanmin(prof), self.nll_min))   # 2 * Delta(-log L)

            def estremo(indici):
                """Cammina dal centro verso un lato e restituisce il theta in cui
                dev attraversa la soglia (interpolazione lineare)."""
                prec = c
                for i in indici:
                    if not np.isfinite(dev[i]):
                        continue
                    if dev[i] > soglia:
                        frac = (soglia - dev[prec]) / (dev[i] - dev[prec])
                        return offsets[prec] + frac * (offsets[i] - offsets[prec]), False
                    prec = i
                return offsets[prec], True                      # soglia mai superata

            o_lo, bordo_lo = estremo(range(c - 1, -1, -1))
            o_hi, bordo_hi = estremo(range(c + 1, n_grid))
            vals = []
            for o in (o_lo, o_hi):
                th = th_hat.copy()
                th[k] += o
                vals.append(float(unpack(T(th))[k]))
            out[name] = tuple(sorted(vals))
            if bordo_lo or bordo_hi:
                print(f"  attenzione: il CI di {name} tocca il bordo della griglia (aumenta `span`)")
        return out


# =============================================================================
# Salvataggio delle mappe
# =============================================================================
def _fetta(vol, z):
    """Volume 3D (dati clinici) -> fetta assiale z; mappe 2D (sintetici) restano uguali."""
    return vol if vol.ndim == 2 else vol[:, :, z]


def _log10(W):
    """log10 di una mappa di pesi W. Dove W = 0 (voxel esclusi) o fuori dalla
    ROI il risultato è NaN, e nel disegno resta bianco."""
    out = np.full(W.shape, np.nan)
    pos = W > 0
    out[pos] = np.log10(W[pos])
    return out


def _cmap(nome):
    import matplotlib.pyplot as plt
    c = plt.get_cmap(nome).copy()
    c.set_bad("white")
    return c


def _no_offset(cb):
    """Evita etichette tipo '1e-12 + 4.93e-1' sulla barra dei colori."""
    try:
        cb.formatter.set_useOffset(False)
        cb.update_ticks()
    except AttributeError:
        pass


def _mosaico(sub, mappe, etichette, titolo, nome_barra, cmap, vmin, vmax, z):
    """Disegna fino a 9 mappe in una griglia 3x3 dentro la sotto-figura `sub`,
    con un'unica barra dei colori (stessa scala per tutti i pazienti)."""
    axs = sub.subplots(3, 3)
    im = None
    for k, ax in enumerate(axs.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if k >= len(mappe):
            ax.axis("off")
            continue
        im = ax.imshow(_fetta(mappe[k], z).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(etichette[k], fontsize=11)
    sub.suptitle(titolo, fontsize=15)
    cb = sub.colorbar(im, ax=axs, location="bottom", shrink=0.8, pad=0.04)
    cb.set_label(nome_barra)
    _no_offset(cb)
    return axs


def _figura_rs(rs, path, z=None):
    """Figura a sé: mappa di radiosensibilità VERA (usata per generare gli esiti)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(_fetta(rs, z).T, origin="lower", cmap=_cmap("viridis"))
    ax.set_title("Radiosensibilità VERA\n(usata per generare gli esiti)", fontsize=12)
    ax.set_xlabel(r"voxel $x$"); ax.set_ylabel(r"voxel $y$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("RS (unità arbitrarie)")
    _no_offset(cb)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _figura_pazienti(dosi, P, W, y, pace, geup, titolo_set,
                     path, z=None, dose_label="dose"):
    """Figura dei pazienti: dose / P-map / W-map di 9 pazienti e tabella
    esiti veri vs PACE predetta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(21, 8.2))
    sd, sp, sw, st = fig.subfigures(1, 4, width_ratios=[1.35, 1.35, 1.35, 1.25])
    lab = [f"P{k + 1}" for k in range(len(dosi))]

    # --- dose, P-map, W-map di 9 pazienti -------------------------------------------
    axs_d = _mosaico(sd, dosi, lab, f"Dose\n({titolo_set})", dose_label, _cmap("inferno"),
             np.nanmin(dosi), np.nanmax(dosi), z)
    _mosaico(sp, P, lab, r"$P$-map predetta" "\n(fase 2)", r"$P$", _cmap("viridis"), 0.0, 1.0, z)
    lw = np.stack([_log10(w) for w in W])
    _mosaico(sw, lw, lab, r"$W$-map" "\n" r"(fase 2, scala $\log_{10}$)", r"$\log_{10} W$", _cmap("magma"),
             np.nanmin(lw), np.nanmax(lw), z)

    # --- tabella esiti veri vs PACE ----------------------------------------------
    # La tabella va allineata alle immagini: il bordo superiore all'altezza
    # della prima riga di mappe (P1-P3), quello inferiore all'ultima (P7-P9).
    # Le posizioni si conoscono solo dopo aver disegnato la figura.
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    top = to_fig.transform(axs_d[0, 0].get_window_extent())[1, 1]
    bottom = to_fig.transform(axs_d[-1, 0].get_window_extent())[0, 1]
    (x0, _), (x1, _) = to_fig.transform(st.bbox)
    w = x1 - x0
    ax_t = fig.add_axes([x0 + 0.06 * w, bottom, 0.88 * w, top - bottom])
    ax_t.axis("off")
    righe = [[lab[k], str(int(y[k])), f"{geup[k]:.3f}", f"{pace[k]:.3f}"] for k in range(len(dosi))]
    # bbox = [0, 0, 1, 1]: la tabella riempie tutto il riquadro ax_t
    tab = ax_t.table(cellText=righe,
                     colLabels=["", "esito\nvero", "gEUp", "PACE\npredetta"],
                     cellLoc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    tab.auto_set_font_size(False); tab.set_fontsize(15)
    for (r, c), cella in tab.get_celld().items():
        if r == 0:
            cella.set_facecolor("#e8e8e8")
            cella.set_text_props(weight="bold")
    st.suptitle("Esiti veri e PACE predetta", fontsize=15)

    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figura_medie(log_or, std_P, mean_W, path, z=None, log_or_label=r"$b_j$  [dose$^{-1}$]"):
    """Figura 2: b_j, deviazione standard di P e W media sui pazienti di
    addestramento."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    pannelli = [
        (log_or, r"$b_j$ stimato in ogni voxel (fase 1)" "\nalto = la dose qui è associata all'esito",
         log_or_label, "inferno", {}),
        (std_P, r"Deviazione standard di $P$ tra i pazienti" "\nalto = il voxel distingue bene i pazienti",
         r"$\sigma_P$", "viridis", {"vmin": 0.0, "vmax": 0.5}),
        (_log10(mean_W), r"$\overline{W}$ (media di $W$ sui pazienti di addestramento)" "\n" r"(scala $\log_{10}$)",
         r"$\log_{10}\,\overline{W}$", "magma", {}),
    ]
    for ax, (vol, titolo, nome, cm, kw) in zip(axs, pannelli):
        im = ax.imshow(_fetta(vol, z).T, origin="lower", cmap=_cmap(cm), **kw)
        ax.set_title(titolo, fontsize=13)
        ax.set_xlabel(r"voxel $x$"); ax.set_ylabel(r"voxel $y$")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(nome)
        _no_offset(cb)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_results(model: PACE, coh: Cohort, out_dir, extra=None, show=None, n_show=9):
    """Salva i numeri (.npz) e tre figure (.png):
      pace_torch_rs.png        radiosensibilità vera + parametri (solo dati sintetici)
      pace_torch_pazienti.png  dose, P-map, W-map di 9 pazienti + tabella esiti
      pace_torch_maps.png      b_j, dev. std di P, W media

    coh   : pazienti di addestramento (per le mappe medie)
    show  : pazienti da mostrare nella figura dei singoli pazienti
            (es. quelli di validazione). Default: `coh`.
    n_show: quanti pazienti mostrare (i primi n_show di `show`, al massimo 9)
    """
    os.makedirs(out_dir, exist_ok=True)
    log_P, log_W, _ = model._train_maps
    res = dict(
        beta=model.beta, mask=coh.mask, dose_scale=coh.dose_scale,
        theta=model.theta.cpu().numpy(), **model.params,
        log_or_map=to_volume(model.beta[:, 1], coh.mask),       # log odds-ratio per dose_scale Gy
        mean_P_map=to_volume(torch.exp(log_P).mean(0).cpu().numpy(), coh.mask),
        mean_W_map=to_volume(torch.exp(log_W).mean(0).cpu().numpy(), coh.mask),
        std_P_map=to_volume(torch.exp(log_P).std(0).cpu().numpy(), coh.mask),
    )

    # --- mappe e predizioni dei pazienti da mostrare ------------------------------
    da_validazione = show is not None
    show = coh if show is None else show
    sub = show.subset(list(range(min(n_show, show.n_patients, 9))))
    with torch.no_grad():
        lP, lW = model.maps(sub)
        pace = torch.special.ndtr(t_stat(model.theta, lP, lW)).cpu().numpy()
        geup = torch.exp(log_geup(unpack(model.theta)[0], lP, lW)).cpu().numpy()
        P_sub, W_sub = torch.exp(lP).cpu().numpy(), torch.exp(lW).cpu().numpy()
    res.update(
        show_ids=np.array(sub.ids), show_y=sub.y, show_pace=pace, show_geup=geup,
        show_dose_maps=np.stack([to_volume(d, show.mask) for d in sub.dose]),
        show_P_maps=np.stack([to_volume(p, show.mask) for p in P_sub]),
        show_W_maps=np.stack([to_volume(w, show.mask) for w in W_sub]),
    )
    if extra:
        res.update(extra)
    np.savez_compressed(os.path.join(out_dir, "pace_torch_results.npz"), **res)

    sintetico = coh.dose_scale == 1.0
    p = model.params
    testo = (f"Dati di addestramento\n"
             f"  pazienti    N = {coh.n_patients}\n"
             f"  voxel       M = {coh.n_voxels}\n"
             f"  eventi        = {int(coh.y.sum())} ({100 * coh.y.mean():.1f}%)\n\n"
             f"Parametri stimati\n"
             f"  nu    = {p['nu']:.3f}\n"
             f"  mu    = {p['mu']:.3f}\n"
             f"  Tp50  = {p['Tp50']:.3f}\n"
             f"  -logL / N = {model.nll_min / coh.n_patients:.3f}")
    print(testo)
    z = int(np.argmax(coh.mask.sum(axis=(0, 1)))) if coh.mask.ndim == 3 else None
    try:
        if res.get("rs_true") is not None:
            _figura_rs(res["rs_true"], os.path.join(out_dir, "pace_torch_rs.png"), z=z)
        _figura_pazienti(res["show_dose_maps"], res["show_P_maps"],
                         res["show_W_maps"], sub.y, pace, geup,
                         "pazienti di validazione" if da_validazione else "pazienti della coorte",
                         os.path.join(out_dir, "pace_torch_pazienti.png"), z=z,
                         dose_label=r"$D$ (unità arbitrarie)" if sintetico else r"$D$ (Gy)")
        _figura_medie(res["log_or_map"], res["std_P_map"], res["mean_W_map"],
                      os.path.join(out_dir, "pace_torch_maps.png"), z=z,
                      log_or_label=(r"$b_j$  [dose$^{-1}$]" if sintetico
                                    else rf"$b_j$  [({coh.dose_scale:g} Gy)$^{{-1}}$]"))
    except ImportError:
        pass
    print(f"[output] risultati salvati in {out_dir}")


# =============================================================================
# main
# =============================================================================
def print_ci(model: PACE):
    """Scrive sul terminale i parametri stimati con i due intervalli di
    confidenza al 95%:
      - Wald: dalla curvatura (Hessiana) di -log L nel minimo, veloce ma
        assume che -log L sia una parabola;
      - profilo di verosimiglianza: il metodo usato nel paper, più lento ma
        corretto anche quando -log L è asimmetrica."""
    print("--- intervalli di confidenza al 95% ---")
    t0 = time.time()
    wald = model.wald_ci()
    prof = model.profile_ci()
    print(f"  {'parametro':10s} {'stima':>8s}   {'CI Wald':>19s}   {'CI profilo':>19s}")
    for name, label in [("nu", "nu"), ("mu", "mu"), ("Tp50", "Tp50")]:
        w_lo, w_hi = wald[name]
        p_lo, p_hi = prof[name]
        print(f"  {label:10s} {model.params[name]:8.3f}   "
              f"[{w_lo:7.3f}, {w_hi:7.3f}]   [{p_lo:7.3f}, {p_hi:7.3f}]")
    print(f"  (calcolati in {time.time() - t0:.1f} s)")


def main():
    ap =argparse.ArgumentParser(description="PACE in PyTorch (dati in silico)")
    ap.add_argument("--synthetic", action="store_true",
                    help="non serve più (i dati sono sempre sintetici): accettato per compatibilità")
    ap.add_argument("--n-synthetic", type=int, default=400, help="numero di mappe sintetiche")
    ap.add_argument("--n-volume", type=float, default=0.5, help="n del g2EUD sintetico (Eq. 4)")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out", default="risultati_torch")
    args = ap.parse_args()

    # Come nel paper: addestramento sul 75% delle mappe, validazione sul resto
    coh = make_synthetic_cohort(args.n_synthetic, n_volume=args.n_volume, seed=1)
    n_tr = int(0.75 * coh.n_patients)
    train, val = coh.subset(range(n_tr)), coh.subset(range(n_tr, coh.n_patients))
    model = PACE(ridge=args.ridge).fit(train)
    print("parametri:", model.params)
    print_metrics("training", classification_metrics(train.y, model.predict_proba(train)))
    print_metrics("validazione", classification_metrics(val.y, model.predict_proba(val)))
    print_ci(model)
    save_results(model, train, args.out, {"rs_true": coh.rs_true}, show=val)


if __name__ == "__main__":
    main()
