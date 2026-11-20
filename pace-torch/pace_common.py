# -*- coding: utf-8 -*-
"""
pace_common.py  —  VERSIONE SOLO PER I DATI IN SILICO
=====================================================

Funzioni "di servizio" usate da `pace_torch.py`. Qui NON c'è programmazione
differenziabile: è tutto NumPy "normale". Il modulo si occupa di:

  1. contenere i dati di una coorte (classe Cohort);
  2. generare dati SINTETICI come nella sezione "In silico Validation" del
     paper (mappe di dose con picchi gaussiani, radiosensibilità nota,
     esiti dal g^2EUD);
  3. calcolare le metriche di prestazione usate nel paper (AUC, accuratezza,
     accuratezza bilanciata, F1, curva di calibrazione; Tabella 1).

Rispetto alla versione completa sono state tolte le funzioni per i dati
clinici (lettura dei file .mat, riduzione di risoluzione dei volumi 3D,
lettura della tabella degli esiti).

Riferimento:
  G. Palma, S. Monti, A. Buonanno, R. Pacelli, L. Cella,
  "PACE: A Probabilistic Atlas for Normal Tissue Complication Estimation in
  Radiation Oncology", Front. Oncol. 9:130 (2019). doi:10.3389/fonc.2019.00130
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# 1) Contenitore dei dati di una coorte
# =============================================================================
@dataclass
class Cohort:
    """Tutto ciò che serve per addestrare PACE.

    dose     : array (N, M) -> dose della mappa i nel voxel j (unità arbitrarie)
    cov      : array (N, K) -> covariate non dosimetriche. Nei dati in silico
               non ce ne sono: K = 0 (è il "PACE_NoAge" del paper).
    y        : array (N,)   -> esiti 0/1
    mask     : array booleano 2D -> quali voxel della griglia sono usati
               (qui tutti); serve per rimettere i risultati "in 2D".
    ids      : nomi delle mappe
    cov_names, cov_mean, cov_std : info sulle covariate (qui vuote)
    dose_scale : scala con cui dividiamo la dose dentro le regressioni
               (qui 1: le dosi sintetiche sono già numeri di ordine 1)
    """

    dose: np.ndarray
    cov: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    ids: list
    cov_names: list = field(default_factory=list)
    cov_mean: np.ndarray | None = None
    cov_std: np.ndarray | None = None
    dose_scale: float = 1.0

    @property
    def n_patients(self):
        return self.dose.shape[0]

    @property
    def n_voxels(self):
        return self.dose.shape[1]

    def subset(self, idx):
        """Restituisce la sotto-coorte con le mappe di indice `idx`
        (utile per dividere in addestramento e validazione)."""
        idx = np.asarray(idx)
        sub = Cohort(self.dose[idx], self.cov[idx], self.y[idx], self.mask,
                     [self.ids[i] for i in idx], self.cov_names,
                     self.cov_mean, self.cov_std, self.dose_scale)
        if hasattr(self, "rs_true"):
            sub.rs_true = self.rs_true
        return sub


def to_volume(values, mask, fill=np.nan):
    """Rimette un vettore di M valori (uno per voxel) nella griglia 2D.
    Serve per disegnare le "mappe" (es. la mappa dei coefficienti, la P-map
    o la W-map della Figura 3)."""
    vol = np.full(mask.shape, fill, dtype=float)
    vol[mask] = values
    return vol


# =============================================================================
# 2) Dati sintetici ("In silico Validation" del paper)
# =============================================================================
def make_synthetic_cohort(n_patients=200, grid=40, n_volume=0.25, rs="s2g",
                          event_rate=0.5, seed=0):
    """Genera una coorte sintetica 2D come nel paper (Figure 4-6).

    - Ogni mappa di dose D_i(x_j) è la somma di 1-4 picchi gaussiani con
      deviazione standard tra il 10% e il 30% del lato della regione e
      altezza tra 0.5 e 1.5 (unità arbitrarie).
    - Si sceglie una mappa di radio-sensibilità RS(x_j): omogenea ("hom")
      o somma di due gaussiane traslate ("s2g", Figura 5A).
    - Per ogni mappa si calcola il g^2EUD (Equazione 4 del paper):
          g2EUD = [ sum_j D^(1/n)(x_j) RS(x_j) / sum_j RS(x_j) ]^n
    - L'esito è O_i = [g2EUD_i > D_th]; qui D_th è scelto come quantile per
      avere la frazione di eventi `event_rate`.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, grid)
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    D = np.zeros((n_patients, grid, grid))
    for i in range(n_patients):
        for _ in range(rng.integers(1, 5)):
            cx, cy = rng.uniform(0, 1, 2)
            sig = rng.uniform(0.10, 0.30)
            h = rng.uniform(0.5, 1.5)
            D[i] += h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig ** 2))

    if rs == "hom":
        RS = np.ones((grid, grid))
    else:  # due gaussiane traslate lungo un asse, come in Figura 5A
        RS = (np.exp(-((X - 0.3) ** 2 + (Y - 0.5) ** 2) / (2 * 0.12 ** 2)) +
              np.exp(-((X - 0.7) ** 2 + (Y - 0.5) ** 2) / (2 * 0.12 ** 2)))

    Df = D.reshape(n_patients, -1)
    RSf = RS.ravel()
    g2eud = ((Df ** (1.0 / n_volume)) @ RSf / RSf.sum()) ** n_volume
    d_th = np.quantile(g2eud, 1 - event_rate)
    y = (g2eud > d_th).astype(np.float64)

    mask = np.ones((grid, grid), dtype=bool)
    coh = Cohort(Df.astype(np.float32), np.zeros((n_patients, 0)), y, mask,
                 [f"syn_{i}" for i in range(n_patients)], [], np.zeros(0),
                 np.ones(0), dose_scale=1.0)
    coh.rs_true = RS  # la "verità" sulla radiosensibilità, per confronto
    return coh


# =============================================================================
# 3) Metriche (Tabella 1 del paper)
# =============================================================================
def auc_score(y, p):
    """Area sotto la curva ROC, calcolata con la statistica di Mann-Whitney:
    è la probabilità che una mappa con evento abbia una predizione più alta
    di una mappa senza evento."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    from scipy.stats import rankdata
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def calibration(y, p, n_groups=7):
    """Curva di calibrazione come in Figura 9B-D: si raggruppano le mappe
    per probabilità predetta, si confronta la frequenza osservata con la
    probabilità media predetta e si fa un fit lineare y = a + b x.
    Una calibrazione ideale ha a = 0, b = 1."""
    order = np.argsort(p)
    groups = np.array_split(order, n_groups)
    xm = np.array([p[g].mean() for g in groups])
    ym = np.array([y[g].mean() for g in groups])
    b, a = np.polyfit(xm, ym, 1)
    r2 = 1 - np.sum((ym - (a + b * xm)) ** 2) / max(np.sum((ym - ym.mean()) ** 2), 1e-12)
    return {"intercept": a, "slope": b, "R2": r2, "x": xm, "y": ym}


def classification_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    pos, neg = max(tp + fn, 1), max(tn + fp, 1)
    out = {
        "AUC": auc_score(y, p),
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (tp / pos + tn / neg),
        "F1": 2 * tp / max(2 * tp + fp + fn, 1),
        "brier": float(np.mean((np.asarray(p) - y) ** 2)),
    }
    try:
        c = calibration(y, np.asarray(p))
        out.update({"calib_intercept": c["intercept"], "calib_slope": c["slope"], "calib_R2": c["R2"]})
    except Exception:
        pass
    return out


def print_metrics(title, m):
    print(f"--- {title} ---")
    for k, v in m.items():
        print(f"  {k:18s} {v: .3f}")
