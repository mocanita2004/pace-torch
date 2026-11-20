# pace-torch

Reimplementazione in **PyTorch**, con la programmazione differenziabile, del modello
**PACE** (*Probabilistic Atlas for normal tissue Complication Estimation*):

> G. Palma, S. Monti, A. Buonanno, R. Pacelli, L. Cella,
> "PACE: A Probabilistic Atlas for Normal Tissue Complication Estimation in Radiation Oncology",
> *Frontiers in Oncology* 9:130 (2019). [doi:10.3389/fonc.2019.00130](https://doi.org/10.3389/fonc.2019.00130)

Il codice riproduce la validazione **in silico** dell'articolo: genera mappe di dose
sintetiche 2D con una radiosensibilità eterogenea nota, addestra PACE e ne valuta le
prestazioni. È stato sviluppato per la tesi di laurea triennale
*[titolo della tesi]*, [Nome Cognome], [Università], [anno].

## Il modello in breve

1. **Fase 1** – per ogni voxel $x_j$ una regressione logistica dell'esito sulla dose
   locale, $P_{ij} = \sigma(a_j + b_j D_{ij})$, risolta con il metodo di Newton
   (gradiente ed hessiana dalla differenziazione automatica, `torch.func`).
2. **Fase 2** – per ogni mappa, una P-map ($P_{ij}$) e una W-map
   ($W_{ij} = 1/\mathrm{CI}_{95}(P_{ij})$, con $W_{ij}=0$ dove $b_j \le 0$).
3. **Fase 3** – la gEUp riassume le mappe e la probabilità finale è
   $\mathrm{PACE}_i = \Phi\big((\mathrm{gEUp}_i - T_{p50})/(\mu\,T_{p50})\big)$;
   i parametri $\nu, \mu, T_{p50}$ sono stimati con Adam e poi rifiniti con Newton.

Gli intervalli di confidenza dei parametri sono calcolati sia con il metodo di Wald
sia con il profilo di verosimiglianza.

## File

| File | Contenuto |
|---|---|
| `pace_torch.py` | il modello PACE (fasi 1–3), gli intervalli di confidenza, le figure e il programma principale |
| `pace_common.py` | generazione dei dati sintetici e metriche di prestazione (AUC, accuratezza, F1, calibrazione); solo NumPy |

## Installazione

Serve Python 3.9 o successivo.

```bash
pip install -r requirements.txt
```

Il codice usa la GPU se disponibile (CUDA), altrimenti la CPU.

## Uso

```bash
python pace_torch.py                              # 400 mappe sintetiche, n = 0.5
python pace_torch.py --n-synthetic 1000 --n-volume 0.25
```

| Opzione | Default | Significato |
|---|---|---|
| `--n-synthetic` | 400 | numero di mappe di dose sintetiche (75% addestramento, 25% validazione) |
| `--n-volume` | 0.5 | parametro $n$ della g$^2$EUD usata per generare gli esiti |
| `--ridge` | 0.01 | penalità ridge sulle regressioni della fase 1 |
| `--out` | `risultati_torch` | cartella in cui salvare i risultati |

## Output

Sul terminale: parametri stimati, metriche di training e validazione, intervalli di
confidenza al 95% (Wald e profilo di verosimiglianza).

Nella cartella di output:

| File | Contenuto |
|---|---|
| `pace_torch_rs.png` | mappa di radiosensibilità vera usata per generare gli esiti |
| `pace_torch_pazienti.png` | dose, P-map e W-map di 9 mappe di validazione, con esiti veri e PACE predetta |
| `pace_torch_maps.png` | coefficiente $b_j$, deviazione standard di $P$ e peso medio $\overline{W}$ in ogni voxel |
| `pace_torch_results.npz` | tutti i risultati numerici (coefficienti, mappe, parametri) |

## Differenze rispetto all'articolo

- Alle regressioni della fase 1 si aggiunge una piccola penalità ridge (non presente
  nell'articolo), per evitare coefficienti infiniti nei voxel con dati separabili.
- Questa versione contiene solo la parte in silico: la lettura dei dati clinici e il
  Leave-One-Out non sono inclusi.

## Licenza

Distribuito con licenza MIT: vedi il file [`LICENSE`](LICENSE).
