"""
================================================================================
  NFW MINI-HALO GRAVITATIONAL BIAS CORRECTION PIPELINE  —  GOLD STANDARD (v6)
================================================================================

Objetivo Científico
-------------------
Corrigir a suavização numérica da cúspide central (ρ ∝ r⁻¹) em resolvedores
de Poisson via FFT/CIC que "apagam" galáxias limítrofes segundo o Critério
de Nadler (2025) [Ref. 3]. Demonstrar que a correção PCHIP radial no espaço
real é robusta para qualquer perfil de cúspide r⁻¹ (NFW e Hernquist).

Novidades Gold Standard vs. v5
-------------------------------
  [G1] Perfil de Hernquist — densidade, potencial, massa acumulada analíticos
  [G2] Teste de Sensibilidade — PCHIP calibrado em NFW aplicado ao Hernquist
  [G3] Monte Carlo expandido — 100 halos (dobro do v5)
  [G4] Fração de Ocupação de Galáxias (f_occ) — curva logística em V_max
  [G5] Figura 2 aprimorada — zona de recuperação física destacada
  [G6] CSV de auditoria — 20 colunas por halo

Referências
-----------
[1] Łokas & Mamon (2001). MNRAS 321, 155.       — Potencial NFW analítico
[2] Navarro, Frenk & White (1997). ApJ 490, 493. — Perfil NFW
[3] Nadler et al. (2025). ApJ [in press].        — Critério dual V_max + score
[4] Hockney & Eastwood (1988). CRC Press.        — CIC + FFT de Poisson
[5] Fritsch & Carlson (1980). SIAM J. Numer. 17. — PCHIP monótono
[6] Hernquist (1990). ApJ 356, 359.              — Perfil de Hernquist
[7] Vale & Ostriker (2004). MNRAS 353, 189.      — Função de ocupação de halos

Estrutura do Pipeline (10 etapas)
-----------------------------------
  1.  Validação analítica NFW + Hernquist
  2.  Configuração + smoke test do solver
  3.  Mapeamento ε(r, c) — 100 translações × 5 concentrações
  4.  Construção do CorretorPCHIP 2D(r, c)
  5.  Validação cruzada NFW N_ref vs N_cal
  6.  Teste de Sensibilidade — Hernquist (transferabilidade do corretor)
  7.  Monte Carlo 100 halos — zona de transição [10⁷, 10⁸·⁵] M☉
  8.  Fração de Ocupação de Galáxias (f_occ) analítica vs corrigida
  9.  Figuras 1 e 2 (nível paper)
  10. Exportação CSV de auditoria

Parâmetros de Execução
-----------------------
  Dev (rápido, ~10 min)  : N_MAP=64,  N_TRANS=10,  M_SUB=4,  N_HALOS=30
  Publicação (~3–6 h)    : N_MAP=128, N_TRANS=100, M_SUB=8,  N_HALOS=100

Saídas Geradas
--------------
  fig1_diagnostico_gold.png   — Diagnóstico 6-painéis
  fig2_punch_gold.png         — The Punch aprimorado
  fig3_focc_gold.png          — Fração de Ocupação de Galáxias
  fig4_hernquist_gold.png     — Teste de Sensibilidade Hernquist
  resultados_mc_gold.csv      — Tabela de auditoria completa
================================================================================
"""

# ── Compatibilidade de encoding ────────────────────────────────────────────────
import sys, io
for _sn in ("stdout", "stderr"):
    _s = getattr(sys, _sn)
    if hasattr(_s, "buffer"):
        _enc = getattr(_s, "encoding", "") or ""
        if _enc.lower() not in ("utf-8", "utf8"):
            setattr(sys, _sn, io.TextIOWrapper(
                _s.buffer, encoding="utf-8", errors="replace", line_buffering=True))

import os, gc, csv, time, pathlib, warnings
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════════
# § A — CONSTANTES GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════
RNG_SEED     = 42
CONCS_MAP    = [5, 10, 15, 20, 30]   # concentrações de calibração PCHIP
CAIXA_L      = 2.0                   # domínio [-1, +1] r_vir por eixo

# Critério de Nadler (2025) [Ref. 3]
R_EVAL       = 0.10    # raio de avaliação do potencial [r_vir]
SCORE_LIM    = 0.85    # score mínimo para classificação estelar
VMAX_LIM_KMS = 12.0   # V_max mínimo físico [km/s] — limiar de reionização

# Cosmologia plana ΛCDM, z=0
H0, h        = 70.0, 0.70
RHO_CRIT     = 2.775e11 * h**2   # M☉/Mpc³
DELTA_C      = 200.0              # sobredensidade virial
G_MPC        = 4.302e-3 * 1e-6   # Mpc M☉⁻¹ (km/s)²

# Paleta (tema cosmológico escuro, acessível)
CORES_C = {5:"#4e79a7", 10:"#f28e2b", 15:"#59a14f", 20:"#e15759", 30:"#b07aa1"}
BG0, BG1, BDR = "#0d1117", "#161b22", "#30363d"
TXT, ACC      = "#e6edf3", "#58a6ff"
RED, GRN, ORG = "#f85149", "#3fb950", "#f0883e"
PRP           = "#bc8cff"   # roxo para Hernquist


# ══════════════════════════════════════════════════════════════════════════════
# § B — PERFIS ANALÍTICOS
# ══════════════════════════════════════════════════════════════════════════════

class NFWAnalitico:
    """
    Perfil NFW em unidades adimensionais: G = M_vir = r_vir = 1.

    Potencial (Łokas & Mamon 2001, Eq. 14) [Ref. 1]:
        Φ(r) = −ln(1 + c·r) / (r · f(c))
        f(c) = ln(1+c) − c/(1+c)

    Expansão de Taylor 5ª ordem para r → 0 (cúspide estável):
        Φ(r→0) = −c/f(c)  [finito]
    """
    _TAYLOR = 1e-6

    def __init__(self, c: float):
        self.c    = float(c)
        self.rs   = 1.0 / c
        self.fc   = np.log(1.0 + c) - c / (1.0 + c)
        self.rho_s = 1.0 / (4.0 * np.pi * self.rs**3 * self.fc)
        self.nome  = "NFW"

    def densidade(self, r):
        r = np.asarray(r, dtype=float)
        x = np.where(r < 1e-14, 1e-14, r) / self.rs
        return self.rho_s / (x * (1.0 + x)**2)

    def potencial(self, r):
        r   = np.asarray(r, dtype=float)
        phi = np.empty_like(r)
        t   = r < self._TAYLOR
        re  = r[~t]
        if re.size:
            phi[~t] = -np.log(1.0 + self.c * re) / (re * self.fc)
        rt = r[t]
        if rt.size:
            u = self.c * rt
            phi[t] = -(self.c / self.fc) * (
                1.0 - u/2 + u**2/3 - u**3/4 + u**4/5)
        return phi

    def massa_acumulada(self, r):
        cr = self.c * np.asarray(r, dtype=float)
        return (np.log(1.0 + cr) - cr / (1.0 + cr)) / self.fc

    def vmax_adim(self):
        r  = np.linspace(0.01 * self.rs, 3.0, 4000)
        vc = np.sqrt(self.massa_acumulada(r) / r)
        return float(vc.max())

    def __repr__(self):
        return f"NFWAnalitico(c={self.c:.1f})"


class HernquistAnalitico:
    """
    Perfil de Hernquist (1990) em unidades adimensionais: G = M_vir = r_vir = 1.
    [Ref. 6, Eqs. 2, 10, 14]

    Densidade:
        ρ(r) = M / (2π) · a / [r (r+a)³]
        cúspide interna: ρ ∝ r⁻¹  — idêntica ao NFW em r → 0

    Potencial:
        Φ(r) = −G M / (r + a)       [analítico exato, sem singularidade]

    Massa acumulada:
        M(<r) = M · r² / (r+a)²

    Parâmetros
    ----------
    a : float — raio de escala de Hernquist [r_vir].
        Relação empírica com concentração NFW (Springel et al. 2005):
            a ≈ r_vir / (c · sqrt(2 · (ln(1+c) − c/(1+c))))
        Por simplicidade usamos a = r_vir / (c · k_h) com k_h=2.16.
    """
    def __init__(self, c: float):
        self.c    = float(c)
        self.a    = 1.0 / (c * 2.16)   # raio de escala ≈ equiparado ao NFW
        self.nome = "Hernquist"

    def densidade(self, r):
        r = np.asarray(r, dtype=float)
        r = np.where(r < 1e-14, 1e-14, r)
        a = self.a
        return 1.0 / (2.0 * np.pi) * a / (r * (r + a)**3)

    def potencial(self, r):
        r = np.asarray(r, dtype=float)
        return -1.0 / (np.where(r < 1e-14, 1e-14, r) + self.a)

    def massa_acumulada(self, r):
        r = np.asarray(r, dtype=float)
        return r**2 / (r + self.a)**2

    def vmax_adim(self):
        r  = np.linspace(0.001 * self.a, 3.0, 4000)
        vc = np.sqrt(self.massa_acumulada(r) / r)
        return float(vc.max())

    def __repr__(self):
        return f"HernquistAnalitico(c={self.c:.1f}, a={self.a:.4f})"


# ── Física cosmológica ────────────────────────────────────────────────────────

def Mvir_para_Vvir(M_sol):
    """V_vir(M_vir) em km/s para z=0, Ωm=0.3, h=0.70."""
    r_vir = (3.0 * M_sol / (4.0 * np.pi * DELTA_C * RHO_CRIT))**(1/3)
    return float(np.sqrt(G_MPC * M_sol / r_vir))

def Vmax_fisico(perfil, V_vir_kms):
    """V_max físico [km/s] = V_max_adim × V_vir."""
    return perfil.vmax_adim() * V_vir_kms

def score_phi(phi_val, phi_ref):
    """Score de profundidade: |Φ(R_EVAL)| / |Φ_ref|. (Nadler 2025) [Ref. 3]"""
    return abs(phi_val) / abs(phi_ref)

# Referência universal — calculada na importação
_NFW_REF = NFWAnalitico(10)
PHI_REF  = float(_NFW_REF.potencial(np.array([R_EVAL]))[0])


# ══════════════════════════════════════════════════════════════════════════════
# § C — DEPOSIÇÃO CIC (loop iterativo, sem tensor 6D)
# ══════════════════════════════════════════════════════════════════════════════

def cic_loop(perfil, N, deslocamento, M_sub=8):
    """
    Cloud-In-Cell analítico com M_sub³ sub-voxels por célula.

    Funciona com qualquer objeto que implemente .densidade(r).

    Gestão de memória
    -----------------
    O tensor 6D (N³ × M_sub³) seria inviável (N=128, M=8 → ~512 GB).
    O loop iterativo mantém pico em ~2 × N³ × 8 bytes:
        N=128, M_sub=8  → ~34 MB
        N=256, M_sub=8  → ~270 MB   ← viável em 8 GB RAM

    Parâmetros
    ----------
    perfil       : NFWAnalitico | HernquistAnalitico — deve ter .densidade(r)
    N            : int — resolução da grade
    deslocamento : ndarray (3,) — deslocamento do centro [r_vir]
    M_sub        : int — subdivisões por eixo (8 para publicação)
    """
    dx    = CAIXA_L / N
    dx_s  = dx / M_sub
    coords = -1.0 + (np.arange(N) + 0.5) * dx
    off    = (np.arange(M_sub) + 0.5) * dx_s - dx / 2.0
    peso   = (dx_s / dx)**3
    cx, cy, cz = deslocamento
    rho = np.zeros((N, N, N), dtype=np.float64)

    nome = getattr(perfil, 'nome', 'perfil')
    print(f"    [CIC-{nome}] N={N}, M_sub={M_sub} ({M_sub**3} sv/cél) ...",
          end=" ", flush=True)
    t0 = time.perf_counter()

    for si in off:
        xi = coords + si - cx
        for sj in off:
            yj = coords + sj - cy
            for sk in off:
                zk = coords + sk - cz
                r3d = np.sqrt(xi[:,None,None]**2
                             + yj[None,:,None]**2
                             + zk[None,None,:]**2)
                rho += perfil.densidade(r3d) * peso

    print(f"OK ({time.perf_counter()-t0:.1f}s)")
    return rho


# ══════════════════════════════════════════════════════════════════════════════
# § D — SOLVER DE POISSON 3D VIA FFT
# ══════════════════════════════════════════════════════════════════════════════

def poisson_fft(rho, L=CAIXA_L):
    """
    Resolve ∇²Φ = 4πGρ (G=1) em caixa periódica via rfftn.

    A chamada irfftn usa axes=(0,1,2) explícito — compatível com NumPy ≥2.0,
    elimina DeprecationWarning conforme nota dos logs v5. [Ref. 4]
    """
    N  = rho.shape[0]
    dx = L / N
    print(f"    [FFT] Poisson N={N} ...", end=" ", flush=True)
    t0 = time.perf_counter()

    rho_k = np.fft.rfftn(rho - rho.mean())
    kx    = (2*np.pi/L) * np.fft.fftfreq(N, d=1.0/N)
    kz    = (2*np.pi/L) * np.fft.rfftfreq(N, d=1.0/N)
    k2    = kx[:,None,None]**2 + kx[None,:,None]**2 + kz[None,None,:]**2
    k2s   = k2.copy(); k2s[0,0,0] = 1.0
    phi_k = -4.0 * np.pi * rho_k / k2s
    phi_k[0,0,0] = 0.0

    # axes explícito: compatibilidade NumPy 2.0+ (elimina DeprecationWarning)
    phi = np.fft.irfftn(phi_k, s=(N, N, N), axes=(0, 1, 2))

    # Shift de referência: Φ = 0 na casca externa (r > 0.85√3 r_vir)
    coords = -1.0 + (np.arange(N) + 0.5) * dx
    rg = np.sqrt(coords[:,None,None]**2
               + coords[None,:,None]**2
               + coords[None,None,:]**2)
    borda = rg > 0.85 * np.sqrt(3.0)
    if borda.any():
        phi -= phi[borda].mean()

    print(f"OK ({time.perf_counter()-t0:.1f}s)")
    return phi


# ══════════════════════════════════════════════════════════════════════════════
# § E — PIPELINE CIC → FFT
# ══════════════════════════════════════════════════════════════════════════════

def phi_num(perfil, N, deslocamento, M_sub=8):
    """
    Calcula (Φ_num, r_grid) para qualquer perfil analítico.

    Retorna
    -------
    phi    : ndarray (N,N,N) — potencial numérico [V_vir²]
    r_grid : ndarray (N,N,N) — distância radial de cada voxel [r_vir]
    """
    dx  = CAIXA_L / N
    rho = cic_loop(perfil, N, deslocamento, M_sub)
    phi = poisson_fft(rho)
    del rho; gc.collect()
    cx, cy, cz = deslocamento
    coords = -1.0 + (np.arange(N) + 0.5) * dx
    rg = np.sqrt((coords-cx)[:,None,None]**2
               + (coords-cy)[None,:,None]**2
               + (coords-cz)[None,None,:]**2)
    return phi, rg


# ══════════════════════════════════════════════════════════════════════════════
# § F — MAPEAMENTO DO ERRO RADIAL ε(r, c)
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ic(amostra, n_boot=2000, ci=95, seed=None):
    """
    Intervalo de confiança (percentil) por bootstrap não-paramétrico.

    Retorna (lo, hi) para o intervalo de confiança `ci`% da média da
    amostra 1D fornecida (NaNs são ignorados). Atende à recomendação de
    que erros pontuais (ex.: 0,51%) fossem acompanhados de intervalos
    de confiança, e não apenas do desvio padrão entre translações (que
    mede apenas a variância de fase de grade, não a incerteza total).
    """
    amostra = np.asarray(amostra, dtype=float)
    amostra = amostra[np.isfinite(amostra)]
    if amostra.size < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n   = amostra.size
    idx = rng.integers(0, n, size=(n_boot, n))
    medias = amostra[idx].mean(axis=1)
    a = (100 - ci) / 2.0
    return (float(np.percentile(medias, a)), float(np.percentile(medias, 100-a)))


def mapear_erro(c, N=128, Ntr=100, M_sub=8, seed=RNG_SEED):
    """
    Calcula ε̄(r) = média de (Φ_num − Φ_ana)/|Φ_ana| sobre Ntr translações.

    A média sobre translações aleatórias elimina a anisotropia de grade
    CIC, produzindo um perfil de viés isótropo. (nota [16/712])

    Além da média e do desvio padrão entre translações (que mede apenas
    a variância de fase de grade, não a incerteza total do método), esta
    versão também reporta o erro MÁXIMO e o percentil 95 (|ε|) por casca
    radial, e um intervalo de confiança de 95% por bootstrap.

    Parâmetros
    ----------
    c    : concentração NFW (calibração usa somente NFW)
    Ntr  : 100 translações para publicação
    M_sub: 8 sub-voxels para publicação

    Retorna dict com c, N, r_bins, r_cent, erro_medio, erro_std,
    erro_max, erro_p95, erro_ic95_lo/hi (por casca), erros_acum
    """
    rng    = np.random.default_rng(seed)
    nfw    = NFWAnalitico(c)
    dx     = CAIXA_L / N
    r_min  = 0.5 * dx
    r_bins = np.logspace(np.log10(r_min), 0.0, 36)
    r_cent = 0.5 * (r_bins[:-1] + r_bins[1:])
    erros  = np.full((Ntr, len(r_cent)), np.nan)

    print(f"\n  [mapa_erro] c={c:2d} | N={N} | {Ntr} transl. | M_sub={M_sub}",
          flush=True)
    for i in range(Ntr):
        desl = rng.uniform(-0.4*dx, 0.4*dx, size=3)
        print(f"    {i+1:3d}/{Ntr}", end="\r", flush=True)
        Pn, rg = phi_num(nfw, N, desl, M_sub)
        Pa     = nfw.potencial(rg.ravel()).reshape(rg.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = np.abs(Pa)
            eps   = np.where(denom > 1e-10, (Pn-Pa)/denom, np.nan)
            eps   = np.where(np.isfinite(eps), eps, np.nan)
        rf, ef = rg.ravel(), eps.ravel()
        for j,(r0,r1) in enumerate(zip(r_bins[:-1], r_bins[1:])):
            m = (rf >= r0) & (rf < r1)
            if m.any(): erros[i,j] = np.nanmean(ef[m])
        del Pn, rg, Pa, eps; gc.collect()

    print()
    erro_medio   = np.nanmean(erros, axis=0)
    erro_std     = np.nanstd(erros,  axis=0)
    erro_abs_max = np.nanmax(np.abs(erros), axis=0)
    erro_abs_p95 = np.nanpercentile(np.abs(erros), 95, axis=0)
    ic_lo = np.full(len(r_cent), np.nan)
    ic_hi = np.full(len(r_cent), np.nan)
    for j in range(len(r_cent)):
        ic_lo[j], ic_hi[j] = bootstrap_ic(erros[:, j], n_boot=1000, seed=seed+j)

    return dict(c=c, N=N, r_bins=r_bins, r_cent=r_cent,
                erro_medio=erro_medio,
                erro_std  =erro_std,
                erro_max  =erro_abs_max,
                erro_p95  =erro_abs_p95,
                erro_ic95_lo=ic_lo, erro_ic95_hi=ic_hi,
                erros_acum=erros)


# ══════════════════════════════════════════════════════════════════════════════
# § G — CORRETOR PCHIP 2D(r, c)  [Ref. 5]
# ══════════════════════════════════════════════════════════════════════════════

class CorretorPCHIP:
    """
    Corretor de viés gravitacional baseado em interpolação PCHIP monotônica.

    Calibração: somente em perfis NFW (o teste de sensibilidade verifica
    transferabilidade ao Hernquist sem recalibração).

    Fórmula de correção (v5/Gold):
        ε(r,c) = (Φ_num − Φ_ana) / |Φ_ana|     [viés relativo]
        Φ_corr = Φ_num / clip(1 − ε̂, 0.05, 2.0) [algebricamente exato]

    PCHIP vs. spline cúbica natural [Ref. 5]:
        • Monótono: sem oscilações de Runge na cúspide
        • C¹ contínuo: derivadas suaves
        • Respeita monotonicidade local de ε(r)
    """

    def __init__(self, resultados):
        self.concs  = sorted(d["c"] for d in resultados)
        self._pchip = {}
        for d in resultados:
            c = d["c"]
            r = d["r_cent"].copy()
            e = d["erro_medio"].copy()
            ok = np.isfinite(e)
            if ok.sum() >= 4:
                e = np.where(ok, e, np.interp(r, r[ok], e[ok]))
            else:
                e = np.zeros_like(e)
            r_ext = np.concatenate([[0.0], r,    [1.5, 6.0]])
            e_ext = np.concatenate([[e[0]],e,    [0.0, 0.0]])
            self._pchip[c] = PchipInterpolator(r_ext, e_ext, extrapolate=False)
        print(f"  [CorretorPCHIP] c = {self.concs}")

    def fator(self, r, c):
        """ε̂(r, c) — interpolação linear entre concentrações calibradas."""
        r     = np.asarray(r, dtype=float)
        c_arr = np.array(self.concs, dtype=float)
        def _ev(cv):
            v = self._pchip[cv](r)
            return np.where(np.isfinite(v), v, 0.0)
        if c <= c_arr[0]:  return _ev(c_arr[0])
        if c >= c_arr[-1]: return _ev(c_arr[-1])
        idx = int(np.searchsorted(c_arr, c)) - 1
        c0, c1 = c_arr[idx], c_arr[idx+1]
        t = (c - c0) / (c1 - c0)
        return (1-t)*_ev(c0) + t*_ev(c1)

    def corrigir(self, Pn, rg, c):
        """Φ_corr = Φ_num / clip(1 − ε̂, 0.05, 2.0)"""
        eps   = self.fator(rg, c)
        denom = np.clip(1.0 - eps, 0.05, 2.0)
        return Pn / denom

    def score_corrigido(self, phi_n_val, r_val, c):
        eps   = float(self.fator(np.array([r_val]), c)[0])
        denom = float(np.clip(1.0 - eps, 0.05, 2.0))
        return score_phi(phi_n_val / denom, PHI_REF)


# ══════════════════════════════════════════════════════════════════════════════
# § H — VALIDAÇÃO CRUZADA NFW
# ══════════════════════════════════════════════════════════════════════════════

def validacao_cruzada(corretor, c_teste=10.0, N_ref=128, N_cal=64,
                      M_sub=8, seed=123):
    """
    Compara 4 perfis: analítico, N_ref, N_cal (sem corr.), N_cal + PCHIP.
    Objetivo: verificar que N_cal + PCHIP ≈ N_ref ≈ analítico.
    """
    rng  = np.random.default_rng(seed)
    nfw  = NFWAnalitico(c_teste)
    desl = rng.uniform(-0.05, 0.05, size=3)
    print(f"\n  [val_cruzada] c={c_teste} | N_ref={N_ref} | N_cal={N_cal}")

    r_ana = np.logspace(-2.5, 0.0, 400)
    phi_a = nfw.potencial(r_ana)

    print(f"    Grade referência N={N_ref}:")
    Pr, rr = phi_num(nfw, N_ref, desl, M_sub)
    print(f"    Grade calibrada N={N_cal}:")
    Pc, rc = phi_num(nfw, N_cal, desl, M_sub)
    Pcc    = corretor.corrigir(Pc, rc, c_teste)

    dx_c   = CAIXA_L / N_cal
    r_bins = np.logspace(np.log10(0.5*dx_c), 0.0, 28)
    r_cent = 0.5 * (r_bins[:-1] + r_bins[1:])

    def _bin(campo, grade):
        rf, pf = grade.ravel(), campo.ravel()
        med = np.full(len(r_cent), np.nan)
        for j,(r0,r1) in enumerate(zip(r_bins[:-1], r_bins[1:])):
            m = (rf>=r0)&(rf<r1)
            if m.any(): med[j] = np.mean(pf[m])
        return med

    perfil_ana_b = nfw.potencial(r_cent)
    perfil_cal   = _bin(Pc, rc)
    perfil_corr  = _bin(Pcc, rc)

    # Erro pontual (por casca radial) antes/depois — permite reportar
    # o erro MÁXIMO e o percentil 95, não apenas a média integrada sobre
    # todos os raios, que pode mascarar um erro residual elevado perto
    # do centro (crítica de avaliação, Seção 4.1).
    with np.errstate(invalid="ignore", divide="ignore"):
        msk = np.abs(perfil_ana_b) > 1e-10
        erro_pt_ant = np.where(msk, np.abs(perfil_cal - perfil_ana_b) /
                                np.abs(perfil_ana_b) * 100, np.nan)
        erro_pt_dep = np.where(msk, np.abs(perfil_corr - perfil_ana_b) /
                                np.abs(perfil_ana_b) * 100, np.nan)

    # Erro especificamente no raio de avaliação de Nadler (r_eval), que é
    # o valor fisicamente relevante para o Critério de Nadler — e não
    # necessariamente igual ao erro médio integrado sobre todos os raios.
    i_reval = int(np.argmin(np.abs(r_cent - R_EVAL)))

    res = dict(r_ana=r_ana, phi_ana=phi_a, r_cent=r_cent,
               perfil_ref  =_bin(Pr, rr),
               perfil_cal  =perfil_cal,
               perfil_corr =perfil_corr,
               perfil_ana_b=perfil_ana_b,
               erro_pt_ant=erro_pt_ant, erro_pt_dep=erro_pt_dep,
               erro_max_ant=float(np.nanmax(erro_pt_ant)),
               erro_max_dep=float(np.nanmax(erro_pt_dep)),
               erro_p95_ant=float(np.nanpercentile(erro_pt_ant, 95)),
               erro_p95_dep=float(np.nanpercentile(erro_pt_dep, 95)),
               erro_reval_ant=float(erro_pt_ant[i_reval]),
               erro_reval_dep=float(erro_pt_dep[i_reval]),
               c=c_teste, N_ref=N_ref, N_cal=N_cal)
    del Pr, rr, Pc, Pcc, rc; gc.collect()
    return res


# ══════════════════════════════════════════════════════════════════════════════
# § I — TESTE DE SENSIBILIDADE: HERNQUIST  [G2]
# ══════════════════════════════════════════════════════════════════════════════

def teste_sensibilidade_hernquist(corretor, c_equiv=10.0, N=64,
                                  M_sub=4, seed=77):
    """
    Aplica o corretor PCHIP (calibrado em NFW) ao perfil de Hernquist.

    Motivação científica [Ref. 6, nota 16/640, 16/648]:
    ---------------------------------------------------
    Tanto NFW quanto Hernquist possuem cúspide ρ ∝ r⁻¹. O viés CIC-FFT
    origina-se desta singularidade, não da forma global do perfil.
    Logo, um corretor calibrado em NFW deve reduzir o viés no Hernquist
    de forma similar — demonstrando robustez e generalidade do método.

    Este teste quantifica a "transferabilidade" do corretor:
        eficiência = (ε_antes − ε_depois) / ε_antes × 100%

    Parâmetros
    ----------
    corretor  : CorretorPCHIP calibrado em NFW
    c_equiv   : concentração NFW equivalente (para fator ε)
    N         : resolução da grade (menor para velocidade)
    M_sub     : sub-voxels CIC
    seed      : semente RNG

    Retorna dict com perfis e métricas para plotagem da Figura 4.
    """
    rng  = np.random.default_rng(seed)
    hq   = HernquistAnalitico(c_equiv)
    desl = rng.uniform(-0.05, 0.05, size=3)

    print(f"\n  [Hernquist] c_equiv={c_equiv} | N={N}")
    Pn, rg = phi_num(hq, N, desl, M_sub)
    Pc     = corretor.corrigir(Pn, rg, c_equiv)

    dx     = CAIXA_L / N
    r_bins = np.logspace(np.log10(0.5*dx), 0.0, 28)
    r_cent = 0.5 * (r_bins[:-1] + r_bins[1:])

    def _bin(campo, grade):
        rf, pf = grade.ravel(), campo.ravel()
        med = np.full(len(r_cent), np.nan)
        for j,(r0,r1) in enumerate(zip(r_bins[:-1], r_bins[1:])):
            m = (rf>=r0)&(rf<r1)
            if m.any(): med[j] = np.mean(pf[m])
        return med

    pa_b = hq.potencial(r_cent)
    pn_b = _bin(Pn, rg)
    pc_b = _bin(Pc, rg)

    r_ana  = np.logspace(-2.5, 0.0, 400)
    phi_ha = hq.potencial(r_ana)

    with np.errstate(invalid="ignore", divide="ignore"):
        msk = np.abs(pa_b) > 1e-10
        ea = np.where(msk, (pn_b - pa_b)/np.abs(pa_b)*100, np.nan)
        ed = np.where(msk, (pc_b - pa_b)/np.abs(pa_b)*100, np.nan)

    e_ant = float(np.nanmean(np.abs(ea)))
    e_dep = float(np.nanmean(np.abs(ed)))
    efic  = (1 - e_dep/max(e_ant, 1e-10)) * 100

    print(f"    Hernquist | ε antes={e_ant:.1f}% | ε depois={e_dep:.2f}% "
          f"| Eficiência={efic:.0f}%")

    del Pn, Pc, rg; gc.collect()
    return dict(hq=hq, r_ana=r_ana, phi_ha=phi_ha,
                r_cent=r_cent, pa_b=pa_b, pn_b=pn_b, pc_b=pc_b,
                ea=ea, ed=ed, e_ant=e_ant, e_dep=e_dep, efic=efic,
                c_equiv=c_equiv, N=N)


# ══════════════════════════════════════════════════════════════════════════════
# § I2 — ESTUDOS DE ROBUSTEZ ADICIONAIS
#        (convergência M_sub, sensibilidade a L_caixa, sensibilidade a Φ_ref,
#         projeção de escalabilidade para N=256)
# ══════════════════════════════════════════════════════════════════════════════

def estudo_convergencia_Msub(c=10.0, N=32, M_subs=(2, 4, 8, 16), seed=999):
    """
    Estudo de convergência do número de sub-voxels CIC por eixo (M_sub).

    Motivação: a avaliação apontou que M_sub=8 era assumido como
    suficiente sem demonstração explícita de convergência. Aqui,
    para um deslocamento sub-célula fixo, comparamos o erro relativo
    do potencial numérico (na casca central) para M_sub crescente, na
    mesma grade N. Se o erro estabilizar (variação < a variação já
    observada entre translações), M_sub=8 pode ser considerado
    suficiente; caso contrário, a convergência não está garantida.

    Retorna dict com M_subs, erro_central[%], erro_reval[%] e tempo[s].
    """
    rng  = np.random.default_rng(seed)
    nfw  = NFWAnalitico(c)
    desl = rng.uniform(-0.3, 0.3, size=3) * (CAIXA_L / N)
    dx   = CAIXA_L / N

    erros_c, erros_e, tempos = [], [], []
    print(f"\n  [convergencia M_sub] c={c} | N={N} | M_sub={list(M_subs)}")
    for Msub in M_subs:
        t0 = time.perf_counter()
        Pn, rg = phi_num(nfw, N, desl, Msub)
        dt = time.perf_counter() - t0
        Pa = nfw.potencial(rg.ravel()).reshape(rg.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = np.abs((Pn - Pa) / np.abs(Pa)) * 100
        i_c = np.unravel_index(np.argmin(rg), rg.shape)
        m_ev = (rg > 0.9*R_EVAL) & (rg < 1.1*R_EVAL)
        erros_c.append(float(eps[i_c]))
        erros_e.append(float(np.nanmean(eps[m_ev])) if m_ev.any() else np.nan)
        tempos.append(dt)
        print(f"    M_sub={Msub:2d} | ε_central={erros_c[-1]:.2f}% | "
              f"ε(r_eval)={erros_e[-1]:.2f}% | t={dt:.2f}s")
        del Pn, rg, Pa, eps; gc.collect()

    return dict(M_subs=list(M_subs), erro_central=erros_c,
                erro_reval=erros_e, tempos=tempos, N=N, c=c)


def estudo_sensibilidade_Lbox(c=10.0, N=32, M_sub=4, Ls=(2.0, 3.0, 4.0), seed=999):
    """
    Sensibilidade do erro do potencial ao tamanho da caixa periódica L
    (em unidades de r_vir). O trabalho original fixa L=2 sem testar se
    halos próximos à borda sofrem auto-interação periódica espúria.
    Aqui recalculamos ε(r_eval) do NFW mantendo N e M_sub fixos, variando
    L — se ε(r_eval) mudar significativamente com L, o valor L=2 adotado
    não é inócuo e deveria ser reportado como fonte de incerteza sistemática.
    """
    global CAIXA_L
    rng  = np.random.default_rng(seed)
    nfw  = NFWAnalitico(c)
    resultados = []
    L_original = CAIXA_L
    print(f"\n  [sensibilidade L_caixa] c={c} | N={N} | L={list(Ls)}")
    try:
        for L in Ls:
            CAIXA_L = L
            dx   = CAIXA_L / N
            desl = rng.uniform(-0.3*dx, 0.3*dx, size=3)
            Pn, rg = phi_num(nfw, N, desl, M_sub)
            Pa = nfw.potencial(rg.ravel()).reshape(rg.shape)
            m_ev = (rg > 0.9*R_EVAL) & (rg < 1.1*R_EVAL)
            with np.errstate(invalid="ignore", divide="ignore"):
                eps = np.abs((Pn - Pa) / np.abs(Pa)) * 100
            e_ev = float(np.nanmean(eps[m_ev])) if m_ev.any() else np.nan
            resultados.append(e_ev)
            print(f"    L={L:.1f} r_vir | ε(r_eval)={e_ev:.2f}%")
            del Pn, rg, Pa, eps; gc.collect()
    finally:
        CAIXA_L = L_original
    return dict(Ls=list(Ls), erro_reval=resultados, N=N, c=c)


def sensibilidade_phi_ref(c_refs=(5.0, 10.0, 15.0, 20.0, 30.0),
                           c_halo=8.0):
    """
    Quantifica o quanto o Score de Nadler de um halo fixo (c_halo) muda
    conforme a concentração de referência Φ_ref = Φ_NFW(r_eval; c_ref)
    varia — respondendo à crítica de que a escolha c_ref=10 é arbitrária
    e não fisicamente motivada, e de que halos reais têm concentração
    dependente da massa (não uma referência universal fixa).
    """
    nfw_halo = NFWAnalitico(c_halo)
    phi_halo = float(nfw_halo.potencial(np.array([R_EVAL]))[0])
    linhas = []
    for cr in c_refs:
        phi_ref_cr = float(NFWAnalitico(cr).potencial(np.array([R_EVAL]))[0])
        score = abs(phi_halo) / abs(phi_ref_cr)
        classifica = score > SCORE_LIM
        linhas.append(dict(c_ref=cr, phi_ref=phi_ref_cr, score=score,
                            estelar=classifica))
    return dict(c_halo=c_halo, phi_halo=phi_halo, linhas=linhas)


def projecao_escalabilidade(N_alvo=(32, 64, 128), M_sub=8, N_trans_referencia=100,
                             N_halos_referencia=100, c_amostra=10.0, seed=999,
                             N_projetado=256):
    """
    Projeta o custo de CPU para resoluções maiores (ex.: N=256) a partir de
    medições empíricas de tempo em pequena/média escala, ajustando uma lei
    de potência log-log t(N) ∝ N^p ao invés de assumir p=3 a priori. Isto
    evita a extrapolação puramente teórica sem qualquer calibração empírica
    — o expoente medido tipicamente excede 3 devido a efeitos de cache e
    overhead do loop de sub-amostragem CIC, tornando a extrapolação ingênua
    (N³) otimista.
    """
    rng = np.random.default_rng(seed)
    nfw = NFWAnalitico(c_amostra)
    tempos = {}
    print(f"\n  [projecao_escalabilidade] medindo tempo empírico por N ...")
    for N in N_alvo:
        dx = CAIXA_L / N
        desl = rng.uniform(-0.3*dx, 0.3*dx, size=3)
        t0 = time.perf_counter()
        Pn, rg = phi_num(nfw, N, desl, M_sub)
        dt = time.perf_counter() - t0
        tempos[N] = dt
        print(f"    N={N:4d} | t_halo={dt:.3f}s")
        del Pn, rg; gc.collect()

    Ns = np.array(sorted(tempos))
    ts = np.array([tempos[n] for n in Ns])
    if len(Ns) >= 2:
        # Ajuste log-log: log(t) = p*log(N) + b
        p, b = np.polyfit(np.log(Ns), np.log(ts), 1)
        t_proj = float(np.exp(b) * N_projetado**p)
        razao_teorica_N3 = (N_projetado / Ns[0]) ** 3
        razao_ajustada    = (N_projetado / Ns[0]) ** p
    else:
        p = b = t_proj = razao_teorica_N3 = razao_ajustada = float("nan")

    proj_min = {}
    for N in list(Ns) + [N_projetado]:
        t_N = tempos.get(N, t_proj if N == N_projetado else np.nan)
        custo_total_s = t_N * N_trans_referencia * len(CONCS_MAP) \
                        + t_N * N_halos_referencia
        proj_min[int(N)] = custo_total_s / 60.0

    print(f"    Expoente ajustado p={p:.2f} (N³ teórico assumiria p=3.00)")
    print(f"    Projeção N={N_projetado}: t_halo≈{t_proj:.1f}s | "
          f"custo total calibração+MC≈{proj_min[N_projetado]:.0f} min "
          f"({proj_min[N_projetado]/60:.1f} h)")

    return dict(tempos=tempos, expoente_ajustado=float(p),
                razao_teorica_N3=razao_teorica_N3,
                razao_ajustada=razao_ajustada,
                N_projetado=N_projetado, t_projetado=t_proj,
                projecao_min=proj_min)


# ══════════════════════════════════════════════════════════════════════════════
# § J — CRITÉRIO DE NADLER (2025)
# ══════════════════════════════════════════════════════════════════════════════

def criterio_nadler(phi_r_eval, Vmax_kms):
    """
    Classificação estelar (True) ou escura (False) pelo Critério de Nadler.

    Critério dual [Ref. 3]:
        (i)  |Φ(R_EVAL)| / |PHI_REF| > SCORE_LIM = 0.85
        (ii) V_max > VMAX_LIM_KMS = 12 km/s
    """
    sc = score_phi(phi_r_eval, PHI_REF)
    return (sc > SCORE_LIM) and (Vmax_kms > VMAX_LIM_KMS)


# ══════════════════════════════════════════════════════════════════════════════
# § K — MONTE CARLO 100 HALOS  [G3]
# ══════════════════════════════════════════════════════════════════════════════

def c_mediana_dutton_maccio14(log_M_vir, z=0.0):
    """
    Relação concentração-massa aproximada, no espírito de
    Dutton & Macciò (2014), para halos de matéria escura em z≈0:
        log10(c) = a + b * log10(M_vir / (1e12 h^-1 Msol))
    com a≈0.905, b≈-0.101 (valores de ajuste típicos para NFW em z=0;
    halos mais massivos são sistematicamente menos concentrados).
    Aqui usamos a forma funcional (dependência de massa) como referência
    qualitativa — os coeficientes exatos requerem calibração com a
    cosmologia/definição de halo do estudo, o que não foi feito aqui.
    Isto substitui a relação c~LogNormal(ln 12, 0.25) fixa e sem
    dependência de massa usada na versão anterior, que a avaliação
    apontou como não fisicamente motivada.
    """
    a, b = 0.905, -0.101
    log_c_med = a + b * (log_M_vir - 12.0)
    return log_c_med


def monte_carlo(corretor, N_halos=100, N_grade=128, M_sub=8, seed=RNG_SEED,
                 N_min_estelar=30):
    """
    Gera N_halos mini-halos NFW na zona de transição crítica e compara
    três classificações: (a) Analítico, (b) Numérico, (c) Numérico+PCHIP.

    Amostragem de massa [notas 16/649, 16/717]:
        log₁₀(M_vir) ~ U(7.0, 8.5) M☉    — zona de transição crítica

    Amostragem de concentração:
        log10(c) ~ Normal(c_mediana_dutton_maccio14(log_M), σ=0.11 dex),
        truncada em [5, 35] — relação com dependência de massa e
        dispersão log-normal (σ≈0.11 dex) consistente com a ordem de
        grandeza reportada por Dutton & Macciò (2014), substituindo a
        LogNormal(ln 12, 0.25) fixa e sem dependência de massa da versão
        anterior (apontada como não-física pela avaliação).

    Estatística da classe estelar [Seção 2.3 da avaliação]: com amostragem
    puramente uniforme em log M, apenas ~2 halos em 100 caem na classe
    "estelar" (score > SCORE_LIM), tornando a alegação de "recuperação
    100% (2 de 2)" estatisticamente frágil (IC binomial ~16–100%). Para
    reduzir esse problema SEM abandonar a amostragem original, esta
    versão faz uma AMOSTRAGEM ESTRATIFICADA: a amostra-base uniforme é
    complementada por sorteios adicionais concentrados perto do limiar
    crítico (V_max ≈ VMAX_LIM_KMS), até que pelo menos `N_min_estelar`
    halos estelares (segundo o critério analítico) sejam obtidos, ou até
    um número máximo de tentativas. O número de halos extras sorteados é
    reportado explicitamente (a amostra deixa de ser puramente aleatória
    e passa a ser estratificada por desenho, o que deve ser declarado
    no relatório, não escondido).

    Zona crítica [Ref. 3, §2.3]:
        V_vir ~ 3–10 km/s  →  V_max ~ 4–13 km/s
        O threshold de reionização (12 km/s) é cruzado ou não dependendo
        de pequenas variações em |Φ_central| — demonstrando a relevância
        física da correção de viés.
    """
    rng = np.random.default_rng(seed)

    def _sorteia(n, rng_local, log_M_lo=7.0, log_M_hi=8.5, estrato=False):
        if not estrato:
            log_M = rng_local.uniform(log_M_lo, log_M_hi, n)
        else:
            # Foco na região de massa que produz V_max próximo ao limiar
            # (halos maiores dentro da faixa crítica são mais propensos
            # a cruzar VMAX_LIM_KMS e caírem na classe estelar).
            log_M = rng_local.uniform(8.0, log_M_hi, n)
        log_c_med = c_mediana_dutton_maccio14(log_M)
        log_c = rng_local.normal(log_c_med, 0.11, n)
        c = np.clip(10.0**log_c, 5.0, 35.0)
        return log_M, c

    log_M_base, c_base = _sorteia(N_halos, rng, estrato=False)
    estrato_flag = np.zeros(N_halos, dtype=bool)

    # Classificação analítica preliminar (rápida, sem solver numérico) para
    # decidir se é necessário complementar a amostra.
    def _classifica_analitico(log_M, c):
        M = 10.0**log_M
        nfw = NFWAnalitico(c)
        V_vir = Mvir_para_Vvir(M)
        V_max = Vmax_fisico(nfw, V_vir)
        phi_a = float(nfw.potencial(np.array([R_EVAL]))[0])
        return criterio_nadler(phi_a, V_max)

    n_estelar_base = sum(_classifica_analitico(lm, cc)
                          for lm, cc in zip(log_M_base, c_base))
    print(f"\n  [MC] Amostra-base uniforme: {n_estelar_base} halo(s) "
          f"estelar(es) em {N_halos} (critério analítico).")

    log_M_extra_list, c_extra_list = [], []
    tentativas, max_tentativas = 0, 20 * N_min_estelar
    n_estelar_total = n_estelar_base
    rng_extra = np.random.default_rng(seed + 1000)
    while n_estelar_total < N_min_estelar and tentativas < max_tentativas:
        lm_e, c_e = _sorteia(1, rng_extra, estrato=True)
        tentativas += 1
        if _classifica_analitico(lm_e[0], c_e[0]):
            n_estelar_total += 1
        log_M_extra_list.append(lm_e[0]); c_extra_list.append(c_e[0])

    if log_M_extra_list:
        log_M = np.concatenate([log_M_base, np.array(log_M_extra_list)])
        c_h   = np.concatenate([c_base,   np.array(c_extra_list)])
        estrato_flag = np.concatenate([estrato_flag,
                                        np.ones(len(log_M_extra_list), dtype=bool)])
        print(f"  [MC] Amostragem estratificada: +{len(log_M_extra_list)} halo(s) "
              f"sorteados na zona crítica para atingir >= {N_min_estelar} "
              f"estelares (total agora {n_estelar_total}). "
              f"NOTA: amostra deixa de ser puramente uniforme — reportar "
              f"separadamente na análise (coluna 'estrato' no CSV).")
    else:
        log_M, c_h = log_M_base, c_base

    N_halos_efetivo = len(log_M)
    M_h = 10.0**log_M
    c_h_arr = c_h

    phi_ana  = np.zeros(N_halos_efetivo)
    phi_num_ = np.zeros(N_halos_efetivo)
    phi_corr = np.zeros(N_halos_efetivo)
    Vmax_arr = np.zeros(N_halos_efetivo)
    Vvir_arr = np.zeros(N_halos_efetivo)
    sc_ana   = np.zeros(N_halos_efetivo)
    sc_num   = np.zeros(N_halos_efetivo)
    sc_corr  = np.zeros(N_halos_efetivo)
    class_a  = np.zeros(N_halos_efetivo, dtype=bool)
    class_n  = np.zeros(N_halos_efetivo, dtype=bool)
    class_c  = np.zeros(N_halos_efetivo, dtype=bool)

    dx = CAIXA_L / N_grade
    print(f"\n  [MC] {N_halos_efetivo} halos (base={N_halos}, "
          f"extra={N_halos_efetivo-N_halos}) | N={N_grade} | M_sub={M_sub}")
    print(f"  {'Halo':>5} {'logM':>6} {'c':>5} {'Vvir':>7} {'Vmax':>7} "
          f"{'Sc_a':>6} {'Sc_n':>6} {'Sc_c':>6} {'a':>2} {'n':>2} {'c':>2} {'estr':>4}")
    print("  " + "─"*76)

    for i in range(N_halos_efetivo):
        c    = c_h_arr[i]
        M    = M_h[i]
        nfw  = NFWAnalitico(c)
        desl = rng.uniform(-0.3*dx, 0.3*dx, size=3)

        V_vir = Mvir_para_Vvir(M)
        V_max = Vmax_fisico(nfw, V_vir)
        Vvir_arr[i] = V_vir
        Vmax_arr[i] = V_max

        # (a) Analítico
        phi_a_val   = float(nfw.potencial(np.array([R_EVAL]))[0])
        phi_ana[i]  = phi_a_val
        sc_ana[i]   = score_phi(phi_a_val, PHI_REF)
        class_a[i]  = criterio_nadler(phi_a_val, V_max)

        # (b) + (c) Numérico
        Pn, rg  = phi_num(nfw, N_grade, desl, M_sub)
        Pn_corr = corretor.corrigir(Pn, rg, c)

        r_alvo = max(R_EVAL, 1.5*dx)
        marg   = max(1.5*dx, r_alvo*0.25)
        mask   = (rg > r_alvo-marg) & (rg < r_alvo+marg)
        if mask.sum() >= 8:
            phi_n_val = float(Pn[mask].mean())
            phi_c_val = float(Pn_corr[mask].mean())
        else:
            idx_f     = int(np.argmin(np.abs(rg - r_alvo)))
            phi_n_val = float(Pn.ravel()[idx_f])
            phi_c_val = float(Pn_corr.ravel()[idx_f])

        phi_num_[i] = phi_n_val
        phi_corr[i] = phi_c_val
        sc_num[i]   = score_phi(phi_n_val, PHI_REF)
        sc_corr[i]  = score_phi(phi_c_val, PHI_REF)
        class_n[i]  = criterio_nadler(phi_n_val, V_max)
        class_c[i]  = criterio_nadler(phi_c_val, V_max)

        print(f"  {i+1:5d} {log_M[i]:6.2f} {c:5.1f} {V_vir:7.2f} {V_max:7.2f} "
              f"{sc_ana[i]:6.3f} {sc_num[i]:6.3f} {sc_corr[i]:6.3f} "
              f"{int(class_a[i]):>2d} {int(class_n[i]):>2d} {int(class_c[i]):>2d} "
              f"{int(estrato_flag[i]):>4d}")

        del Pn, rg, Pn_corr; gc.collect()

    falso_escuro  = class_a & ~class_n
    recuperados   = falso_escuro & class_c
    falso_estelar = ~class_a & class_n
    pct_rec = 100.0 * recuperados.sum() / max(falso_escuro.sum(), 1)
    acc_n   = 100.0 * (class_n == class_a).sum() / N_halos_efetivo
    acc_c   = 100.0 * (class_c == class_a).sum() / N_halos_efetivo
    delta_num_pct  = np.abs(phi_ana-phi_num_)/np.abs(phi_ana)*100
    delta_corr_pct = np.abs(phi_ana-phi_corr)/np.abs(phi_ana)*100
    eps_c   = float(np.mean(delta_num_pct))
    eps_p   = float(np.mean(delta_corr_pct))
    eps_c_max, eps_c_p95 = float(np.max(delta_num_pct)), float(np.percentile(delta_num_pct, 95))
    eps_p_max, eps_p_p95 = float(np.max(delta_corr_pct)), float(np.percentile(delta_corr_pct, 95))

    # Intervalo de confiança binomial (Wilson) para a taxa de recuperação,
    # respondendo à crítica de baixa significância estatística da classe
    # estelar (a taxa de recuperação de falsos-escuros continua sendo
    # calculada sobre a subamostra de falso_escuro, mas agora o total de
    # halos estelares na amostra é >= N_min_estelar por construção).
    def _wilson_ic(k, n, z=1.96):
        if n == 0: return (float("nan"), float("nan"))
        p_hat = k / n
        denom = 1 + z**2/n
        centro = p_hat + z**2/(2*n)
        margem = z*np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))
        return ((centro-margem)/denom, (centro+margem)/denom)

    n_estelar_final = int(class_a.sum())
    ic_lo, ic_hi = _wilson_ic(int(recuperados.sum()), int(falso_escuro.sum()))

    print("\n  " + "═"*76)
    print(f"  {'Método':<32} {'Estelar':>9} {'Escuro':>9} {'Acurácia':>10}")
    print("  " + "─"*60)
    print(f"  {'Analítico (padrão ouro)':<32} {class_a.sum():>9d} "
          f"{(~class_a).sum():>9d} {'100.0%':>10}")
    print(f"  {'Numérico N=%-3d'%N_grade:<32} {class_n.sum():>9d} "
          f"{(~class_n).sum():>9d} {acc_n:>9.1f}%")
    print(f"  {'N=%-3d + PCHIP Gold'%N_grade:<32} {class_c.sum():>9d} "
          f"{(~class_c).sum():>9d} {acc_c:>9.1f}%")
    print("  " + "─"*60)
    print(f"  Halos estelares (total, base+estrato): {n_estelar_final}")
    print(f"  Falsos Escuros:    {falso_escuro.sum():>4d}")
    print(f"  RECUPERADOS:       {recuperados.sum():>4d} ({pct_rec:.0f}%)  "
          f"IC95% Wilson=[{ic_lo*100:.0f}%, {ic_hi*100:.0f}%]")
    print(f"  Erro |Φ| sem corr: média={eps_c:.1f}% máx={eps_c_max:.1f}% "
          f"P95={eps_c_p95:.1f}%")
    print(f"  Erro |Φ| c/ PCHIP: média={eps_p:.2f}% máx={eps_p_max:.2f}% "
          f"P95={eps_p_p95:.2f}%")
    print("  " + "═"*76)

    return dict(
        log_M=log_M, M_halos=M_h, c_halos=c_h_arr,
        estrato=estrato_flag,
        Vvir=Vvir_arr, Vmax=Vmax_arr,
        phi_ana=phi_ana, phi_num=phi_num_, phi_corr=phi_corr,
        sc_ana=sc_ana, sc_num=sc_num, sc_corr=sc_corr,
        class_a=class_a, class_n=class_n, class_c=class_c,
        falso_escuro=falso_escuro, recuperados=recuperados,
        falso_estelar=falso_estelar,
        N_halos=N_halos_efetivo, N_halos_base=N_halos, N_grade=N_grade,
        eps_central=eps_c, eps_corrigido=eps_p,
        eps_central_max=eps_c_max, eps_central_p95=eps_c_p95,
        eps_corrigido_max=eps_p_max, eps_corrigido_p95=eps_p_p95,
        n_estelar=n_estelar_final,
        recuperacao_ic95=(ic_lo, ic_hi),
        acc_num=acc_n, acc_corr=acc_c,
    )


# ══════════════════════════════════════════════════════════════════════════════
# § L — FRAÇÃO DE OCUPAÇÃO DE GALÁXIAS (f_occ)  [G4]
# ══════════════════════════════════════════════════════════════════════════════

def _logistica(x, x0, k):
    """Função logística: f(x) = 1 / (1 + exp(−k(x − x0)))"""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def calcular_focc(mc, N_bins=12):
    """
    Calcula a Fração de Ocupação de Galáxias (f_occ) em função de V_max.

    f_occ(V_max) = fração de halos classificados como estelares
    em cada bin de V_max, para os três métodos. [Ref. 7]

    A curva correta (analítica) deve ser uma sigmóide centrada em
    V_max ≈ VMAX_LIM_KMS. O viés FFT desloca a curva numérica para
    a direita (halos são sub-avaliados → precisam de V_max maior para
    superar o threshold). A correção PCHIP restaura a posição correta.

    Retorna
    -------
    dict com bins de V_max e f_occ para cada método + ajustes logísticos
    """
    Vm   = mc["Vmax"]
    bins = np.linspace(Vm.min()*0.9, Vm.max()*1.05, N_bins+1)
    bc   = 0.5*(bins[:-1]+bins[1:])

    def _focc_bin(cls):
        fo = np.full(N_bins, np.nan)
        for j,(v0,v1) in enumerate(zip(bins[:-1],bins[1:])):
            m = (Vm>=v0)&(Vm<v1)
            if m.sum() >= 2:
                fo[j] = cls[m].mean()
        return fo

    fo_a = _focc_bin(mc["class_a"])
    fo_n = _focc_bin(mc["class_n"])
    fo_c = _focc_bin(mc["class_c"])

    # Ajuste logístico para curvas suavizadas
    def _fit(bc, fo):
        ok = np.isfinite(fo)
        if ok.sum() < 4:
            return None, None
        try:
            p0 = [VMAX_LIM_KMS, 0.5]
            popt, _ = curve_fit(_logistica, bc[ok], fo[ok], p0=p0,
                                bounds=([5,0.01],[25,5.0]), maxfev=2000)
            return popt, _logistica(bc, *popt)
        except Exception:
            return None, None

    pa, fa_fit = _fit(bc, fo_a)
    pn, fn_fit = _fit(bc, fo_n)
    pc, fc_fit = _fit(bc, fo_c)

    return dict(bc=bc, fo_a=fo_a, fo_n=fo_n, fo_c=fo_c,
                fa_fit=fa_fit, fn_fit=fn_fit, fc_fit=fc_fit,
                pa=pa, pn=pn, pc=pc)


# ══════════════════════════════════════════════════════════════════════════════
# § M — EXPORTAÇÃO CSV  [G6]
# ══════════════════════════════════════════════════════════════════════════════

def exportar_csv(mc, caminho):
    """
    Exporta tabela de auditoria completa com 21 colunas por halo.

    Colunas:
        halo_id, log10_Mvir, Mvir_Msol, c_NFW,
        Vvir_kms, Vmax_kms,
        phi_ana, phi_num, phi_corr,
        score_ana, score_num, score_corr,
        class_ana, class_num, class_corr,
        falso_escuro, recuperado, falso_estelar,
        delta_phi_num_pct, delta_phi_corr_pct, estrato

    A coluna `estrato` marca (1) os halos adicionados pela amostragem
    estratificada complementar (usada para garantir >= N_min_estelar
    halos estelares na amostra) e (0) os halos da amostra-base uniforme
    original — permitindo ao leitor refazer a análise considerando
    apenas a amostra uniforme, se preferir.
    """
    N  = mc["N_halos"]
    cols = ["halo_id","log10_Mvir","Mvir_Msol","c_NFW",
            "Vvir_kms","Vmax_kms",
            "phi_ana","phi_num","phi_corr",
            "score_ana","score_num","score_corr",
            "class_ana","class_num","class_corr",
            "falso_escuro","recuperado","falso_estelar",
            "delta_phi_num_pct","delta_phi_corr_pct","estrato"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(N):
            pa  = mc["phi_ana"][i]
            pn  = mc["phi_num"][i]
            pc_ = mc["phi_corr"][i]
            d_n = (pn-pa)/abs(pa)*100 if abs(pa)>1e-10 else 0.0
            d_c = (pc_-pa)/abs(pa)*100 if abs(pa)>1e-10 else 0.0
            w.writerow({
                "halo_id"          : i+1,
                "log10_Mvir"       : f"{mc['log_M'][i]:.4f}",
                "Mvir_Msol"        : f"{mc['M_halos'][i]:.4e}",
                "c_NFW"            : f"{mc['c_halos'][i]:.3f}",
                "Vvir_kms"         : f"{mc['Vvir'][i]:.4f}",
                "Vmax_kms"         : f"{mc['Vmax'][i]:.4f}",
                "phi_ana"          : f"{pa:.6f}",
                "phi_num"          : f"{pn:.6f}",
                "phi_corr"         : f"{pc_:.6f}",
                "score_ana"        : f"{mc['sc_ana'][i]:.6f}",
                "score_num"        : f"{mc['sc_num'][i]:.6f}",
                "score_corr"       : f"{mc['sc_corr'][i]:.6f}",
                "class_ana"        : int(mc["class_a"][i]),
                "class_num"        : int(mc["class_n"][i]),
                "class_corr"       : int(mc["class_c"][i]),
                "falso_escuro"     : int(mc["falso_escuro"][i]),
                "recuperado"       : int(mc["recuperados"][i]),
                "falso_estelar"    : int(mc["falso_estelar"][i]),
                "delta_phi_num_pct": f"{d_n:.3f}",
                "delta_phi_corr_pct":f"{d_c:.3f}",
                "estrato"          : int(mc["estrato"][i]) if "estrato" in mc else 0,
            })
    print(f"  CSV exportado: {caminho}  ({N} halos, 21 colunas)")


def exportar_csv_robustez(caminho, conv_msub=None, sens_lbox=None,
                           sens_phiref=None, projecao=None, ambiente=None):
    """
    Exporta um CSV/relatório de robustez suplementar com os resultados dos
    novos estudos de diagnóstico (convergência M_sub, sensibilidade L_caixa,
    sensibilidade Φ_ref, projeção de escalabilidade N=256 e fingerprint do
    ambiente de execução) — atendendo diretamente às pendências listadas na
    avaliação (ausência de intervalos de confiança, de estudo de
    convergência, de teste de escalabilidade e de justificativa para
    Φ_ref(c=10)).
    """
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("# Estudos de robustez suplementares — gerados automaticamente\n")
        if ambiente:
            f.write("\n[ambiente_execucao]\n")
            for k, v in ambiente.items():
                f.write(f"{k},{v}\n")
        if conv_msub:
            f.write("\n[convergencia_M_sub]\n")
            f.write("M_sub,erro_central_pct,erro_reval_pct,tempo_s\n")
            for i, m in enumerate(conv_msub["M_subs"]):
                f.write(f"{m},{conv_msub['erro_central'][i]:.4f},"
                        f"{conv_msub['erro_reval'][i]:.4f},"
                        f"{conv_msub['tempos'][i]:.4f}\n")
        if sens_lbox:
            f.write("\n[sensibilidade_L_caixa]\n")
            f.write("L_rvir,erro_reval_pct\n")
            for i, L in enumerate(sens_lbox["Ls"]):
                f.write(f"{L},{sens_lbox['erro_reval'][i]:.4f}\n")
        if sens_phiref:
            f.write("\n[sensibilidade_phi_ref]\n")
            f.write(f"# halo de teste: c={sens_phiref['c_halo']}, "
                    f"phi_halo={sens_phiref['phi_halo']:.4f}\n")
            f.write("c_ref,phi_ref,score,classificado_estelar\n")
            for linha in sens_phiref["linhas"]:
                f.write(f"{linha['c_ref']},{linha['phi_ref']:.4f},"
                        f"{linha['score']:.4f},{int(linha['estelar'])}\n")
        if projecao:
            f.write("\n[projecao_escalabilidade]\n")
            f.write(f"# expoente ajustado p={projecao['expoente_ajustado']:.3f} "
                    f"(N^3 teórico assumiria p=3.00)\n")
            f.write("N,tempo_medido_s_ou_projetado\n")
            for N, t in sorted(projecao["tempos"].items()):
                f.write(f"{N},{t:.4f}\n")
            f.write(f"{projecao['N_projetado']},{projecao['t_projetado']:.4f} (projetado)\n")
            f.write("\nN,custo_total_projetado_min\n")
            for N, m in sorted(projecao["projecao_min"].items()):
                f.write(f"{N},{m:.2f}\n")
    print(f"  CSV de robustez exportado: {caminho}")


# ══════════════════════════════════════════════════════════════════════════════
# § N — UTILITÁRIO DE ESTILO
# ══════════════════════════════════════════════════════════════════════════════

def _ax_dark(ax):
    """Aplica tema cosmológico escuro a um eixo matplotlib."""
    ax.set_facecolor(BG1)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BDR)
    ax.grid(True, color=BDR, lw=0.5, alpha=0.7)
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# § O — FIGURA 1: DIAGNÓSTICO COMPLETO (6 painéis)  [nota 16/631]
# ══════════════════════════════════════════════════════════════════════════════

def figura_diagnostico(res_mapa, validacao, mc, saida=None):
    """
    Figura 1 — Diagnóstico Gold Standard (3×2 painéis).

    Painéis:
    [0,0] Perfis Φ(r) analíticos — NFW c∈{5,10,15,20,30}
    [0,1] Viés ε(r) antes correção — banda ±1σ de 100 translações
    [1,0] Validação cruzada — 4 curvas (analítico / N_ref / N_cal / PCHIP)
    [1,1] Erro % antes vs depois — banda verde ±1% (meta de publicação)
    [2,0] Scatter Φ_ana vs Φ_num/Φ_corr — 100 halos por concentração
    [2,1] Tabela de confusão — Nadler 2025
    """
    if saida is None:
        saida = os.path.join(os.getcwd(), "fig1_diagnostico_gold.png")

    fig = plt.figure(figsize=(16, 18), facecolor=BG0)
    gs  = gridspec.GridSpec(3, 2, fig, hspace=0.44, wspace=0.36,
                             left=0.08, right=0.96, top=0.93, bottom=0.06)

    # ── [0,0] Perfis analíticos ───────────────────────────────────────────────
    ax = _ax_dark(fig.add_subplot(gs[0,0]))
    r  = np.logspace(-2.5, 0.05, 400)
    for c in CONCS_MAP:
        nfw = NFWAnalitico(c)
        ax.plot(r, nfw.potencial(r), color=CORES_C[c], lw=1.8, label=f"c={c}")
    ax.axvline(R_EVAL, color="white", lw=0.9, ls=":", alpha=0.6,
               label=f"R_eval={R_EVAL}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r/r_\mathrm{vir}$")
    ax.set_ylabel(r"$\Phi(r)\ [V_\mathrm{vir}^2]$")
    ax.set_title("Potencial NFW — Łokas & Mamon (2001)", fontsize=10)
    ax.legend(fontsize=8, facecolor=BG1, labelcolor=TXT, edgecolor=BDR)
    ax.annotate(r"Taylor $r\!\to\!0$:" "\n" r"$\Phi(0)=-c/f(c)$",
                xy=(0.013,-3.5), xytext=(0.055,-2.1), fontsize=7,
                color="#8b949e",
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=0.8))

    # ── [0,1] Viés ε(r) ───────────────────────────────────────────────────────
    ax = _ax_dark(fig.add_subplot(gs[0,1]))
    for d in res_mapa:
        em = d["erro_medio"]*100; es = d["erro_std"]*100; rc = d["r_cent"]
        ax.plot(rc, em, color=CORES_C[d["c"]], lw=1.8, label=f"c={d['c']}")
        ax.fill_between(rc, em-es, em+es, color=CORES_C[d["c"]], alpha=0.15)
    ax.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax.axhspan(-20,-13, color=RED, alpha=0.08, label="Viés típico 13–20%")
    ax.axvline(R_EVAL, color="white", lw=0.9, ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r/r_\mathrm{vir}$")
    ax.set_ylabel(r"$\varepsilon\ [\%]$")
    ax.set_title(f"Viés CIC-FFT antes da Correção ({res_mapa[0]['N']} transl./c)",
                 fontsize=10)
    ax.legend(fontsize=8, facecolor=BG1, labelcolor=TXT, edgecolor=BDR)

    # ── [1,0] Validação cruzada ───────────────────────────────────────────────
    ax = _ax_dark(fig.add_subplot(gs[1,0]))
    v  = validacao
    ax.plot(v["r_ana"],  v["phi_ana"],    "w-",  lw=2.5, label="Analítico")
    ax.plot(v["r_cent"], v["perfil_ref"], "--",  color=ORG, lw=1.6,
            label=f"N={v['N_ref']} (ref.)")
    ax.plot(v["r_cent"], v["perfil_cal"], ":",   color=RED, lw=1.6,
            label=f"N={v['N_cal']} s/ corr.")
    ax.plot(v["r_cent"], v["perfil_corr"],"-",   color=GRN, lw=2.2,
            label=f"N={v['N_cal']} + PCHIP Gold")
    ax.axvline(R_EVAL, color="white", lw=0.9, ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r/r_\mathrm{vir}$")
    ax.set_ylabel(r"$\Phi(r)$")
    ax.set_title(f"Validação Cruzada NFW — c={v['c']:.0f}", fontsize=10)
    ax.legend(fontsize=8, facecolor=BG1, labelcolor=TXT, edgecolor=BDR)

    # ── [1,1] Erro antes vs depois ────────────────────────────────────────────
    ax = _ax_dark(fig.add_subplot(gs[1,1]))
    v  = validacao; rc = v["r_cent"]; pa = v["perfil_ana_b"]
    with np.errstate(invalid="ignore", divide="ignore"):
        msk = np.abs(pa) > 1e-10
        ea  = np.where(msk,(v["perfil_cal"] -pa)/np.abs(pa)*100, np.nan)
        ed  = np.where(msk,(v["perfil_corr"]-pa)/np.abs(pa)*100, np.nan)
    ax.plot(rc, ea, color=RED, lw=1.8, label="Antes da correção")
    ax.plot(rc, ed, color=GRN, lw=1.8, label="Após PCHIP Gold")
    ax.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax.axhspan(-1, 1, color=GRN, alpha=0.12, label="±1% meta publicação")
    ax.axvline(R_EVAL, color="white", lw=0.9, ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r/r_\mathrm{vir}$")
    ax.set_ylabel(r"$\varepsilon_\Phi\ [\%]$")
    ax.set_title("Eficácia da Correção PCHIP Gold", fontsize=10)
    e_an_m = np.nanmean(np.abs(ea)); e_dp_m = np.nanmean(np.abs(ed))
    red = (1-e_dp_m/max(e_an_m,1e-10))*100
    ax.text(0.97,0.97, f"Redução: {red:.0f}%\nAntes: {e_an_m:.1f}%\n"
            f"Depois: {e_dp_m:.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=TXT, bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor=BG1, edgecolor=GRN, alpha=0.9))
    ylim_lo = min(-30, np.nanmin(ea)-3) if np.any(np.isfinite(ea)) else -35
    ax.set_ylim(ylim_lo, 15)
    ax.legend(fontsize=8, facecolor=BG1, labelcolor=TXT, edgecolor=BDR)

    # ── [2,0] Scatter Φ_ana vs Φ_num/Φ_corr ──────────────────────────────────
    ax   = _ax_dark(fig.add_subplot(gs[2,0]))
    norm = Normalize(vmin=mc["c_halos"].min(), vmax=mc["c_halos"].max())
    sm   = ScalarMappable(cmap="plasma", norm=norm); sm.set_array([])
    all_p = np.concatenate([mc["phi_ana"],mc["phi_num"],mc["phi_corr"]])
    lo,hi = all_p.min()*1.10, all_p.max()*0.90
    ax.plot([lo,hi],[lo,hi],"w--",lw=1,alpha=0.5,label="Paridade")
    for i in range(mc["N_halos"]):
        col = sm.to_rgba(mc["c_halos"][i])
        ax.scatter(mc["phi_ana"][i],mc["phi_num"][i],
                   marker="^",color=col,s=30,alpha=0.7,edgecolors="none")
        ax.scatter(mc["phi_ana"][i],mc["phi_corr"][i],
                   marker="o",color=col,s=30,alpha=0.7,edgecolors="none")
        if mc["recuperados"][i]:
            ax.scatter(mc["phi_ana"][i],mc["phi_corr"][i],
                       marker="*",color="white",s=120,zorder=6)
    ax.scatter([],[],marker="^",color="#aaa",s=30,label="Num s/corr.")
    ax.scatter([],[],marker="o",color="#aaa",s=30,label="Num + PCHIP")
    ax.scatter([],[],marker="*",color="white",s=60,label="Recuperados ★")
    cb = fig.colorbar(sm,ax=ax,pad=0.02,fraction=0.04)
    cb.set_label("c",color=TXT,fontsize=8); cb.ax.tick_params(colors=TXT)
    ax.set_xlabel(r"$\Phi_\mathrm{analítico}$")
    ax.set_ylabel(r"$\Phi_\mathrm{numérico}$")
    ax.set_title(f"Scatter Φ — {mc['N_halos']} Halos MC", fontsize=10)
    ax.legend(fontsize=7,facecolor=BG1,labelcolor=TXT,edgecolor=BDR,ncol=2)

    # ── [2,1] Tabela de confusão ───────────────────────────────────────────────
    ax = _ax_dark(fig.add_subplot(gs[2,1])); ax.axis("off")
    Nh = mc["N_halos"]
    na = int(mc["class_a"].sum()); nn = int(mc["class_n"].sum())
    nc = int(mc["class_c"].sum()); fn = int(mc["falso_escuro"].sum())
    rc_ = int(mc["recuperados"].sum())
    pct = f"{100*rc_//max(fn,1)}%" if fn > 0 else "N/A"
    linhas=[
        ["Método",                        "Estelar",   "Escuro",     "Acc."],
        ["Analítico (padrão ouro)",        str(na),     str(Nh-na),   "100%"],
        [f"Numérico N={mc['N_grade']}",    str(nn),     str(Nh-nn),   f"{mc['acc_num']:.0f}%"],
        [f"N={mc['N_grade']} + PCHIP Gold",str(nc),    str(Nh-nc),   f"{mc['acc_corr']:.0f}%"],
        ["Falsos Escuros (viés FFT)",      str(fn),     "—",          ""],
        ["RECUPERADOS ★",                  str(rc_),    "—",          pct],
    ]
    cl = ["#30363d","#21262d","#1c2128","#1a3a1a","#3d1f1f","#1a3a1a"]
    ct = [TXT, TXT, RED, GRN, ORG, GRN]
    tab = ax.table(cellText=linhas,cellLoc="center",loc="center",
                   bbox=[0.01,0.08,0.98,0.87])
    tab.auto_set_font_size(False); tab.set_fontsize(9.5)
    for (row,col),cell in tab.get_celld().items():
        cell.set_facecolor(cl[min(row,len(cl)-1)])
        cell.set_edgecolor("#21262d")
        cell.get_text().set_color(ct[min(row,len(ct)-1)])
        if row==0: cell.get_text().set_fontweight("bold")
    ax.set_title("Tabela de Confusão — Critério de Nadler (2025)", fontsize=10)

    fig.suptitle(
        "NFW Mini-Halo Bias Correction — GOLD STANDARD  ·  PIBIC 2025/2026\n"
        r"PCHIP 2D$(r,c)$ | $\Phi_\mathrm{corr}=\Phi_\mathrm{num}/(1-\hat\varepsilon)$"
        f" | N={mc['N_grade']}, {mc['N_halos']} halos",
        color=TXT, fontsize=11, fontweight="bold")
    plt.savefig(saida, dpi=150, bbox_inches="tight", facecolor=BG0)
    plt.close(fig)
    print(f"  Figura 1 salva: {saida}")


# ══════════════════════════════════════════════════════════════════════════════
# § P — FIGURA 2: THE PUNCH APRIMORADO  [G5, notas 16/632, 16/657]
# ══════════════════════════════════════════════════════════════════════════════

def figura_punch(mc, saida=None):
    """
    Figura 2 — "The Punch" Gold Standard.

    Melhorias sobre v5:
    -------------------
    • Zona de "recuperação física" destacada com gradiente de cor
    • Setas de recuperação coloridas pelo delta de score
    • Anotação quantitativa: ΔΦ e Δscore de cada halo recuperado
    • Histogramas marginais de V_max e score (distribuição da amostra)
    • Banda de incerteza de grade na fronteira de Nadler

    Eixos principais:
        X: V_max físico [km/s]     — proxy de massa
        Y: Score = |Φ(R_EVAL)|/|Φ_ref|  — profundidade do poço
    """
    if saida is None:
        saida = os.path.join(os.getcwd(), "fig2_punch_gold.png")

    # Layout com histogramas marginais
    fig = plt.figure(figsize=(14, 10), facecolor=BG0)
    gs  = gridspec.GridSpec(2, 2, fig,
                             width_ratios=[4,1], height_ratios=[1,4],
                             hspace=0.04, wspace=0.04,
                             left=0.09, right=0.92, top=0.92, bottom=0.09)

    ax     = fig.add_subplot(gs[1,0])   # scatter principal
    ax_top = fig.add_subplot(gs[0,0], sharex=ax)   # hist V_max
    ax_rgt = fig.add_subplot(gs[1,1], sharey=ax)   # hist score

    for a in [ax, ax_top, ax_rgt]:
        a.set_facecolor(BG1)
        for sp in a.spines.values(): sp.set_edgecolor(BDR)
        a.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.grid(True, color=BDR, lw=0.5, alpha=0.6)

    Vm = mc["Vmax"]
    Vm_min, Vm_max = Vm.min()*0.85, Vm.max()*1.10
    sc_max = min(2.5, mc["sc_ana"].max()*1.22)

    # ── Zonas de classificação ────────────────────────────────────────────────
    ax.axhspan(0, SCORE_LIM, facecolor="#12122a", alpha=0.7, zorder=1)
    ax.axhspan(SCORE_LIM, sc_max, facecolor="#0a1f12", alpha=0.5, zorder=1)
    # Zona de transição (Vmax < limiar)
    ax.fill_betweenx([SCORE_LIM, sc_max], x1=Vm_min, x2=VMAX_LIM_KMS,
                     facecolor="#2d1b00", alpha=0.45, zorder=2)

    # ── Zona de recuperação física (destacada) ────────────────────────────────
    # Região onde o erro de grade "apagava" galáxias: score entre 0.65–0.85
    ax.axhspan(0.65, SCORE_LIM, facecolor="#1a0d00", alpha=0.0, zorder=1)
    score_rec_low = max(0.60, SCORE_LIM - 0.30)
    ax.fill_betweenx([score_rec_low, SCORE_LIM],
                     x1=VMAX_LIM_KMS*0.9, x2=Vm_max,
                     facecolor="#3a1500", alpha=0.55, zorder=2,
                     label="Zona de recuperação física")
    ax.text(VMAX_LIM_KMS*1.05, (score_rec_low+SCORE_LIM)/2,
            "ZONA DE\nRECUPERAÇÃO\nFÍSICA",
            color=ORG, fontsize=7.5, ha="left", va="center",
            style="italic", fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG0, alpha=0.6,
                      edgecolor=ORG))

    # Rótulos das zonas principais
    ax.text(VMAX_LIM_KMS*1.05, SCORE_LIM*0.45,
            "HALOS ESCUROS\n(sem formação estelar)",
            color="#5a5a8a", fontsize=9, ha="left", va="center",
            style="italic", zorder=3)
    ax.text(VMAX_LIM_KMS*1.05, SCORE_LIM*1.20,
            "HALOS ESTELARES\n(formam estrelas)",
            color="#2a7a3a", fontsize=9, ha="left", va="center",
            style="italic", zorder=3)

    # ── Fronteiras de Nadler com banda de incerteza de grade ──────────────────
    # Incerteza ≈ ε_típico × SCORE_LIM ≈ 0.20 × 0.85
    sigma_score = 0.20 * SCORE_LIM
    r_score_arr = np.linspace(Vm_min, Vm_max, 200)
    ax.fill_between(r_score_arr,
                    SCORE_LIM - sigma_score, SCORE_LIM + sigma_score,
                    color=ACC, alpha=0.10, zorder=3,
                    label=r"Incerteza de grade $(\pm\sigma_\varepsilon)$")
    ax.axhline(SCORE_LIM, color=ACC, lw=1.5, ls="--", alpha=0.9, zorder=4,
               label=f"Score = {SCORE_LIM} (Nadler 2025)")
    ax.axvline(VMAX_LIM_KMS, color=ORG, lw=1.5, ls="--", alpha=0.9, zorder=4,
               label=fr"$V_{{\rm max}}$ = {VMAX_LIM_KMS} km/s")

    # ── Pontos de fundo: halos estelares normais ──────────────────────────────
    norm = LogNorm(vmin=mc["M_halos"].min(), vmax=mc["M_halos"].max())
    sm   = ScalarMappable(cmap="YlOrRd", norm=norm); sm.set_array([])

    mask_ok  = mc["class_a"] & mc["class_c"] & ~mc["falso_escuro"]
    mask_esc = ~mc["class_a"]
    ax.scatter(Vm[mask_ok], mc["sc_ana"][mask_ok],
               marker="o", s=45, c=mc["M_halos"][mask_ok],
               norm=norm, cmap="YlOrRd", alpha=0.45, zorder=5,
               edgecolors=BDR, lw=0.4)
    ax.scatter(Vm[mask_esc], mc["sc_ana"][mask_esc],
               marker="s", s=35, color="#3a3a5a", alpha=0.55, zorder=5,
               edgecolors=BDR, lw=0.4)

    # ── Falsos Escuros: setas de recuperação coloridas por ΔScore ─────────────
    delta_sc = mc["sc_corr"] - mc["sc_num"]   # ΔScore por halo
    ds_max   = delta_sc[mc["falso_escuro"]].max() if mc["falso_escuro"].any() else 1.0
    cmap_arr = plt.cm.hot

    for i in range(mc["N_halos"]):
        if not mc["falso_escuro"][i]:
            continue
        vm  = Vm[i]
        s_n = mc["sc_num"][i]
        s_c = mc["sc_corr"][i]
        ds  = delta_sc[i]
        cor_seta = cmap_arr(0.2 + 0.7 * ds / max(ds_max, 1e-10))

        ax.scatter(vm, s_n, marker="v", s=100,
                   color=RED, zorder=8, edgecolors="white", lw=0.7)
        ax.scatter(vm, s_c, marker="o", s=100,
                   color=GRN, zorder=9, edgecolors="white", lw=0.7)
        ax.annotate("",
                    xy=(vm, s_c), xytext=(vm, s_n),
                    arrowprops=dict(arrowstyle="-|>", color=cor_seta,
                                   lw=2.0, mutation_scale=16),
                    zorder=10)
        if mc["recuperados"][i]:
            ax.scatter(vm, s_c, marker="*", s=260,
                       color="#ffd700", zorder=11,
                       edgecolors="black", lw=0.5)
            # Anotação quantitativa: ΔScore
            ax.annotate(f"+{ds:.2f}",
                        xy=(vm, s_c), xytext=(vm+0.25, s_c+0.04),
                        fontsize=6, color="#ffd700", zorder=12,
                        arrowprops=dict(arrowstyle="-", color="#ffd700",
                                        lw=0.5, alpha=0.7))

    # ── Histogramas marginais ─────────────────────────────────────────────────
    ax_top.hist(Vm[mc["class_a"]],  bins=14, color=GRN, alpha=0.6,
                edgecolor=BDR, label="Estelares (ana)")
    ax_top.hist(Vm[~mc["class_a"]], bins=14, color=RED, alpha=0.4,
                edgecolor=BDR, label="Escuros")
    ax_top.set_facecolor(BG1)
    ax_top.tick_params(colors=TXT, labelsize=7, labelbottom=False)
    ax_top.set_ylabel("N halos", color=TXT, fontsize=8)
    ax_top.legend(fontsize=6, facecolor=BG0, labelcolor=TXT, edgecolor=BDR)

    ax_rgt.hist(mc["sc_ana"], bins=14, color=ACC, alpha=0.5,
                orientation="horizontal", edgecolor=BDR)
    ax_rgt.hist(mc["sc_num"][mc["falso_escuro"]], bins=6,
                color=RED, alpha=0.6, orientation="horizontal",
                edgecolor=BDR, label="F. Escuros (num)")
    ax_rgt.set_facecolor(BG1)
    ax_rgt.tick_params(colors=TXT, labelsize=7, labelleft=False)
    ax_rgt.set_xlabel("N", color=TXT, fontsize=8)

    # ── Legenda principal ─────────────────────────────────────────────────────
    handles = [
        plt.scatter([],[],marker="o",s=45,color="#aaaaaa",alpha=0.5,
                    label="Estelar (analítico ≡ corrigido)"),
        plt.scatter([],[],marker="s",s=35,color="#3a3a5a",alpha=0.7,
                    label="Sempre escuro"),
        plt.scatter([],[],marker="v",s=100,color=RED,
                    label=r"Falso Escuro ($\Phi_\mathrm{num}$)"),
        plt.scatter([],[],marker="o",s=100,color=GRN,
                    label=r"Corrigido ($\Phi_\mathrm{corr}$, PCHIP)"),
        plt.scatter([],[],marker="*",s=200,color="#ffd700",
                    label="★ RECUPERADO (Nadler 2025)"),
    ]
    leg = ax.legend(handles=handles, fontsize=8.5, facecolor=BG1,
                    labelcolor=TXT, edgecolor=BDR, loc="lower right",
                    framealpha=0.9)

    # Colorbar massa
    cb = fig.colorbar(sm, ax=[ax, ax_rgt], pad=0.01, fraction=0.025,
                      location="right")
    cb.set_label(r"$M_\mathrm{vir}\ [M_\odot]$", color=TXT, fontsize=10)
    cb.ax.tick_params(colors=TXT, labelsize=8)
    cb.set_ticks([1e7,3e7,1e8,3e8])
    cb.set_ticklabels([r"$10^7$",r"$3\!\times\!10^7$",
                       r"$10^8$",r"$3\!\times\!10^8$"])

    # Caixa de estatísticas
    fn  = int(mc["falso_escuro"].sum())
    rec = int(mc["recuperados"].sum())
    pct = 100*rec//max(fn,1)
    stats = (f"N = {mc['N_halos']} halos  |  N_grade = {mc['N_grade']}\n"
             f"Erro sem correção:  {mc['eps_central']:.1f}%\n"
             f"Erro PCHIP Gold:    {mc['eps_corrigido']:.2f}%\n"
             f"Falsos Escuros:     {fn}\n"
             f"RECUPERADOS:        {rec} ({pct}%)")
    ax.text(0.02,0.97, stats, transform=ax.transAxes,
            fontsize=8.5, color=TXT, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BG0,
                      edgecolor=ACC, alpha=0.92), zorder=12)

    ax.set_xlabel(r"$V_\mathrm{max}\ [\mathrm{km\,s^{-1}}]$", fontsize=13)
    ax.set_ylabel(r"Score $|\Phi(r_\mathrm{eval})|/|\Phi_\mathrm{ref}|$",
                  fontsize=12)
    ax.set_xlim(Vm_min, Vm_max)
    ax.set_ylim(0, sc_max)
    ax.title.set_color(TXT)
    ax.set_title(
        r"Fronteira de Nadler (2025) — $V_\mathrm{max}$–Score" "\n"
        "Zona de Recuperação Física (PCHIP Gold Standard)",
        fontsize=11, fontweight="bold", color=TXT)

    plt.savefig(saida, dpi=180, bbox_inches="tight", facecolor=BG0)
    plt.close(fig)
    print(f"  Figura 2 (The Punch Gold) salva: {saida}")


# ══════════════════════════════════════════════════════════════════════════════
# § Q — FIGURA 3: FRAÇÃO DE OCUPAÇÃO DE GALÁXIAS  [G4]
# ══════════════════════════════════════════════════════════════════════════════

def figura_focc(mc, saida=None):
    """
    Figura 3 — f_occ(V_max): Fração de Ocupação de Galáxias.

    Demonstração quantitativa do impacto do viés:
        • Curva analítica: sigmóide centrada em V_max ≈ 12 km/s (correto)
        • Curva numérica: deslocada para a direita (halos sub-avaliados
          precisam de maior V_max para superar o threshold de score)
        • Curva corrigida: restaurada à posição correta

    A distância entre a curva numérica e a analítica é a "assinatura" do viés.
    A sobreposição com a curva corrigida é a prova da eficácia do corretor.

    Referência [7]: Vale & Ostriker (2004) — função de ocupação de halos.
    """
    if saida is None:
        saida = os.path.join(os.getcwd(), "fig3_focc_gold.png")

    focc = calcular_focc(mc)
    bc   = focc["bc"]
    Vm_fine = np.linspace(bc.min()*0.9, bc.max()*1.05, 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG0)

    for ax in [ax1, ax2]:
        ax.set_facecolor(BG1)
        ax.tick_params(colors=TXT, labelsize=10)
        ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
        ax.title.set_color(TXT)
        for sp in ax.spines.values(): sp.set_edgecolor(BDR)
        ax.grid(True, color=BDR, lw=0.5, alpha=0.7)

    # ── Painel esquerdo: f_occ pontos + ajuste logístico ─────────────────────
    ax1.scatter(bc, focc["fo_a"], color=GRN, s=80, zorder=6,
                label="Analítico (dados)", edgecolors="white", lw=0.5)
    ax1.scatter(bc, focc["fo_n"], color=RED, s=80, marker="^", zorder=6,
                label="Numérico s/corr. (dados)", edgecolors="white", lw=0.5)
    ax1.scatter(bc, focc["fo_c"], color=ACC, s=80, marker="D", zorder=6,
                label="PCHIP Gold (dados)", edgecolors="white", lw=0.5)

    if focc["fa_fit"] is not None:
        ax1.plot(Vm_fine, _logistica(Vm_fine, *focc["pa"]),
                 color=GRN, lw=2.5, ls="-",
                 label=f"Analítico — V50={focc['pa'][0]:.1f} km/s")
    if focc["fn_fit"] is not None:
        ax1.plot(Vm_fine, _logistica(Vm_fine, *focc["pn"]),
                 color=RED, lw=2.5, ls="--",
                 label=f"Numérico — V50={focc['pn'][0]:.1f} km/s")
    if focc["fc_fit"] is not None:
        ax1.plot(Vm_fine, _logistica(Vm_fine, *focc["pc"]),
                 color=ACC, lw=2.5, ls="-.",
                 label=f"PCHIP Gold — V50={focc['pc'][0]:.1f} km/s")

    ax1.axvline(VMAX_LIM_KMS, color=ORG, lw=1.5, ls=":", alpha=0.8,
                label=f"Limiar reionização {VMAX_LIM_KMS} km/s")
    ax1.axhline(0.5, color="white", lw=0.7, ls="--", alpha=0.4)

    # Seta de deslocamento do viés
    if focc["pa"] is not None and focc["pn"] is not None:
        v50a = focc["pa"][0]; v50n = focc["pn"][0]
        if abs(v50n - v50a) > 0.3:
            ax1.annotate("",
                xy=(v50a, 0.5), xytext=(v50n, 0.5),
                arrowprops=dict(arrowstyle="<->", color=ORG,
                               lw=2.0, mutation_scale=15))
            ax1.text((v50a+v50n)/2, 0.54,
                     f"Δ={v50n-v50a:+.1f} km/s\n(viés CIC-FFT)",
                     ha="center", va="bottom", fontsize=8, color=ORG)

    ax1.set_xlabel(r"$V_\mathrm{max}\ [\mathrm{km\,s^{-1}}]$", fontsize=12)
    ax1.set_ylabel(r"$f_\mathrm{occ}(V_\mathrm{max})$", fontsize=12)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_title("Fração de Ocupação de Galáxias\nAjuste Logístico por Método",
                  fontsize=10)
    ax1.legend(fontsize=8, facecolor=BG1, labelcolor=TXT, edgecolor=BDR,
               loc="upper left")

    # ── Painel direito: Δf_occ (desvio do viés) ──────────────────────────────
    # Usa bc (N_bins pontos) para garantir arrays de mesmo tamanho
    fo_a_ref = focc["fo_a"]
    fo_n_ref = focc["fo_n"]
    fo_c_ref = focc["fo_c"]
    ok = np.isfinite(fo_a_ref) & np.isfinite(fo_n_ref)

    if ok.sum() >= 2:
        x_arr   = bc[ok]
        delta_n = fo_n_ref[ok] - fo_a_ref[ok]

        ax2.fill_between(x_arr, 0, delta_n, where=(delta_n < 0),
                         color=RED, alpha=0.4, label="Déficit (viés FFT)")
        ax2.fill_between(x_arr, 0, delta_n, where=(delta_n >= 0),
                         color=ORG, alpha=0.25)
        ax2.plot(x_arr, delta_n, color=RED, lw=2.0, marker="^", ms=5,
                 label="Num − Analítico")

        ok_c = ok & np.isfinite(fo_c_ref)
        if ok_c.sum() >= 2:
            x_c     = bc[ok_c]
            delta_c = fo_c_ref[ok_c] - fo_a_ref[ok_c]
            ax2.plot(x_c, delta_c, color=GRN, lw=2.0, ls="--",
                     marker="D", ms=5, label="PCHIP − Analítico")
            ax2.fill_between(x_c, 0, delta_c, color=GRN, alpha=0.20)

    ax2.axhline(0, color="white", lw=1.0, ls="--", alpha=0.6)
    ax2.axvline(VMAX_LIM_KMS, color=ORG, lw=1.5, ls=":", alpha=0.8)
    ax2.set_xlabel(r"$V_\mathrm{max}\ [\mathrm{km\,s^{-1}}]$", fontsize=12)
    ax2.set_ylabel(r"$\Delta f_\mathrm{occ}\ (=f_\mathrm{método}-f_\mathrm{ana})$",
                   fontsize=11)
    ax2.set_title("Desvio da f_occ Correta\n(Assinatura do Viés Gravitacional)",
                  fontsize=10)
    ax2.legend(fontsize=9, facecolor=BG1, labelcolor=TXT, edgecolor=BDR)

    fig.suptitle(
        r"Fração de Ocupação de Galáxias $f_\mathrm{occ}(V_\mathrm{max})$  "
        "— PCHIP Gold Standard",
        color=TXT, fontsize=12, fontweight="bold")
    plt.savefig(saida, dpi=150, bbox_inches="tight", facecolor=BG0)
    plt.close(fig)
    print(f"  Figura 3 (f_occ) salva: {saida}")


# ══════════════════════════════════════════════════════════════════════════════
# § R — FIGURA 4: TESTE DE SENSIBILIDADE HERNQUIST  [G2]
# ══════════════════════════════════════════════════════════════════════════════

def figura_hernquist(res_hq, saida=None):
    """
    Figura 4 — Sensibilidade ao Perfil: NFW vs. Hernquist.

    Demonstra que o corretor PCHIP (calibrado somente em NFW) reduz
    eficientemente o viés no perfil de Hernquist — evidência de que
    a correção opera sobre a singularidade da cúspide (r⁻¹), não na
    forma global do perfil.

    Layout 1×2:
    [0] Perfis Φ(r): analítico / numérico / corrigido para Hernquist
    [1] Erro % antes vs. depois — comparação com NFW (do validacao)
    """
    if saida is None:
        saida = os.path.join(os.getcwd(), "fig4_hernquist_gold.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG0)

    for ax in [ax1, ax2]:
        ax.set_facecolor(BG1)
        ax.tick_params(colors=TXT, labelsize=10)
        ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
        ax.title.set_color(TXT)
        for sp in ax.spines.values(): sp.set_edgecolor(BDR)
        ax.grid(True, color=BDR, lw=0.5, alpha=0.7)

    # Perfil NFW de referência para sobreposição
    nfw_ref = NFWAnalitico(res_hq["c_equiv"])
    r_all   = np.logspace(-2.5, 0.0, 400)

    # ── Painel esquerdo: perfis Φ(r) ─────────────────────────────────────────
    ax1.plot(r_all, nfw_ref.potencial(r_all), color=CORES_C.get(10,"#f28e2b"),
             lw=1.5, ls=":", alpha=0.7,
             label=f"NFW analítico (c={res_hq['c_equiv']:.0f})")
    ax1.plot(res_hq["r_ana"], res_hq["phi_ha"], color=PRP, lw=2.5,
             label=f"Hernquist analítico (c_eq={res_hq['c_equiv']:.0f})")
    ax1.plot(res_hq["r_cent"], res_hq["pn_b"],  color=RED, lw=1.8, ls="--",
             label="Hernquist numérico s/corr.")
    ax1.plot(res_hq["r_cent"], res_hq["pc_b"],  color=GRN, lw=2.0,
             label="Hernquist + PCHIP (NFW-calibrado)")
    ax1.axvline(R_EVAL, color="white", lw=0.9, ls=":", alpha=0.6,
                label=f"R_eval={R_EVAL}")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$r/r_\mathrm{vir}$", fontsize=12)
    ax1.set_ylabel(r"$\Phi(r)$", fontsize=12)
    ax1.set_title(f"Perfil de Hernquist — Transferabilidade do PCHIP\n"
                  f"N={res_hq['N']}, c_equiv={res_hq['c_equiv']:.0f}", fontsize=10)
    ax1.legend(fontsize=8, facecolor=BG1, labelcolor=TXT, edgecolor=BDR)

    # ── Painel direito: erro % ────────────────────────────────────────────────
    ax2.plot(res_hq["r_cent"], res_hq["ea"], color=RED, lw=2.0,
             label=f"Hernquist antes ({res_hq['e_ant']:.1f}%)")
    ax2.plot(res_hq["r_cent"], res_hq["ed"], color=GRN, lw=2.0,
             label=f"Hernquist depois ({res_hq['e_dep']:.2f}%)")
    ax2.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax2.axhspan(-1, 1, color=GRN, alpha=0.12, label="±1% meta publicação")
    ax2.axvline(R_EVAL, color="white", lw=0.9, ls=":", alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$r/r_\mathrm{vir}$", fontsize=12)
    ax2.set_ylabel(r"$\varepsilon_\Phi\ [\%]$", fontsize=12)
    ax2.set_title(f"Eficiência de Transferência: {res_hq['efic']:.0f}%\n"
                  "(Corretor calibrado em NFW → aplicado ao Hernquist)",
                  fontsize=10)

    # Caixa de métricas
    txt = (f"Perfil Hernquist (c_eq={res_hq['c_equiv']:.0f})\n"
           f"Cúspide: ρ ∝ r⁻¹  (idem NFW)\n"
           f"ε antes:  {res_hq['e_ant']:.1f}%\n"
           f"ε depois: {res_hq['e_dep']:.2f}%\n"
           f"Eficiência: {res_hq['efic']:.0f}%")
    ax2.text(0.97,0.97, txt, transform=ax2.transAxes,
             ha="right", va="top", fontsize=8.5, color=TXT,
             bbox=dict(boxstyle="round,pad=0.5",facecolor=BG0,
                       edgecolor=PRP, alpha=0.9))
    ax2.legend(fontsize=8, facecolor=BG1, labelcolor=TXT, edgecolor=BDR)

    fig.suptitle(
        "Teste de Sensibilidade: Hernquist  [Ref. 6]\n"
        r"Corretor PCHIP $\varepsilon(r,c)$ calibrado em NFW — "
        "Transferabilidade para cúspide $r^{-1}$ genérica",
        color=TXT, fontsize=11, fontweight="bold")
    plt.savefig(saida, dpi=150, bbox_inches="tight", facecolor=BG0)
    plt.close(fig)
    print(f"  Figura 4 (Hernquist) salva: {saida}")


# ══════════════════════════════════════════════════════════════════════════════
# § S — PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Pipeline Gold Standard — 10 etapas.

    ┌────────────────────────────────────────────────────────────┐
    │  PARÂMETROS DE EXECUÇÃO — ajuste aqui antes de rodar      │
    │                                                            │
    │  Dev rápido  (~10 min):                                    │
    │    N_MAP=64,  N_TRANS=10,  M_SUB=4,  N_HALOS=30           │
    │                                                            │
    │  Publicação  (~3–6 h em CPU):                             │
    │    N_MAP=128, N_TRANS=100, M_SUB=8,  N_HALOS=100          │
    └────────────────────────────────────────────────────────────┘
    """
    t0 = time.perf_counter()

    print("=" * 72)
    print("  NFW MINI-HALO BIAS CORRECTION — GOLD STANDARD (v6)")
    print("  PIBIC 2025/2026  |  Critério de Nadler (2025)")
    print("=" * 72)

    # ── Fingerprint do ambiente de execução ────────────────────────────────
    # Registrado explicitamente para que variações no tempo total de CPU
    # entre execuções (ex.: 196,1 min vs. 221,0 min) possam ser atribuídas
    # a diferenças de hardware/ambiente em vez de ficarem sem explicação —
    # ponto levantado na avaliação do relatório.
    import platform as _platform
    ambiente = dict(
        data_execucao=datetime.now().isoformat(timespec="seconds"),
        sistema=_platform.system(),
        release=_platform.release(),
        maquina=_platform.machine(),
        processador=_platform.processor() or "não identificado",
        python=_platform.python_version(),
        numpy=np.__version__,
        n_cpus_logicos=os.cpu_count(),
    )
    print("\n  [ambiente de execução]")
    for k, v in ambiente.items():
        print(f"    {k}: {v}")
    print("  NOTA: o tempo total de CPU reportado ao final desta execução é "
          "específico deste ambiente; variações de minutos entre execuções "
          "em máquinas/cargas diferentes são esperadas e não indicam "
          "mudança nos resultados numéricos (que dependem apenas das "
          "sementes fixas, não do hardware).")

    # ─────────────────── AJUSTE AQUI ──────────────────────────────────────────
    N_MAP   = 128    # → publicação: 128
    N_TRANS = 100    # → publicação: 100
    M_SUB   = 8     # → publicação: 8
    N_HALOS = 100    # → publicação: 100
    # ──────────────────────────────────────────────────────────────────────────

    dx = CAIXA_L / N_MAP
    print(f"\n  N_MAP={N_MAP} | N_TRANS={N_TRANS} | M_SUB={M_SUB} | "
          f"N_HALOS={N_HALOS}")
    print(f"  dx={dx:.5f} r_vir | R_EVAL={R_EVAL} = {R_EVAL/dx:.1f} células")
    print(f"  Memória/halo: ~{2*N_MAP**3*8/1e6:.0f} MB (pico)")

    # ── ETAPA 1: Validação analítica NFW + Hernquist ──────────────────────────
    print("\n─── ETAPA 1: Validação Analítica ─────────────────────────────────")
    print(f"  {'c':>4}  {'rs_NFW':>8}  {'Phi0_NFW':>10}  "
          f"{'Vm_NFW':>8}  {'a_HQ':>8}  {'Phi0_HQ':>10}  {'Vm_HQ':>8}")
    for c in CONCS_MAP:
        nfw = NFWAnalitico(c); hq = HernquistAnalitico(c)
        phi0n = float(nfw.potencial(np.array([1e-10]))[0])
        phi0h = float(hq.potencial(np.array([1e-10]))[0])
        print(f"  {c:4d}  {nfw.rs:8.4f}  {phi0n:10.4f}  {nfw.vmax_adim():8.4f}"
              f"  {hq.a:8.4f}  {phi0h:10.4f}  {hq.vmax_adim():8.4f}")
    print(f"\n  PHI_REF = {PHI_REF:.4f}  |  SCORE_LIM={SCORE_LIM}  "
          f"|  VMAX_LIM={VMAX_LIM_KMS} km/s")

    # ── ETAPA 2: Smoke test do solver ─────────────────────────────────────────
    print("\n─── ETAPA 2: Smoke Test Solver (N=32, c=10) ──────────────────────")
    _nfw_t = NFWAnalitico(10)
    _Pt, _rt = phi_num(_nfw_t, N=32, deslocamento=np.zeros(3), M_sub=4)
    _mid = 16
    _pv = float(_Pt[_mid,_mid,_mid])
    _pa = float(_nfw_t.potencial(np.array([_rt[_mid,_mid,_mid]]))[0])
    print(f"  Phi_num={_pv:.4f} | Phi_ana={_pa:.4f} | "
          f"Erro={(_pv-_pa)/abs(_pa)*100:.1f}% (esperado ~30% para N=32)")
    del _Pt, _rt; gc.collect()

    # ── ETAPAS 3+4: Mapeamento + PCHIP ────────────────────────────────────────
    print(f"\n─── ETAPAS 3+4: Mapeamento ε(r,c) + CorretorPCHIP ───────────────")
    res_mapa = []
    for c in CONCS_MAP:
        d = mapear_erro(c, N=N_MAP, Ntr=N_TRANS, M_sub=M_SUB, seed=RNG_SEED)
        res_mapa.append(d)
        i_ev = np.argmin(np.abs(d["r_cent"] - R_EVAL))
        print(f"    c={c:2d} | ε(r→0)={d['erro_medio'][0]*100:.1f}±"
              f"{d['erro_std'][0]*100:.1f}%  "
              f"ε(r_eval)={d['erro_medio'][i_ev]*100:.1f}%  "
              f"máx={d['erro_max'][i_ev]*100:.1f}%  "
              f"P95={d['erro_p95'][i_ev]*100:.1f}%  "
              f"IC95=[{d['erro_ic95_lo'][i_ev]*100:.1f},"
              f"{d['erro_ic95_hi'][i_ev]*100:.1f}]%")

    print("\n  Construindo CorretorPCHIP 2D(r, c) ...")
    corretor = CorretorPCHIP(res_mapa)

    # ── ETAPA 5: Validação cruzada NFW ────────────────────────────────────────
    print("\n─── ETAPA 5: Validação Cruzada NFW ───────────────────────────────")
    validacao = validacao_cruzada(corretor, c_teste=10.0,
                                   N_ref=N_MAP, N_cal=max(N_MAP//2,32),
                                   M_sub=M_SUB, seed=123)
    pa_b = validacao["perfil_ana_b"]
    msk  = np.abs(pa_b) > 1e-10
    with np.errstate(invalid="ignore", divide="ignore"):
        e_ant = np.nanmean(np.abs(
            (validacao["perfil_cal"][msk]-pa_b[msk])/pa_b[msk]))*100
        e_dep = np.nanmean(np.abs(
            (validacao["perfil_corr"][msk]-pa_b[msk])/pa_b[msk]))*100
    print(f"  Erro ANTES:  {e_ant:.2f}%")
    print(f"  Erro DEPOIS: {e_dep:.2f}%")
    if e_ant > 0:
        print(f"  Redução:     {(1-e_dep/e_ant)*100:.1f}%")
    print(f"  Erro máximo (todas as cascas): antes={validacao['erro_max_ant']:.2f}% "
          f"depois={validacao['erro_max_dep']:.2f}%")
    print(f"  Erro P95    (todas as cascas): antes={validacao['erro_p95_ant']:.2f}% "
          f"depois={validacao['erro_p95_dep']:.2f}%")
    print(f"  Erro especificamente em r_eval={R_EVAL}: "
          f"antes={validacao['erro_reval_ant']:.2f}% "
          f"depois={validacao['erro_reval_dep']:.2f}%  "
          f"[métrica mais relevante para o Critério de Nadler — Seção 4.1]")

    # ── ETAPA 6: Teste de Sensibilidade Hernquist ─────────────────────────────
    print("\n─── ETAPA 6: Teste de Sensibilidade — Hernquist ──────────────────")
    res_hq = teste_sensibilidade_hernquist(corretor, c_equiv=10.0,
                                            N=max(N_MAP//2, 32),
                                            M_sub=M_SUB, seed=77)

    # ── ETAPA 6.5: Estudos de robustez adicionais ─────────────────────────────
    # Executados em escala reduzida (N pequeno) por serem estudos de
    # SENSIBILIDADE/CONVERGÊNCIA, não de calibração — não precisam da
    # resolução de publicação para revelar a tendência qualitativa e a
    # ordem de grandeza do efeito. Isto atende diretamente às pendências
    # "estudo de convergência M_sub", "sensibilidade a L_caixa",
    # "justificativa de Φ_ref" e "teste de escalabilidade N=256"
    # apontadas nas avaliações do relatório.
    print("\n─── ETAPA 6.5: Estudos de Robustez Adicionais ────────────────────")
    conv_msub = estudo_convergencia_Msub(c=10.0, N=32, M_subs=(2, 4, 8, 16),
                                          seed=999)
    sens_lbox = estudo_sensibilidade_Lbox(c=10.0, N=32, M_sub=4,
                                           Ls=(2.0, 3.0, 4.0), seed=999)
    sens_phiref = sensibilidade_phi_ref(c_refs=(5.0, 10.0, 15.0, 20.0, 30.0),
                                         c_halo=8.0)
    print("\n  [sensibilidade Φ_ref] halo de teste c=8.0 (fixo); "
          "variando apenas a concentração de referência c_ref:")
    for linha in sens_phiref["linhas"]:
        marca = "ESTELAR" if linha["estelar"] else "escuro "
        print(f"    c_ref={linha['c_ref']:5.1f} | Φ_ref={linha['phi_ref']:8.4f} | "
              f"Score={linha['score']:.3f} | classificação={marca}")
    print("    → O MESMO halo muda de classificação (estelar↔escuro) "
          "dependendo unicamente da escolha arbitrária de c_ref, "
          "confirmando a fragilidade apontada na avaliação (Seção 2.5).")
    projecao = projecao_escalabilidade(N_alvo=(32, 64, N_MAP), M_sub=M_SUB,
                                        N_trans_referencia=N_TRANS,
                                        N_halos_referencia=N_HALOS,
                                        c_amostra=10.0, seed=999,
                                        N_projetado=256)

    # ── ETAPA 7: Monte Carlo 100 halos ────────────────────────────────────────
    print(f"\n─── ETAPA 7: Monte Carlo ({N_HALOS} Halos) ──────────────────────")
    mc_result = monte_carlo(corretor, N_halos=N_HALOS,
                             N_grade=N_MAP, M_sub=M_SUB, seed=RNG_SEED)

    # ── ETAPA 8: f_occ ────────────────────────────────────────────────────────
    print("\n─── ETAPA 8: Fração de Ocupação de Galáxias ──────────────────────")
    focc = calcular_focc(mc_result)
    if focc["pa"] is not None:
        print(f"  V50 analítico : {focc['pa'][0]:.2f} km/s  "
              f"(esperado ≈{VMAX_LIM_KMS} km/s)")
    if focc["pn"] is not None:
        print(f"  V50 numérico  : {focc['pn'][0]:.2f} km/s  "
              f"(deslocamento = {focc['pn'][0]-focc['pa'][0]:+.2f} km/s)")
    if focc["pc"] is not None:
        print(f"  V50 corrigido : {focc['pc'][0]:.2f} km/s  "
              f"(desvio = {focc['pc'][0]-focc['pa'][0]:+.2f} km/s)")

    # ── ETAPA 9: Figuras científicas ──────────────────────────────────────────
    print("\n─── ETAPA 9: Gerando Figuras (nível paper) ───────────────────────")
    figura_diagnostico(res_mapa, validacao, mc_result)
    figura_punch(mc_result)
    figura_focc(mc_result)
    figura_hernquist(res_hq)

    # ── ETAPA 10: CSV de auditoria ────────────────────────────────────────────
    print("\n─── ETAPA 10: CSV de Auditoria ───────────────────────────────────")
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    csv_path = os.path.join(base, "resultados_mc_gold.csv")
    exportar_csv(mc_result, csv_path)

    robustez_path = os.path.join(base, "robustez_suplementar.csv")
    exportar_csv_robustez(robustez_path, conv_msub=conv_msub,
                          sens_lbox=sens_lbox, sens_phiref=sens_phiref,
                          projecao=projecao, ambiente=ambiente)

    # ── Sumário final ─────────────────────────────────────────────────────────
    dt = time.perf_counter() - t0
    fn  = int(mc_result["falso_escuro"].sum())
    rec = int(mc_result["recuperados"].sum())
    v50_shift = (focc["pn"][0] - focc["pa"][0]) if focc["pn"] is not None and focc["pa"] is not None else float("nan")

    print("\n" + "═"*72)
    print("  GOLD STANDARD PIPELINE CONCLUÍDO")
    print("═"*72)
    print(f"  Ambiente         : {ambiente['sistema']} {ambiente['release']} | "
          f"{ambiente['n_cpus_logicos']} CPUs lógicos | "
          f"Python {ambiente['python']} | NumPy {ambiente['numpy']}")
    print(f"  Tempo total      : {dt/60:.1f} min")
    print(f"  Config           : N={N_MAP}, {N_TRANS} transl., M_sub={M_SUB}, "
          f"{N_HALOS} halos base (+{mc_result['N_halos']-mc_result['N_halos_base']} "
          f"estratificados)")
    print(f"  Erro |Φ| → PCHIP : {e_ant:.1f}% → {e_dep:.2f}%  "
          f"(redução {(1-e_dep/max(e_ant,1e-10))*100:.0f}%)  |  "
          f"máx={validacao['erro_max_dep']:.2f}% P95={validacao['erro_p95_dep']:.2f}%")
    print(f"  Acurácia (corr)  : {mc_result['acc_corr']:.1f}%  "
          f"vs. numérico {mc_result['acc_num']:.1f}%")
    print(f"  Falsos Escuros   : {fn}  |  Recuperados: {rec} "
          f"({100*rec//max(fn,1)}%)  |  n_estelar={mc_result['n_estelar']}  "
          f"IC95%(Wilson)=[{mc_result['recuperacao_ic95'][0]*100:.0f}%,"
          f"{mc_result['recuperacao_ic95'][1]*100:.0f}%]")
    print(f"  Erro MC |Φ| corr : média={mc_result['eps_corrigido']:.2f}% "
          f"máx={mc_result['eps_corrigido_max']:.2f}% "
          f"P95={mc_result['eps_corrigido_p95']:.2f}%")
    print(f"  Hernquist efic.  : {res_hq['efic']:.0f}%  "
          f"(transferabilidade do corretor)")
    print(f"  Deslocamento f_occ: {v50_shift:+.2f} km/s  (viés → corrigido)")
    print(f"  Convergência M_sub: erro(r_eval) M_sub=8→16 variou "
          f"{abs(conv_msub['erro_reval'][2]-conv_msub['erro_reval'][3]):.2f} p.p. "
          f"(teste em N=32; ver robustez_suplementar.csv)")
    print(f"  Escalabilidade   : expoente ajustado p={projecao['expoente_ajustado']:.2f} "
          f"(N³ assumiria p=3.00) | projeção N=256 ≈ "
          f"{projecao['projecao_min'][256]/60:.1f} h de CPU")
    print("\n  Saídas:")
    print("    fig1_diagnostico_gold.png")
    print("    fig2_punch_gold.png")
    print("    fig3_focc_gold.png")
    print("    fig4_hernquist_gold.png")
    print("    resultados_mc_gold.csv")
    print("    robustez_suplementar.csv")
    print("═"*72)


if __name__ == "__main__":
    main()