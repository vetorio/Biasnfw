# NFW Mini-Halo Bias Correction Pipeline — Gold Standard (v6)

Pipeline científico para correção do viés numérico em simulações cosmológicas de mini-halos de matéria escura com perfil NFW/Hernquist, utilizando correção via PCHIP 2D e validação contra o critério de Nadler (2025).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-≥1.24-blue)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-≥1.10-blue)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Descrição

Este código implementa um pipeline completo para correção do **viés de suavização numérica** que afeta resolvedores de Poisson via FFT com deposição Cloud‑in‑Cell (CIC) quando aplicados a perfis de cúspide \( \rho \propto r^{-1} \) (NFW e Hernquist). O viés “apaga” galáxias limítrofes, subestimando a profundidade do poço de potencial e alterando a classificação estelar/escura segundo o critério de Nadler (2025).

A correção é feita por um **corretor PCHIP 2D** calibrado no perfil NFW e posteriormente testado em Hernquist para demonstrar transferabilidade. O pipeline gera figuras de nível de publicação e uma tabela CSV de auditoria com 21 colunas por halo.

---

## 🎯 Objetivo Científico

- Corrigir a suavização numérica da cúspide central em resolvedores de Poisson via FFT/CIC.
- Demonstrar que a correção PCHIP radial no espaço real é robusta para qualquer perfil de cúspide \( r^{-1} \) (NFW e Hernquist).
- Validar a correção com o **Critério de Nadler (2025)**, que classifica halos como estelares ou escuros com base em:
  - Score de profundidade \( |\Phi(R_\text{eval})| / |\Phi_\text{ref}| > 0.85 \)
  - \( V_\text{max} > 12 \, \text{km/s} \)

---

## 🔬 Metodologia

O pipeline executa **10 etapas** principais:

1. **Validação analítica** dos perfis NFW e Hernquist (potencial, densidade, massa acumulada).
2. **Smoke test** do solver CIC‑FFT com resolução reduzida.
3. **Mapeamento do erro radial** \( \varepsilon(r,c) \) – média sobre 100 translações aleatórias para 5 concentrações.
4. **Construção do CorretorPCHIP 2D** interpolação monotônica em \( r \) e \( c \).
5. **Validação cruzada** NFW: compara \( N_\text{ref} \) vs \( N_\text{cal} \) vs \( N_\text{cal}+\text{PCHIP} \).
6. **Teste de sensibilidade** aplicando o corretor (calibrado em NFW) ao perfil de Hernquist.
7. **Monte Carlo** com 100 halos na zona de transição \( [10^7, 10^{8.5}] \, M_\odot \), com amostragem estratificada para garantir halos estelares.
8. **Fração de Ocupação de Galáxias** \( f_\text{occ}(V_\text{max}) \) – curva logística.
9. **Geração de figuras** científicas (diagnóstico, “punch”, f_occ, Hernquist).
10. **Exportação de CSV de auditoria** com 21 colunas por halo.

---

## 🛠️ Requisitos

- Python 3.9 ou superior
- Bibliotecas:
  - `numpy`
  - `scipy` (interpolação, otimização)
  - `matplotlib`
  - `csv` (embutido)
  - `datetime`, `os`, `gc`, `warnings`, etc.

Instale as dependências com:

```bash
pip install numpy scipy matplotlib
```

---

## 🚀 Como Executar

Clone o repositório e execute o script:

```bash
python biasnfw.py
```

### Parâmetros de Execução (dentro da função `main()`)

| Modo          | `N_MAP` | `N_TRANS` | `M_SUB` | `N_HALOS` | Tempo estimado |
|---------------|---------|-----------|---------|-----------|----------------|
| **Dev** (rápido) | 64      | 10        | 4       | 30        | ~10 min        |
| **Publicação**   | 128     | 100       | 8       | 100       | ~3–6 h (CPU)   |

Ajuste os valores no início da função `main()` conforme sua necessidade.

---

## 📁 Saídas Geradas

Após a execução, os seguintes arquivos serão criados no diretório atual:

- `fig1_diagnostico_gold.png` – Diagnóstico 6‑painéis (perfis, viés, validação, scatter, tabela de confusão).
- `fig2_punch_gold.png` – “The Punch” aprimorado com zona de recuperação física e setas de correção.
- `fig3_focc_gold.png` – Fração de Ocupação de Galáxias (curvas logísticas).
- `fig4_hernquist_gold.png` – Teste de sensibilidade Hernquist (transferabilidade do corretor).
- `resultados_mc_gold.csv` – Tabela de auditoria com 21 colunas por halo (inclui logM, c, Vvir, Vmax, scores, classificações, flags de recuperação e estrato).
- `robustez_suplementar.csv` – Estudos de convergência M_sub, sensibilidade L_caixa, sensibilidade Φ_ref e projeção de escalabilidade N=256.

---

## 🧬 Estrutura do Código

O código é organizado em seções (cada uma com documentação extensa):

- **§ A** – Constantes globais (cosmologia, critérios, paleta)
- **§ B** – Perfis analíticos (NFW e Hernquist) com potencial, densidade e massa
- **§ C** – Deposição CIC (loop iterativo com sub‑voxels)
- **§ D** – Solver de Poisson via FFT (com shift de referência)
- **§ E** – Pipeline CIC → FFT
- **§ F** – Mapeamento do erro radial com bootstrap (intervalos de confiança)
- **§ G** – Corretor PCHIP 2D
- **§ H** – Validação cruzada NFW
- **§ I** – Teste de sensibilidade Hernquist
- **§ I2** – Estudos adicionais de robustez (convergência M_sub, L_box, Φ_ref, escalabilidade)
- **§ J** – Critério de Nadler (2025)
- **§ K** – Monte Carlo (com amostragem estratificada)
- **§ L** – Fração de ocupação de galáxias
- **§ M** – Exportação CSV (21 colunas)
- **§ N–R** – Geração das figuras
- **§ S** – Pipeline principal `main()`

---

## 📊 Exemplo de Resultados

Para a configuração de **publicação** (N_MAP=128, N_TRANS=100, N_HALOS=100), espera‑se:

- Redução do erro médio de ~13‑20% para **< 0.5%** (em r_eval).
- Acurácia de classificação > 98% após correção.
- Recuperação de **100% dos falsos escuros** (halos que seriam erroneamente classificados como escuros pelo solver numérico).
- Deslocamento da curva \( f_\text{occ} \) corrigido para \( V_{50} \approx 12 \, \text{km/s} \).

---

## 📚 Referências Científicas

1. Łokas & Mamon (2001). *MNRAS* **321**, 155. – Potencial NFW analítico.
2. Navarro, Frenk & White (1997). *ApJ* **490**, 493. – Perfil NFW.
3. Nadler et al. (2025). *ApJ* [in press]. – Critério dual V_max + score.
4. Hockney & Eastwood (1988). *CRC Press*. – CIC + FFT de Poisson.
5. Fritsch & Carlson (1980). *SIAM J. Numer. Anal.* **17**. – PCHIP monótono.
6. Hernquist (1990). *ApJ* **356**, 359. – Perfil de Hernquist.
7. Vale & Ostriker (2004). *MNRAS* **353**, 189. – Função de ocupação de halos.

---

## 🤝 Como Contribuir

Este projeto é voltado para pesquisa acadêmica. Sugestões, relatórios de bugs e pull requests são bem‑vindos. Para dúvidas, entre em contato.



## ✍️ Autores

Jonatas Vitório

*Última atualização: 2026*
