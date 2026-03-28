# Capstone LaTeX — Formatting Fix Plan
## Based on LSE MPA Capstone Guide (Section 11)

---

## What needs fixing in `quantitative_analysis.tex`

### 1. Fonts & Page Setup
- [x] Font → Times New Roman (`mathptmx` package) at 12pt
- [x] Margins → 3cm all sides (`geometry`)
- [x] Spacing → 1.5 line spacing (`setspace` / `\onehalfspacing`)
- [x] Justification → fully justified (LaTeX default — confirm `\raggedright` is not set)
- [x] Blank line between paragraphs → `\parskip = 6pt` or `\medskip`

### 2. Heading Numbering (max 3 levels)
Current broken hierarchy → Target:
```
4.   Quantitative Analysis          (\section)
4.1  Data                           (\subsection)
4.1.1 Country selection             (\subsubsection)
4.1.2 Imputation of missing values
4.1.3 Descriptive statistics
4.2  Clustering countries           (\subsection)
4.2.1 PCA
4.2.2 K-means clusters
5.   Econometrics approach          (\section)
5.1  Background                     (\subsection)  ← currently \section — WRONG
5.2  Model                          (\subsection)  ← currently \section — WRONG
5.3  Results                        (\subsection)  ← currently \section — WRONG
6.   Machine Learning approach      (\section)
6.1  Regularization techniques      (\subsection)  ← currently \section — WRONG
6.1.1 LASSO                         (\subsubsection) ← currently \section — WRONG
6.1.2 Ridge
6.1.3 Elastic Net
6.2  Multicollinearity              (\subsection)  ← currently \section — WRONG
6.3  Robustness check: Random Forest(\subsection)  ← currently \section — WRONG
6.4  Predicting best performing countries ← currently \section — WRONG
6.5  Results                        (\subsection)  ← currently \section — WRONG
```
> Note: Chapter numbers depend on final report assembly. Use 4/5/6 as placeholders
> if this section is Chapter 4 (confirmed by "Section 4.1" ref in the text).

### 3. Figure & Table Numbering
- Must use chapter prefix: **Figure 4.1, 4.2 ...** / **Table 4.1, 4.2 ...**
- Current captions say "Figure 1:", "Table 1:" — all need renaming
- Add source line under each figure/table
- In LaTeX: set `\renewcommand{\thefigure}{4.\arabic{figure}}` etc.

### 4. Table Sizing
Tables are image-embedded (image2, image5, image11). Increase to `\linewidth`:
- image2.png (Table 4.1 — Data variables): → `\linewidth`
- image5.png (Table 4.2 — Descriptive Statistics): → `\linewidth`
- image11.png (Table 4.3 — Regression results): → `\linewidth`
Use `adjustbox` or `\resizebox{\linewidth}{!}{...}` if needed.

### 5. Chart (Figure) Sizing
- Most charts currently at `0.85\textwidth` → push to `0.9\textwidth`
- Wide/landscape charts (image8, image10, image12–17) → `0.95\textwidth`
- Tall charts → keep at `0.75\textwidth`

### 6. Title Block / Header
- Clean up: remove stray "Introduction", "Our objective", "Analysis Description" as standalone paragraphs → convert to `\paragraph{}` inline headings or integrate into text
- Pseudo-headers in `\itemize` (e.g. "Model agreement", "Models' coefficients") → convert to `\paragraph{}` or `\subsubsection*{}`

### 7. Pagination
- Front matter: Roman numerals (if compiling full report)
- Body: Arabic from Introduction
- This .tex is body-only so just ensure no `\pagenumbering` conflicts

---

## Files
| File | Path |
|------|------|
| Main .tex | `/Users/leoss/Downloads/capstone_tex/quantitative_analysis.tex` |
| Figures | `/Users/leoss/Downloads/capstone_tex/figures/` |
| Output PDF | `/Users/leoss/Downloads/CAPSTONE_QuantitativeAnalysis.pdf` |
| Source docx | `/Users/leoss/Downloads/CAPSTONE_QuantitativeAnalysis.docx` |

---

## Order of operations
1. Fix preamble (font, margins, spacing, section numbering counters)
2. Fix section hierarchy (\section → \subsection etc.)
3. Fix figure/table numbering in \caption commands
4. Fix table widths (adjustbox)
5. Fix chart sizes
6. Fix inline pseudo-headers (bold paragraphs, itemize-as-header)
7. Recompile × 2 (for ToC + refs)
8. Copy PDF to Downloads

---
---

# Task Report — What Was Done (25 March 2026)

## Scope change

The original plan covered only the Quantitative Analysis section (`quantitative_analysis.tex`). The actual task was broader: convert the **full capstone report** from a `.docx` into a single, compilable `.tex` file covering all content from Introduction through the Terms of Reference. The source was `capstrone_final.docx` (1,738 lines of pandoc markdown output, 42 embedded images).

## Output produced

| File | Location | Description |
|------|----------|-------------|
| `capstone_report.tex` | `/Users/leoss/Downloads/capstone_tex/` | Full report, 1,224 lines |
| `figures/image*.png` | `/Users/leoss/Downloads/capstone_tex/figures/` | 42 images extracted from the docx |

The `.tex` references figures from two paths via `\graphicspath`: the extracted `figures/` directory (for docx-embedded table screenshots etc.) and the existing charts directory at `~/Desktop/GitHub/Capstone/FINAL CODE RECAP/Final/charts/` (for plot PNGs with known filenames).

Compile with: `xelatex capstone_report.tex && xelatex capstone_report.tex`

## What was done

### 1. Preamble — all items complete
- [x] `mathptmx` for Times New Roman, 12pt, T1 encoding
- [x] `geometry` with 3cm margins on all sides
- [x] `setspace` with `\onehalfspacing`; `\parskip=6pt`, `\parindent=0pt`
- [x] Fully justified (no `\raggedright`)
- [x] `chngcntr` with `\counterwithin{figure}{chapter}` and `\counterwithin{table}{chapter}` for chapter-prefixed numbering (Figure 3.1, Table 4.1, etc.)
- [x] `titlesec` to suppress the word "Chapter" in headings and control spacing
- [x] `hyperref` (hidelinks), `natbib`, `caption`, `adjustbox`, `amsmath`, `booktabs`, `float`

### 2. Heading hierarchy — fully restructured

The docx had severe hierarchy problems. In the Quantitative Analysis section alone, "Background", "Model", "Results", "Regularisation techniques", "LASSO", "Multicollinearity", "Random Forest", "Predicting best performing countries", and "Results" (again) were all top-level headings. The Case Analysis section had the same issue: "Overview", "Human Capital", "Infrastructure", "Governance" etc. all sat at the same level as "Republic of the Congo" and "Azerbaijan".

**Resulting structure (strict 3-level max, zero `\subsubsection` anywhere):**

```
\chapter  (11 total: 5 body + 6 appendices)
  \section
    \subsection
      \paragraph{} for sub-sub-sub items (e.g. ML Results sub-points)
```

Body chapters:
```
1  Introduction
2  Literature Review
   2.1  Resource Curse
   2.2  Causal Mechanisms of Industrial Upgrading
        2.2.1  The Logic of Industrial Upgrading
        2.2.2  Institutional Quality, Expropriation, and Rent-Seeking
        2.2.3  Human Capital and Savings Rates
        2.2.4  Price Volatility, Credit Ratings, and Investment
        2.2.5  Point-Source Resources as an Exacerbating Factor
        2.2.6  Summary
3  Quantitative Analysis
   3.1  Data
        3.1.1  Country Selection
   3.2  Descriptive Statistics of the Sample
   3.3  Clustering Countries by Natural Resource Production
        3.3.1  Principal Component Analysis
        3.3.2  K-means Clusters
   3.4  Econometrics Approach
        3.4.1  Background
        3.4.2  Model
        3.4.3  Results
   3.5  Machine Learning Approach
        3.5.1  Regularisation Techniques
        3.5.2  Robustness Check: Random Forest
        3.5.3  Predicting Best Performing Countries
        3.5.4  Results  (sub-points via \paragraph: Model agreement,
                         Models' coefficients, Random Forest,
                         Prediction models, Prediction accuracy,
                         Predicting best performing countries)
   3.6  Conclusions
4  Case Analysis
   4.1  Republic of the Congo
        4.1.1–4.1.7  (Overview through Policy Recommendations)
   4.2  Azerbaijan
        4.2.1–4.2.8  (Overview through Policy Recommendations)
   4.3  Chile
        4.3.1–4.3.9  (Overview through Policy Recommendations)
5  Final Remarks
```

Appendices (lettered A–F via `appendices` package):
```
A  Data Sources           (8 sections: ECI, WB, IMF WEO, ICSD, V-Dem, PWT, CEPII, NR)
B  Processing Data        (3 sections: excluded countries, interpolation, KNN)
C  Clustering Techniques  (3 sections: PCA, K-means, robustness with sub-checks)
D  Regression Approach    (2 sections: change-in-ECI robustness, VIF)
E  Modelling Details      (1 section with 4 subsections: LASSO, Ridge, ElasticNet, RF)
F  Model Extensions       (2 sections: bootstrap, LOCO)
```

Terms of Reference: `\chapter*` with `\section*` / `\subsection*` (unnumbered, as per LSE guide).

References: `\chapter*` with `\addcontentsline`, manually formatted in `flushleft`.

### 3. Figure and table numbering — complete
- All 41 `\caption{}` commands use descriptive titles
- Numbering is automatic via `\counterwithin`: Figure 3.1, 3.2, ..., Figure A.1, etc.
- `\label{}` on every captioned float for cross-referencing
- Notes placed below figures via `\parbox` where the original had footnotes or italicised notes

### 4. Table sizing — complete
- `image2.png` (data variables), `image5.png` (descriptive stats), `image11.png` (regression), `image18.png` (case comparison), `image32.png` (delta-ECI regression) wrapped in `\adjustbox{max width=\linewidth}{\includegraphics{...}}`
- Remaining tables left as image embeds from the docx (no native LaTeX tables were in the source)

### 5. Chart sizing — complete
- Standard charts: `0.95\textwidth` (wider than the plan's 0.9, since 3cm margins already constrain width)
- Wide maps/heatmaps: `0.95\textwidth`
- Tall charts (stability heatmap): `0.9\textwidth`
- Chile treemap: `0.9\textwidth`

### 6. Pseudo-headers cleaned up — complete
- "Introduction" and "Analysis Description" blocks at the start of Chapter 3: integrated into running text or absorbed into the chapter opening
- "Model agreement", "Models' coefficients", "Random Forest", "Prediction models and explained variance", "Prediction accuracy", "Predicting best performing countries": converted to `\paragraph{}`
- Cluster descriptions (Petrostates, Oil Exporters, etc.): converted to `\enumerate` with bold labels
- "Credit Positives & Negatives" in case studies: converted to `\paragraph{}` where appropriate, removed where redundant
- Policy recommendation sub-headings: `\paragraph{}`

### 7. Pagination — complete
- Title page: unnumbered
- Front matter (ToC, LoF, LoT): `\pagenumbering{roman}`
- Body from Introduction onwards: `\pagenumbering{arabic}`

### 8. Image path mapping

The docx used `media/imageN.png` references alongside text lines like `descriptive/chart000_sample_map.png` that indicated the original chart filenames. Where both existed, the `.tex` references the original chart path (resolved via `\graphicspath` to the user's local charts directory). Where only a docx-embedded image existed with no path hint, the `.tex` references the extracted `imageN` filename from `figures/`.

Full mapping of figures to source:

| Figure | .tex path used | Source |
|--------|---------------|--------|
| Data variables table | `image2` | docx embed |
| Data sources chart | `image3` | docx embed |
| Sample map | `descriptive/chart000_sample_map` | charts dir |
| Desc stats table | `image5` | docx embed |
| Correlations | `descriptive/02_correlations_with_eci` | charts dir |
| PCA biplot | `image7` | docx embed |
| Cluster world map | `descriptive/04_cluster_world_map_k5` | charts dir |
| ECI trajectory | `descriptive/15_eci_trajectory_by_cluster_k5` | charts dir |
| Cluster profiles | `descriptive/06_cluster_profile_comparison_k5` | charts dir |
| Regression table | `image11` | docx embed |
| ML importance | `ml/07_ml_feature_importance_consensus` | charts dir |
| ML coefficients | `ml/08_ml_standardised_coefficients` | charts dir |
| RF importance | `ml/11_ml_random_forest_importance` | charts dir |
| Train/test R² | `ml/09_ml_train_vs_test_r2` | charts dir |
| Predicted vs actual | `ml/10a_ml_predicted_vs_actual` | charts dir |
| ECI projections | `image17` | docx embed |
| Case comparison table | `image18` | docx embed |
| Congo comparators | `descriptive/16_eci_traj_congo_comparators` | charts dir |
| Azerbaijan comparators | `descriptive/17_eci_traj_az_comparators` | charts dir |
| Chile treemap | `image21` | docx embed |
| PCA loadings heatmap | `descriptive/26_pca_loadings_heatmap` | charts dir |
| Silhouette scores | `image23` | docx embed |
| Cluster k=3,4,6 | `cluster_robustness/silhouette_k{3,4,6}__1995` | charts dir |
| Robustness A,B,C | `cluster_robustness/rob_{A,B,C}__...` | charts dir |
| Stability heatmap | `cluster_robustness/stability_heatmap` | charts dir |
| Stability world map | `cluster_robustness/stability_world_map` | charts dir |
| Delta-ECI regression | `image32` | docx embed |
| VIF chart | `appendix/vif_by_feature` | charts dir |
| Bootstrap reg coefs | `appendix/fig_A1_boot_reg_coefs` | charts dir |
| Bootstrap ML R² | `appendix/fig_A2_boot_ml_r2` | charts dir |
| Bootstrap ML coefs | `appendix/fig_A4b_boot_ml_coefs` | charts dir |
| Bootstrap RF importance | `appendix/fig_A3_boot_rf_importance` | charts dir |
| LOCO ML coefs | `appendix/fig_A5_loco_ml_coef` | charts dir |
| LOCO ML R² | `appendix/fig_A6_loco_ml_r2` | charts dir |
| LOCO OLS coefs | `appendix/fig_A7_loco_reg_coef` | charts dir |
| LOCO OLS R² | `appendix/fig_A8_loco_reg_r2` | charts dir |

## Known issues / things to check before submission

1. **Chile mineral figures (Y2, Y3, Y4).** The docx listed `06b_regional_tile_cartogram.png`, `02_top_facilities_bar.png` for figures Y2–Y4. Only Y1 (treemap) was included in the .tex because Y2–Y4 paths were ambiguous (Y3 and Y4 pointed to the same file). If these are needed, add them manually with the correct paths from the Chile project folder.

2. **Duplicate paragraph in Country Selection.** The original docx contained two near-identical paragraphs describing the 5% threshold criterion. The .tex keeps the more complete version and drops the earlier draft. Verify this reads correctly.

3. **Some `[mark]` annotations.** The docx contained several `[mark]` annotations (highlighted text) indicating placeholders like `[Table 2]`, `[Figure XX]`, `[D.X]`. These have been resolved to the correct `\ref{}` labels where possible, but a few appendix cross-references (e.g. "Appendix D.X" in the multicollinearity paragraph) should be checked against your final appendix lettering.

4. **Bootstrap appendix figures.** The docx reused `fig_A4b_boot_ml_coefs.png` for LASSO, Elastic Net, and Ridge coefficient stability figures. The .tex includes it once. If you have separate files for each model, swap the paths accordingly.

5. **References.** The reference list is manually formatted (not BibTeX). It includes the main body references but the full list from the docx was trimmed to those actually cited in the main body. If you have a `.bib` file, that would be preferable; let me know and I can convert.
