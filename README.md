# age-of-titans

SN Ia progenitor ages for TITAN host galaxies: host SFH posteriors (Bagpipes, Iyer+2019 SFH)
are multiplied by a power-law delay-time distribution to give each SN's progenitor-age PDF.
Code for Murakami et al., *Old Universe, Young SNe Ia*.

## Layout

| file | what it does | output |
|---|---|---|
| `config.py` | input paths, DTD variants, sample cuts, literature constants | |
| `tprog/dtd.py` | DTD, SFH x DTD -> P(t_prog), inverse-CDF sampling | |
| `tprog/data.py` | analysis sample (6,983 hosts), SF/Q x mass subsamples, readers | |
| `tprog/style.py` | matplotlib style | |
| `01_compute_progenitor_ages.py` | progenitor-age samples for all hosts and all six DTD variants (one pass over the SFH file) | `data/ages/tprog_*.h5` |
| `02_build_host_table.py` | host table + per-host E[t_prog] percentiles (baseline DTD) | `results/host_sample_with_tprog.csv` |
| `03_table_dtd_variants.py` | Table DTD_variants_stats; subsample mean ages; SF-Q age difference and predicted HR offsets | `results/dtd_variants.csv` |
| `04_fig_sfh_to_tprog.py` | Fig. SFH_to_t_prog | `figures/SFH_to_t_prog.pdf` |
| `05_fig_progenitor_age_histogram.py` | Fig. progenitor_age_histogram | `figures/progenitor_age_histogram.pdf` |
| `06_fig_hr_separation_tests.py` | Fig. HR_separation_tests | `figures/HR_separation_tests.pdf` |
| `07_fig_tsn_vs_tgal.py` | Fig. tSN_vs_tGal | `figures/tSN_vs_tGal.pdf` |
| `08_fig_age_vs_redshift.py` | Fig. age_vs_redshift | `figures/age_vs_redshift.pdf` |
| `09_fig_wdsim_s25_comparison.py` | Fig. WDsim_and_S25_comparison (appendix) | `figures/WDsim_and_S25_comparison.pdf` |

Inputs live in `data/inputs/` (host properties, reliable-fit list, d_DLR, SN table, WD
simulation grids). The SFH posterior file is set by `config.SFH_H5` or `$TPROG_SFH_H5`, and
the host-property table by `$TPROG_HOST_PROPS`.

## Running (Midway)

    sbatch slurm/01_compute_progenitor_ages.sbatch      # all hosts x 6 DTDs; resumable
    python 02_build_host_table.py
    python 03_table_dtd_variants.py
    for s in 0[4-9]_fig_*.py; do python $s; done

Paper figures made elsewhere (Tweddle): SED_to_SFH_3gals_NEW, tSN_vs_tGal_vs_HR_means,
HR_vs_z_W21_evolution_colourmatched, csfh+spad, iyer_comparison_with_tmwa, 2024vjb_*.

The exploratory notebooks this replaces (`SFH_Iyer/analysis*.ipynb`, `SFH_to_SNR*.ipynb`) are
in the git history before the restructuring commit.
