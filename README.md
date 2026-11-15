# pn-junction-devsim
# Silicon pn junction and solar cell (DEVSIM)
- devsim_pn.py : device model (mesh, physics, sweeps, current read-out)
- run_all.py   : runs all cases in parallel, makes plots and CSV files
- extra.py     : validation numbers and solar-cell design study

Requirements: Python 3, devsim, numpy, matplotlib
Install: pip install devsim numpy matplotlib
Run: python run_all.py   (or: python run_all.py serial)
Outputs: JV_*.png, bands_*.png, solar_results.csv, thickness_table.csv
