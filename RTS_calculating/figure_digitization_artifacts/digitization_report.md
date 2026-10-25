Digitization execution report
Working directory: C:\Users\HP\Documents\Harry\XRTS\RTS_calculating
Source: figure.png, panel a, 1019 x 971 pixels. Axis E [eV] includes negative and positive values about zero; incident E0=2.96 keV separately annotated. No energy conversion applied.
Calibration uses zero-based original image coordinates, sixfold nearest-neighbor visual crops, RGB channels each below 80, midpoint of contiguous black tick pixels.
X tick black columns 507-509: center 508 -> 0 eV. X tick black columns 807-809: center 808 -> 40 eV.
Y tick at column 214, black rows 787-789: center 788 -> 0. Y tick at column 214, black rows 284-286: center 285 -> 2.
Extraction: distill_data.py; smoothing: flatness_data.py. Changed only USE_INTERACTIVE and MANUAL_CALIB. Original saved in distill_data.py.before_calibration.
Smoothing parameters unchanged: green sigma=2.5; blue sigma=1.5; GAP_THRESHOLD=5.0.
Both extraction comparisons opened: red points follow corresponding green/blue curves without obvious deviation.
Both smoothing comparisons opened: experimental main peak and right-hand broad feature retained. Negative experimental noise retained. SIF peak near zero.
File | rows | x min | x max | y min | y max | peak x
exp_0.txt | 1205 | -39.8666666667 | 57.3333333333 | -0.190854870775 | 2.46918489066 | -1.46666666667
sif_0.txt | 643 | -37.8666666667 | 47.4666666667 | 0.00198807157058 | 2.36381709742 | -0.133333333333
exp_data.txt | 731 | -39.8666666667 | 57.3333333333 | -0.14601569187 | 2.2678390266 | -1.333333343
sif_data.txt | 642 | -37.8666666667 | 47.4666666667 | 0.00198807157058 | 2.3455024776 | -0.1333333428
Both final files: two numeric columns, no headers, all finite, strictly increasing x.
run_rms_Dornheim.py is in this directory. No copies needed. RMS fitting not run.
Final paths:
C:\Users\HP\Documents\Harry\XRTS\RTS_calculating\exp_data.txt
C:\Users\HP\Documents\Harry\XRTS\RTS_calculating\sif_data.txt
Local .venv dependencies installed; pip check passed. All other scripts and source figure retain original SHA-256 hashes. AST check confirms only allowed assignments changed.
