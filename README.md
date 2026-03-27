# Professional Nonlinear Pushover Dashboard

A Streamlit teaching dashboard for simplified nonlinear pushover analysis of building frames up to 10 storeys.

## Features
- Up to 10 storeys
- Per-storey inputs for:
  - height
  - number of columns and beams
  - equivalent column EI
  - beam plastic moment capacity
  - column plastic moment capacity
  - seismic weight
  - ultimate drift ratio
- Load pattern options:
  - Uniform
  - Triangular
  - First-mode-like
  - User-defined
- Professional teaching outputs:
  - nonlinear pushover capacity curve
  - idealized bilinear curve
  - target roof displacement
  - hinge states: Elastic / IO / LS / CP / Failed
  - drift checks
  - soft-storey screening
  - Excel and CSV exports

## Important note
This is a **simplified educational nonlinear surrogate model**. It is intended for:
- teaching pushover concepts
- quick parametric studies
- conceptual structural behavior exploration

It is **not a substitute** for detailed nonlinear FEM analysis in ETABS, SAP2000, OpenSees, SeismoStruct, or similar software.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Suggested future enhancements
- section property library for RC and steel
- automatic beam/column capacity from section dimensions
- moment-curvature module
- FEMA 356 / ASCE 41 performance point procedure
- frame elevation sketch
- comparison with modal response spectrum or time history
