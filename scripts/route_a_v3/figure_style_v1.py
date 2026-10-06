#!/usr/bin/env python3
"""figure_style_v1: shared publication style for DeltaBench main figures.

W2 pass 2 (2026-10-07): the four main-figure producers historically used
per-figure ad-hoc rcParams (DejaVu Sans 11 / dpi 200-300 mixed / mixed tick
sizes). This module centralizes the style so all four render consistently:

  - font: DejaVu Sans (matplotlib default, always available), base size 9
  - raster dpi 300 (PDFs stay vector; dpi only affects the PNG twins)
  - axes.linewidth 0.8, consistent tick/label sizes
  - unified "figure manifest" recording the style version per render

Usage in each producer (idempotent, does not change any plotted value):
    from figure_style_v1 import apply
    apply()   # sets rcParams; returns the style version string for manifests

The producers already assert their plotted values against frozen artifacts
(figure1: matrix JSON families; figure4: bottomline frozen-vs-figure
abs<5e-5; first_order/density: values embedded from frozen JSONs). Re-render
under this style therefore changes pixels, never numbers.
"""
STYLE_VERSION = "figure_style_v1 (2026-10-07): DejaVu Sans base 9, axes.linewidth 0.8, raster dpi 300, unified tick sizes 7.5/8.5, axis label 9.0, title 10.5"

_RC = {
    "font.family": ["DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply() -> str:
    import matplotlib

    matplotlib.rcParams.update(_RC)
    return STYLE_VERSION
