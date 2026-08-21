"""Shared publication-quality matplotlib style for the paper figures.

Uses real LaTeX (usetex) with the Times text + Times math fonts (newtx), so the
figure typography is byte-identical to the elsarticle ``times`` body font of the
manuscript. Falls back to a Times-like STIX/mathtext style if a LaTeX toolchain is
unavailable, so the scripts still run on machines without TeX.
"""
import matplotlib as mpl


def _usetex_ok():
    import shutil
    return all(shutil.which(t) for t in ("latex", "dvipng", "gs"))


def apply():
    base = {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2.3, "ytick.minor.size": 2.3,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "legend.frameon": False,
    }
    if _usetex_ok():
        base.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
            "font.family": "serif",
        })
    else:  # graceful fallback: Times-like serif with STIX math
        base.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
        })
    mpl.rcParams.update(base)
