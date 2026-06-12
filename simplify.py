"""
RO9 — Chen2020 callable parameter handling for DFN sensitivity analysis.

- κ, D_e: 상수화 (reference state c_e=1000 mol/m³, T=298.15 K)
- j₀±:    m_ref 를 InputParameter 로 노출 (custom function wrapping)

Usage:
    from ro9_utils import (
        get_nominals,
        j0_neg_with_mref, j0_pos_with_mref,
        apply_kappa, apply_De, apply_j0n, apply_j0p,
        make_base_param, get_base_inputs,
    )
"""

import pybamm


# ══════════════════════════════════════════════════════════════
# Reference state
# ══════════════════════════════════════════════════════════════
C_E_REF = 1000.0
T_REF_K = 298.15


# ══════════════════════════════════════════════════════════════
# Nominal value 계산 (모듈 로드 시 한 번)
# ══════════════════════════════════════════════════════════════
def _compute_nominals():
    p = pybamm.ParameterValues("Chen2020")
    kappa_nom = float(p["Electrolyte conductivity [S.m-1]"](C_E_REF, T_REF_K))
    De_nom    = float(p["Electrolyte diffusivity [m2.s-1]"](C_E_REF, T_REF_K))
    m_ref_neg_nom = 6.48e-7
    m_ref_pos_nom = 3.42e-6
    return kappa_nom, De_nom, m_ref_neg_nom, m_ref_pos_nom


KAPPA_NOMINAL, DE_NOMINAL, M_REF_NEG_NOMINAL, M_REF_POS_NOMINAL = _compute_nominals()


def get_nominals():
    """Return (kappa, D_e, m_ref_neg, m_ref_pos) nominal values."""
    return KAPPA_NOMINAL, DE_NOMINAL, M_REF_NEG_NOMINAL, M_REF_POS_NOMINAL


# ══════════════════════════════════════════════════════════════
# Custom j0 functions (m_ref 노출)
# ══════════════════════════════════════════════════════════════
def j0_neg_with_mref(c_e, c_s_surf, c_s_max, T):
    """Anode exchange-current density with m_ref^- as InputParameter."""
    m_ref = pybamm.InputParameter("Negative electrode exchange-current density [A.m-2]")
    arrh = pybamm.exp(35000 / pybamm.constants.R * (1/298.15 - 1/T))
    return m_ref * arrh * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5


def j0_pos_with_mref(c_e, c_s_surf, c_s_max, T):
    """Cathode exchange-current density with m_ref^+ as InputParameter."""
    m_ref = pybamm.InputParameter("Positive electrode exchange-current density [A.m-2]")
    arrh = pybamm.exp(17800 / pybamm.constants.R * (1/298.15 - 1/T))
    return m_ref * arrh * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5


# ══════════════════════════════════════════════════════════════
# Parameter set modifiers
# ══════════════════════════════════════════════════════════════
def apply_kappa(p):
    """In-place: replace κ with constant nominal value."""
    p["Electrolyte conductivity [S.m-1]"] = KAPPA_NOMINAL


def apply_De(p):
    """In-place: replace D_e with constant nominal value."""
    p["Electrolyte diffusivity [m2.s-1]"] = DE_NOMINAL


def apply_j0n(p):
    """In-place: replace j₀⁻ with custom function exposing m_ref^-."""
    p["Negative electrode exchange-current density [A.m-2]"] = j0_neg_with_mref


def apply_j0p(p):
    """In-place: replace j₀⁺ with custom function exposing m_ref^+."""
    p["Positive electrode exchange-current density [A.m-2]"] = j0_pos_with_mref


# ══════════════════════════════════════════════════════════════
# Convenience: 4개 callable 모두 처리된 base parameter set
# ══════════════════════════════════════════════════════════════
def make_base_param():
    """
    Return Chen2020 ParameterValues with all 4 callable parameters
    transformed for SDAE sensitivity:
      - κ, D_e: constant
      - j₀±:    custom function with m_ref exposed
    """
    p = pybamm.ParameterValues("Chen2020")
    apply_kappa(p)
    apply_De(p)
    apply_j0n(p)
    apply_j0p(p)
    return p


def get_base_inputs():
    """
    Return dict of nominal InputParameter values that must be passed
    to sim.solve(inputs=...) whenever j₀± custom functions are active.
    """
    return {
        "Negative electrode exchange-current density [A.m-2]": M_REF_NEG_NOMINAL,
        "Positive electrode exchange-current density [A.m-2]": M_REF_POS_NOMINAL,
    }