import numpy as np
import h5py
import glob
import os
import illustris_python as il


def compute_gas_temperature(u, xe, UnitMass_in_g=1.989e43, UnitLength_in_cm=3.085678e21, UnitTime_in_s=3.15576e16):

    # Constants
    gamma = 5.0 / 3.0
    XH = 0.76                       # Hydrogen mass fraction
    mp = 1.6726219e-24              # Proton mass [g]
    k_B = 1.380649e-16              # Boltzmann constant [erg/K]

    # Code energy unit in CGS: UnitEnergy = UnitMass * UnitLength^2 / UnitTime^2
    UnitEnergy_in_erg = UnitMass_in_g * UnitLength_in_cm**2 / UnitTime_in_s**2

    # Mean molecular weight μ = 4 / (1 + 3*XH + 4*XH*xe) * mp
    mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * xe) * mp

    # Temperature formula:
    # T = (γ - 1) * u * UnitEnergy/UnitMass * μ / k_B
    T = (gamma - 1.0) * u * (UnitEnergy_in_erg / UnitMass_in_g) * mu / k_B

    return T
