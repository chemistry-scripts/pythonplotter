# %% Imports and load files
import csv
import math
from pathlib import Path
import numpy as np

import cclib
import scipy.constants as constants
import matplotlib.pyplot as plot
from cclib.parser.utils import convertor


def extract(logfile):
    """Extract excited state energies and oscillator strength from logfile

    Fill excited_states with tuples containing wavelength in nm and oscillator strength.
    """
    excited_states = list()

    parser = cclib.io.ccopen(logfile)
    data = parser.parse()

    for wv, os in zip(data.etenergies, data.etoscs):
        wv = convertor(wv, "wavenumber", "nm")
        excited_states.append((wv, os))

    return excited_states


def generate(excited_states, plt_range, grid_size, sigma=0.4, normalization=True):
    """Generate gaussian functions for each excitation energy, and sum them all

    - Generate a range of wavelengths to use to generate the spectrum
    - At each point, compute the absorbance including the contribution of all excitations
    - Normalize according to the max absorbance if necessary
    - Return a 2D line ready to be plotted.
    """

    # Generate the range of wavelengths
    grid = range(plt_range[0], plot_range[1] + 1, grid_size)

    # Generate data for each point
    uv_spectrum = list()

    electron_charge = 4.803204e-10
    electron_mass = 9.10938e-31

    cnst = (
        math.sqrt(math.pi)
        * electron_charge**2
        * constants.Avogadro
        / (1000 * math.log(10) * constants.c**2 * electron_mass)
    )
    cnst = 1.3062974e8

    sigma = (
        np.power(10, 9) * constants.Planck * constants.c / 1.602176634e-19 / sigma
    )  # convert to wavenumber

    for point in grid:
        fit = 0.0
        for wv, os in excited_states:
            fit += (
                cnst
                * os
                / sigma
                * np.exp(-np.power((10**7 / point - 10**7 / wv) / (sigma), 2))
            )
        uv_spectrum += [(point, fit)]

    # Normalize if requested
    if normalization:
        max_energy = max(point[1] for point in uv_spectrum)  # Retrieve max energy
        uv_spectrum = [
            (wavelength, energy / max_energy) for (wavelength, energy) in uv_spectrum
        ]

    # Return final spectrum line
    return uv_spectrum


# %%
# List of files to plot
files_root = Path("data/color-prediction/")
files = list()
for file in files_root.iterdir():
    if file.is_file() and file.suffix == ".log":
        files.append(file)

sigma = 0.4  # Broadening for Gaussians, in eV
plot_range = [250, 800]  # Range of spectrum to display in nm
plot_grid = 1  # Grid precision (distance between two generated points)

write_data = True
plot_data = False

for file in files:
    # Extract data and generate the line to plot
    data = extract(file.as_posix())
    uv_spectrum = generate(data, plot_range, plot_grid, sigma, normalization=False)

    # Reformat data
    wavelengths = [point[0] for point in uv_spectrum]
    absorbances = [point[1] for point in uv_spectrum]

    absorbance_max_idx = np.argmax(absorbances)
    lambda_max = wavelengths[absorbance_max_idx]

    if write_data:
        csv_file = Path(files_root, file.stem + ".csv")
        with open(csv_file, "w") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows(zip(wavelengths, absorbances))
        with open(Path(files_root, file.stem + "_max.dat"), "w") as f:
            f.write(file.stem + " lambda_max = " + str(lambda_max) + "\n")

    if plot_data:
        # Setup the plot
        fig, ax = plot.subplots()

        ax.set_xlabel("Energy (nm)")
        ax.set_ylabel("Normalized absorbance")
        ax.set_title(file.stem)
        ax.plot(wavelengths, absorbances)
        fig.show()

        # Save image
        img_file = Path(files_root, file.stem + ".png")
        fig.savefig(img_file, dpi=300)
