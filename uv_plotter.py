# %% Imports and load files
import csv
import math
from pathlib import Path
import numpy as np
import pandas as pd

import cclib
import scipy.constants as constants
import matplotlib.pyplot as plot
from cclib.parser.utils import convertor


def extract_data_from_logs(logfile):
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


def generate_spectrum(
    excited_states, plt_range, grid_size, sigma=0.4, normalization=True
):
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


def uvvis_to_xyz_color(wavelengths, intensities):
    """
    Generate xyz color coordinates according to CIE 1931 standard, using a D65 illuminant

    Input: UV-Vis spectrum with wavelengths in nm and absorbance absolute values

    Output: xyz data
    """

    # Setup the CIE standard, from the csv file
    cmf = pd.read_csv("CIE-xyzs.csv", index_col=0, sep=";")
    cmf.dropna(axis=0, inplace=True)
    cmf = cmf.astype(float)
    cmf.columns = ["x", "y", "z", "S"]

    # Truncate the UV-Vis according to the CIE standard
    xmin = int(min(cmf.index))
    xmax = int(max(cmf.index))

    spectrum = pd.DataFrame(np.column_stack((wavelengths, intensities)))
    spectrum.columns = ["wavelengths", "intensities"]

    truncated_spectrum = spectrum[
        (xmin <= spectrum["wavelengths"]) & (spectrum["wavelengths"] <= xmax)
    ]

    # Convert to transmittance
    Imin = np.min(truncated_spectrum["intensities"])
    Imax = np.max(truncated_spectrum["intensities"])

    new_I = (truncated_spectrum["intensities"] - Imin) / (Imax - Imin)

    trs_I = np.power(10, -new_I)

    X = np.sum(trs_I * cmf["S"] * cmf["x"])
    Y = np.sum(trs_I * cmf["S"] * cmf["y"])
    Z = np.sum(trs_I * cmf["S"] * cmf["z"])

    den = np.sum(cmf["S"] * cmf["y"])

    if den != 0.0:
        X, Y, Z = X / den, Y / den, Z / den
    xyz = [X, Y, Z]

    return xyz


def xyz_to_RGB(xyz):
    """
    Conversion of xyz coordinates into RGB coordinates.

    - xyz: list: xyz coordinates.

    return:
    - np.array (3,): rgb coordinates.
    """

    xyz = np.array(xyz).reshape(-1, 1)

    M = np.array(
        [
            [3.2410, -1.5374, -0.4986],
            [-0.9692, 1.8760, 0.0416],
            [0.0556, -0.2040, 1.0570],
        ]
    )

    norm_rgb = np.dot(M, xyz)
    norm_rgb[norm_rgb < 0] = 0
    norm_rgb[norm_rgb > 1] = 1
    rgb = np.round(norm_rgb * 255, 0)
    return rgb.flatten().astype(int)


def xyz_to_Lab(xyz):
    """
    Conversion of xyz coordinates into Lab coordinates.

    - xyz: list: xyz coordinates.

    return:
    - np.array (3,): Lab coordinates.
    """

    def f(T, eps):
        if T > eps:
            return np.power(T, 1 / 3)
        return np.power(29 / 6, 2) / 3 * T + 4 / 29

    eps = np.power(6 / 29, 3)

    cmf = pd.read_csv("CIE-xyzs.csv", index_col=0, sep=";")
    cmf.dropna(axis=0, inplace=True)
    cmf = cmf.astype(float)
    cmf.columns = ["x", "y", "z", "S"]

    Xref = np.sum(cmf["S"] * cmf["x"]) / np.sum( cmf["S"] * cmf["y"])
    Yref = 1
    Zref = np.sum(cmf["S"] * cmf["z"]) / np.sum( cmf["S"] * cmf["y"])


    X, Y, Z = xyz
    L = 116 * f(Y / Yref, eps) - 16
    a = 500 * (f(X / Xref, eps) - f(Y / Yref, eps))
    b = 200 * (f(Y / Yref, eps) - f(Z / Zref, eps))

    return [L, a, b]


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

write_data = False
plot_data = False
generate_Lab = True

for file in files:
    # Extract data and generate the line to plot
    data = extract_data_from_logs(file.as_posix())
    uv_spectrum = generate_spectrum(
        data, plot_range, plot_grid, sigma, normalization=False
    )

    # Reformat data
    wavelengths = [point[0] for point in uv_spectrum]
    absorbances = [point[1] for point in uv_spectrum]

    absorbance_max_idx = np.argmax(absorbances)
    lambda_max = wavelengths[absorbance_max_idx]

    print(file.stem + ": " + str(lambda_max))

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

    if generate_Lab:
        xyz = uvvis_to_xyz_color(wavelengths, absorbances)
        rgb = xyz_to_RGB(xyz)
        Lab = xyz_to_Lab(xyz)
        print(rgb)
        print(Lab)
