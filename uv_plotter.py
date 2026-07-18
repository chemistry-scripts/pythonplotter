import csv
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import cclib
import scipy.constants as constants
import matplotlib.pyplot as plot
from cclib.parser.utils import convertor


def extract_data_from_logs(logfile, correct_wavelength=False, wv_correction=0):
    """Extract excited state energies and oscillator strength from logfile

    Fill excited_states with tuples containing wavelength in nm and oscillator strength.
    """
    excited_states = list()

    parser = cclib.io.ccopen(logfile)
    data = parser.parse()

    for wv, os in zip(data.etenergies, data.etoscs):
        wv = convertor(wv, "wavenumber", "nm")
        if correct_wavelength:
            wv = wv + wv_correction
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
    grid = range(plt_range[0], plt_range[1] + 1, grid_size)

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


def get_colorscheme():
    # Setup the CIE standard, from the csv file
    cmf = pd.read_csv("CIE-xyzs.csv", index_col=0, sep=";")
    cmf.dropna(axis=0, inplace=True)
    cmf = cmf.astype(float)
    cmf.columns = ["x", "y", "z", "S"]
    return cmf


def uvvis_to_xyz_color(
    wavelengths, intensities, plot_grid, normalize=True, length=1, concentration=1
):
    """
    Generate xyz color coordinates according to CIE 1931 standard, using a D65 illuminant

    Input: UV-Vis spectrum with wavelengths in nm and absorbance absolute values

    Output: xyz data
    """

    cmf = get_colorscheme()
    # Truncate the UV-Vis according to the CIE standard
    xmin = int(min(cmf.index))
    xmax = int(max(cmf.index))

    spectrum = pd.DataFrame(np.column_stack((wavelengths, intensities)))
    spectrum.columns = ["wavelengths", "intensities"]

    truncated_spectrum = spectrum[
        (xmin <= spectrum["wavelengths"]) & (spectrum["wavelengths"] <= xmax)
    ]

    # Filter cmf where wavelengths are needed
    wv_list = [int(i) for i in truncated_spectrum["wavelengths"]]
    truncated_cmf = cmf.loc[wv_list]

    # Convert to transmittance
    if normalize:
        Imin = np.min(truncated_spectrum["intensities"])
        Imax = np.max(truncated_spectrum["intensities"])

        new_I = (truncated_spectrum["intensities"] - Imin) / (Imax - Imin)
    else:
        new_I = truncated_spectrum["intensities"] * concentration * length

    trs_I = np.power(10, -new_I)

    X = np.sum(
        trs_I.to_numpy() * truncated_cmf["S"].to_numpy() * truncated_cmf["x"].to_numpy()
    )
    Y = np.sum(
        trs_I.to_numpy() * truncated_cmf["S"].to_numpy() * truncated_cmf["y"].to_numpy()
    )
    Z = np.sum(
        trs_I.to_numpy() * truncated_cmf["S"].to_numpy() * truncated_cmf["z"].to_numpy()
    )

    den = np.sum(truncated_cmf["S"] * truncated_cmf["y"])

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

    cmf = get_colorscheme()

    Xref = np.sum(cmf["S"] * cmf["x"]) / np.sum(cmf["S"] * cmf["y"])
    Yref = 1
    Zref = np.sum(cmf["S"] * cmf["z"]) / np.sum(cmf["S"] * cmf["y"])

    X, Y, Z = xyz
    L = 116 * f(Y / Yref, eps) - 16
    a = 500 * (f(X / Xref, eps) - f(Y / Yref, eps))
    b = 200 * (f(Y / Yref, eps) - f(Z / Zref, eps))

    return [L, a, b]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="UV-Vis Spectrum Plotter - Generate spectra and color coordinates from log files"
    )

    # File input
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/color-prediction-neutral",
        help="Directory containing .log files to process"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Single .log file to process (overrides --input-dir)"
    )

    # Spectrum generation parameters
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.4,
        help="Broadening for Gaussian functions, in eV (default: 0.4)"
    )
    parser.add_argument(
        "--plot-range",
        type=int,
        nargs=2,
        default=[250, 800],
        metavar=("MIN", "MAX"),
        help="Range of spectrum to display in nm (default: 250 800)"
    )
    parser.add_argument(
        "--plot-grid",
        type=int,
        default=1,
        help="Grid precision (distance between two generated points) (default: 1)"
    )
    parser.add_argument(
        "--wv-correction",
        type=int,
        default=27,
        help="Wavelength correction shift in nm (default: 27)"
    )

    # Color calculation parameters
    parser.add_argument(
        "--concentration",
        type=float,
        default=0.000275,
        help="Sample concentration (default: 0.000275)"
    )
    parser.add_argument(
        "--path-length",
        type=float,
        default=0.1,
        help="Path length in cm (default: 0.1)"
    )

    # Output options
    parser.add_argument(
        "--write-data",
        action="store_true",
        default=True,
        help="Write CSV data files (default: True)"
    )
    parser.add_argument(
        "--no-write-data",
        action="store_false",
        dest="write_data",
        help="Disable writing CSV data files"
    )
    parser.add_argument(
        "--plot-data",
        action="store_true",
        default=True,
        help="Generate plots (default: True)"
    )
    parser.add_argument(
        "--no-plot-data",
        action="store_false",
        dest="plot_data",
        help="Disable generating plots"
    )
    parser.add_argument(
        "--generate-Lab",
        action="store_true",
        default=True,
        help="Generate Lab color coordinates (default: True)"
    )
    parser.add_argument(
        "--no-generate-Lab",
        action="store_false",
        dest="generate_Lab",
        help="Disable generating Lab color coordinates"
    )
    parser.add_argument(
        "--correct-wavelength",
        action="store_true",
        default=False,
        help="Apply wavelength correction (default: False)"
    )
    parser.add_argument(
        "--normalize-data",
        action="store_true",
        default=False,
        help="Normalize data (default: False)"
    )

    return parser.parse_args()

### Start of main function

def main():
    args = parse_arguments()

    # List of files to plot
    if args.file:
        files_root = Path(args.file).parent
        files = [Path(args.file)] if Path(args.file).suffix == ".log" else []
    else:
        files_root = Path(args.input_dir)
        files = list()
        for file in files_root.iterdir():
            if file.is_file() and file.suffix == ".log":
                files.append(file)

    # General parameters
    sigma = args.sigma
    plot_range = args.plot_range
    plot_grid = args.plot_grid
    wv_correction = args.wv_correction
    concentration = args.concentration
    path_length = args.path_length

    # What to do here
    write_data = args.write_data
    plot_data = args.plot_data
    generate_Lab = args.generate_Lab
    correct_wavelength = args.correct_wavelength
    normalize_data = args.normalize_data

    for file in files:
        # Extract data and generate the line to plot
        data = extract_data_from_logs(file.as_posix(), correct_wavelength, wv_correction)
        uv_spectrum = generate_spectrum(
            data, plot_range, plot_grid, sigma, normalization=normalize_data
        )

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

            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Normalized absorbance")
            ax.set_xlim(left=plot_range[0], right=plot_range[1])
            ax.set_title(file.stem)
            ax.plot(wavelengths, absorbances)
            fig.show()

            # Save image
            img_file = Path(files_root, file.stem + ".png")
            fig.savefig(img_file, dpi=300)
            plot.close()

        if generate_Lab:
            xyz = uvvis_to_xyz_color(
                wavelengths,
                absorbances,
                plot_grid,
                normalize=normalize_data,
                length=path_length,
                concentration=concentration,
            )
            rgb = xyz_to_RGB(xyz)
            Lab = xyz_to_Lab(xyz)

            xyz = [str(float(i)) for i in xyz]
            rgb = [str(int(i)) for i in rgb]
            Lab = [str(float(i)) for i in Lab]

            with open(Path(files_root, file.stem + "_color.dat"), "w") as f:
                f.write(file.stem + " xyz = " + ", ".join(xyz) + "\n")
                f.write(file.stem + " rgb = " + ", ".join(rgb) + "\n")
                f.write(file.stem + " Lab = " + ", ".join(Lab) + "\n")

    if __name__ == "__main__":
        main()
