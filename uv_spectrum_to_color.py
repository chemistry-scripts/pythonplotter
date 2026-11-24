import argparse
import csv
import logging
import sys
from pathlib import Path

import cclib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cclib.parser.utils import convertor
from scipy import constants

# Colour-matching function data lives next to this script, not in the CWD.
_CIE_FILE = Path(__file__).with_name("CIE-xyzs.csv")
_cmf_cache = None  # loaded lazily and cached (it is read several times)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("uv_plotter")


def extract_data_from_logs(logfile, correct_wavelength=False, wv_correction=0):
    """Extract excited state energies and oscillator strength from a logfile.

    Returns a list of ``(wavelength_nm, oscillator_strength)`` tuples.
    Raises ``ValueError`` if no excited-state data can be found.
    """
    parser = cclib.io.ccopen(logfile)
    data = parser.parse()

    if data is None or data.etenergies is None or data.etoscs is None:
        raise ValueError(f"no excited-state data found in {logfile}")

    excited_states = []
    for wv, osc in zip(data.etenergies, data.etoscs):
        wv = convertor(wv, "wavenumber", "nm")
        if correct_wavelength:
            wv = wv + wv_correction
        excited_states.append((float(wv), float(osc)))
    return excited_states


def extract_spectrum(csvfile, delimiter=";", has_header=False):
    """Read an experimental spectrum (wavelength, absorbance) from a CSV file.

    By default a ``;`` separator and no header row are assumed. Set
    ``has_header=True`` to skip the first line.
    """
    header = 0 if has_header else None
    data = pd.read_csv(csvfile, sep=delimiter, header=header).to_numpy()
    if data.shape[1] < 2:
        raise ValueError(f"expected at least 2 columns in {csvfile}")

    return [(float(row[0]), float(row[1])) for row in data]


def generate_spectrum(
    excited_states, plt_range, grid_size, sigma=0.4, normalization=True
):
    """Generate a Gaussian-broadened absorption spectrum.

    - Builds a wavelength grid between ``plt_range[0]`` and ``plt_range[1]``
      with step ``grid_size``.
    - At each point, sums the Gaussian contribution of every excitation.
    - Optionally normalizes so the peak absorbance equals 1.
    - Returns a list of ``(wavelength, absorbance)`` tuples.
    """
    grid = range(plt_range[0], plt_range[1] + 1, grid_size)

    if not excited_states:
        return [(point, 0.0) for point in grid]

    # Peak amplitude constant and sigma converted from eV to wavenumber (cm^-1).
    cnst = 1.3062974e8
    sigma_wavenumber = (
        1e9 * constants.Planck * constants.c / constants.eV / sigma
    )  # cm^-1

    uv_spectrum = []
    for point in grid:
        fit = 0.0
        for wv, osc in excited_states:
            fit += (
                cnst
                * osc
                / sigma_wavenumber
                * np.exp(-np.power((1e7 / point - 1e7 / wv) / sigma_wavenumber, 2))
            )
        uv_spectrum.append((float(point), float(fit)))

    if normalization:
        max_absorbance = max(value for _, value in uv_spectrum)
        if max_absorbance > 0:
            uv_spectrum = [
                (wavelength, value / max_absorbance)
                for (wavelength, value) in uv_spectrum
            ]

    return uv_spectrum


def get_colorscheme():
    """Return the cached CIE 1931 standard observer colour-matching functions."""
    global _cmf_cache
    if _cmf_cache is None:
        cmf = pd.read_csv(_CIE_FILE, index_col=0, sep=";")
        cmf.dropna(axis=0, inplace=True)
        cmf = cmf.astype(float)
        cmf.columns = ["x", "y", "z", "S"]
        _cmf_cache = cmf
    return _cmf_cache


def uvvis_to_xyz_color(
    wavelengths, intensities, normalize=True, length=1, concentration=1
):
    """Generate xyz color coordinates (CIE 1931, D65 illuminant).

    Input: UV-Vis spectrum with wavelengths in nm and absorbance values.
    Output: ``[X, Y, Z]`` coordinates.
    """
    cmf = get_colorscheme()

    # Truncate the UV-Vis according to the CIE standard range.
    xmin = int(min(cmf.index))
    xmax = int(max(cmf.index))

    spectrum = pd.DataFrame(np.column_stack((wavelengths, intensities)))
    spectrum.columns = ["wavelengths", "intensities"]

    truncated_spectrum = spectrum[
        (xmin <= spectrum["wavelengths"]) & (spectrum["wavelengths"] <= xmax)
    ]

    # Filter the colour-matching functions at the required wavelengths.
    wv_list = [int(i) for i in truncated_spectrum["wavelengths"]]
    truncated_cmf = cmf.loc[wv_list]

    # Convert absorbance to transmittance.
    if normalize:
        Imin = np.min(truncated_spectrum["intensities"])
        Imax = np.max(truncated_spectrum["intensities"])
        if Imax == Imin:  # flat spectrum -> avoid division by zero
            new_I = np.zeros_like(truncated_spectrum["intensities"])
        else:
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

    return [X, Y, Z]


def xyz_to_RGB(xyz):
    """Convert xyz coordinates into sRGB coordinates (0-255 integers)."""
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
    """Convert xyz coordinates into CIE L*a*b* coordinates (D65 white point)."""

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


def normalize_spectrum(wavelengths, absorbances):
    """Min-max normalize absorbances; returns new (wavelengths, absorbances)."""
    values = np.asarray(absorbances, dtype=float)
    lo, hi = values.min(), values.max()
    if hi == lo:  # flat spectrum
        return list(wavelengths), np.zeros_like(values).tolist()
    normalized = (values - lo) / (hi - lo)
    return list(wavelengths), normalized.tolist()


# The two file types are mutually exclusive: a computation is a .log/.out file,
# an experimental spectrum is a .csv file. Keeping them separate avoids, e.g.,
# picking up experimental CSV files when scanning a directory for computations.
_INPUT_SUFFIXES = {
    "computation": {".log", ".out"},
    "experimental": {".csv"},
}


def find_input_files(args):
    """Resolve the list of input files from the CLI arguments.

    Files are filtered by ``--file-type`` so that the two cases never mix:
    only ``.log``/``.out`` are returned for inputs that are treated as
    computations, and only ``.csv`` for experimental spectra.
    """
    allowed = _INPUT_SUFFIXES[args.file_type]

    if args.file:
        path = Path(args.file)
        if not path.is_file() or path.suffix.lower() not in allowed:
            logger.error(
                "invalid input file '%s' for file type '%s' (expected %s)",
                args.file,
                args.file_type,
                ", ".join(sorted(allowed)),
            )
            return path.parent, []
        return path.parent, [path]

    root = Path(args.input_dir)
    if not root.is_dir():
        logger.error("input directory does not exist: %s", root)
        return root, []

    all_files = [f for f in root.iterdir() if f.is_file()]
    files = [f for f in all_files if f.suffix.lower() in allowed]

    skipped = len(all_files) - len(files)
    if skipped:
        logger.warning(
            "skipped %d file(s) in %s not matching file type '%s' (expected %s)",
            skipped,
            root,
            args.file_type,
            ", ".join(sorted(allowed)),
        )
    return root, files


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="UV-Vis Spectrum Plotter - Generate spectra and color coordinates from log files"
    )

    # --- File input ---------------------------------------------------------
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/color-prediction-neutral",
        help="Directory containing .log/.out/.csv files to process",
    )
    parser.add_argument(
        "--file", type=str, help="Single input file to process (overrides --input-dir)"
    )
    parser.add_argument(
        "--file-type",
        type=str,
        default="computation",
        choices=["computation", "experimental"],
        help="File type: 'computation' (.log/.out, TD-DFT) or 'experimental' (.csv)",
    )

    # --- Spectrum generation ------------------------------------------------
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.4,
        help="Gaussian broadening, in eV (default: 0.4)",
    )
    parser.add_argument(
        "--plot-range",
        type=int,
        nargs=2,
        default=[250, 800],
        metavar=("MIN", "MAX"),
        help="Wavelength range to display, in nm (default: 250 800)",
    )
    parser.add_argument(
        "--plot-grid",
        type=int,
        default=1,
        help="Grid step between generated points, in nm (default: 1)",
    )
    parser.add_argument(
        "--wv-correction",
        type=float,
        default=27,
        help="Wavelength correction shift in nm, applied only with --correct-wavelength (default: 27)",
    )

    # --- Experimental CSV parsing ------------------------------------------
    parser.add_argument(
        "--delimiter",
        type=str,
        default=";",
        help="Delimiter of experimental CSV files (default: ';')",
    )
    parser.add_argument(
        "--csv-header",
        action="store_true",
        help="Skip the header row of experimental CSV files (default: False)",
    )

    # --- Color calculation --------------------------------------------------
    parser.add_argument(
        "--concentration",
        type=float,
        default=0.000275,
        help="Sample concentration in mol/L (default: 0.000275)",
    )
    parser.add_argument(
        "--path-length",
        type=float,
        default=0.1,
        help="Path length in cm (default: 0.1)",
    )

    # --- Output options -----------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for written files (default: the inputs' directory)",
    )
    parser.add_argument(
        "--write-data",
        action="store_true",
        help="Write CSV / lambda_max data files (default: False)",
    )
    parser.add_argument(
        "--plot-data",
        action="store_true",
        help="Save PNG plots (default: False)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display plots in a window (default: False)",
    )
    parser.add_argument(
        "--generate-Lab",
        action="store_true",
        help="Compute and write xyz/RGB/Lab color coordinates (default: False)",
    )
    parser.add_argument(
        "--correct-wavelength",
        action="store_true",
        help="Apply the wavelength correction (--wv-correction) (default: False)",
    )
    parser.add_argument(
        "--normalize-data",
        action="store_true",
        help="Min-max normalize absorbances (default: False)",
    )

    return parser.parse_args()


def _validate_args(args):
    problems = []
    if args.sigma <= 0:
        problems.append("--sigma must be positive")
    if args.plot_range[0] >= args.plot_range[1]:
        problems.append("--plot-range MIN must be smaller than MAX")
    if args.plot_grid <= 0:
        problems.append("--plot-grid must be positive")
    if args.concentration < 0:
        problems.append("--concentration cannot be negative")
    if args.path_length < 0:
        problems.append("--path-length cannot be negative")
    if problems:
        for message in problems:
            logger.error(message)
        raise SystemExit(2)


def main():
    args = parse_arguments()
    _validate_args(args)

    files_root, files = find_input_files(args)
    if not files:
        logger.warning("no matching input files found")
        return

    data_is_from_computation = args.file_type == "computation"
    output_dir = Path(args.output_dir) if args.output_dir else files_root
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (args.write_data or args.plot_data or args.generate_Lab):
        logger.warning(
            "no output requested (use --write-data, --plot-data and/or --generate-Lab)"
        )

    processed, failed = 0, 0

    for file in files:
        try:
            if data_is_from_computation:
                data = extract_data_from_logs(
                    file.as_posix(), args.correct_wavelength, args.wv_correction
                )
                uv_spectrum = generate_spectrum(
                    data,
                    args.plot_range,
                    args.plot_grid,
                    args.sigma,
                    normalization=False,  # normalized once, below
                )
            else:
                uv_spectrum = extract_spectrum(
                    file.as_posix(),
                    delimiter=args.delimiter,
                    has_header=args.csv_header,
                )

            wavelengths = [point[0] for point in uv_spectrum]
            absorbances = [point[1] for point in uv_spectrum]

            # Normalize a single time, consistently for both data types.
            if args.normalize_data:
                wavelengths, absorbances = normalize_spectrum(wavelengths, absorbances)

            if not absorbances:
                logger.warning("%s: spectrum is empty, skipping", file.name)
                continue

            max_idx = int(np.argmax(absorbances))
            lambda_max = wavelengths[max_idx]

            base = output_dir / file.stem

            if args.write_data:
                _write_data(base, wavelengths, absorbances, lambda_max)

            if args.plot_data:
                _save_plot(base, wavelengths, absorbances, args, show=args.show)

            if args.generate_Lab:
                _write_color(base, wavelengths, absorbances, args)

            logger.info("%s: lambda_max = %.1f nm", file.name, lambda_max)
            processed += 1

        except Exception as exc:  # keep processing the rest of the batch
            logger.error("%s: failed (%s: %s)", file.name, type(exc).__name__, exc)
            failed += 1

    logger.info("done: %d processed, %d failed", processed, failed)


def _write_data(base, wavelengths, absorbances, lambda_max):
    csv_path = Path(str(base) + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(zip(wavelengths, absorbances))

    max_path = Path(str(base) + "_max.dat")
    with open(max_path, "w") as f:
        f.write(f"{base.name} lambda_max = {lambda_max:g}\n")


def _save_plot(base, wavelengths, absorbances, args, show=False):
    fig, ax = plt.subplots()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized absorbance" if args.normalize_data else "Absorbance")
    ax.set_xlim(left=args.plot_range[0], right=args.plot_range[1])
    ax.set_title(base.name)
    ax.plot(wavelengths, absorbances)

    img_path = Path(str(base) + ".png")
    fig.savefig(img_path, dpi=300)
    if show:
        fig.show()
    plt.close(fig)


def _write_color(base, wavelengths, absorbances, args):
    xyz = uvvis_to_xyz_color(
        wavelengths,
        absorbances,
        normalize=args.normalize_data,
        length=args.path_length,
        concentration=args.concentration,
    )
    rgb = xyz_to_RGB(xyz)
    lab = xyz_to_Lab(xyz)

    color_path = Path(str(base) + "_color.dat")
    with open(color_path, "w") as f:
        f.write(f"{base.name} xyz = " + ", ".join(f"{v:.4f}" for v in xyz) + "\n")
        f.write(f"{base.name} rgb = " + ", ".join(str(int(v)) for v in rgb) + "\n")
        f.write(f"{base.name} Lab = " + ", ".join(f"{v:.4f}" for v in lab) + "\n")


if __name__ == "__main__":
    main()
