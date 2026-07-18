from math import pi, sqrt, atan2, sin, exp

# This function written in Python is not affiliated with the CIE (International Commission on Illumination),
# and is released into the public domain. It is provided "as is" without any warranty, express or implied.

# The classic CIE ΔE2000 implementation, which operates on two L*a*b* colors, and returns their difference.
# "l" ranges from 0 to 100, while "a" and "b" are unbounded and commonly clamped to the range of -128 to 127.
def ciede_2000(l_1, a_1, b_1, l_2, a_2, b_2):
    """
    Calculate CIEDE2000 color difference between two L*a*b* colors.

    Args:
        l_1, a_1, b_1: First color in L*a*b* space
        l_2, a_2, b_2: Second color in L*a*b* space

    Returns:
        The CIEDE2000 color difference (ΔE00)
    """
    # Parametric factors for different viewing conditions (textures, backgrounds, etc.)
    k_l = k_c = k_h = 1.0

    # Constants used in the formula
    CONST_25_7 = 25**7

    # Step 1: Calculate chroma components (C')
    # Average chroma of the two colors
    c_prime_1 = sqrt(a_1**2 + b_1**2)
    c_prime_2 = sqrt(a_2**2 + b_2**2)
    c_prime_avg = (c_prime_1 + c_prime_2) * 0.5

    # Chroma correction factor (G) - depends on average chroma
    c_prime_avg_7 = c_prime_avg**7
    g = 1.0 + 0.5 * (1.0 - sqrt(c_prime_avg_7 / (c_prime_avg_7 + CONST_25_7)))

    # Corrected chroma values
    c_1 = sqrt((a_1 * g)**2 + b_1**2)
    c_2 = sqrt((a_2 * g)**2 + b_2**2)

    # Step 2: Calculate hue angles (h')
    h_prime_1 = atan2(b_1, a_1 * g)
    h_prime_2 = atan2(b_2, a_2 * g)

    # Normalize hue angles to [0, 2π]
    h_prime_1 += 2.0 * pi * (h_prime_1 < 0.0)
    h_prime_2 += 2.0 * pi * (h_prime_2 < 0.0)

    # Step 3: Calculate hue difference components
    hue_diff = abs(h_prime_2 - h_prime_1)

    # Cross-implementation consistent rounding for π
    if pi - 1e-14 < hue_diff < pi + 1e-14:
        hue_diff = pi

    # Mean and half-difference of hue angles
    h_m = (h_prime_1 + h_prime_2) * 0.5
    h_d = (h_prime_2 - h_prime_1) * 0.5

    # Adjust for hue angles in different quadrants
    if pi < hue_diff:
        h_d += pi
        # 📜 Sharma’s formulation doesn’t use the next line, but the one after it,
        # and these two variants differ by ±0.0003 on the final color differences.
        h_m += pi
        # h_m += pi if h_m < pi else -pi

    # Step 4: Calculate hue rotation term (R_T)
    c_avg = (c_1 + c_2) * 0.5
    p = 36.0 * h_m - 55.0 * pi
    c_avg_7 = c_avg**7
    r_t = -2.0 * sqrt(c_avg_7 / (c_avg_7 + CONST_25_7)) * sin(pi / 3.0 * exp(p**2 / (-25.0 * pi**2)))

    # Step 5: Calculate lightness difference component
    l_avg = (l_1 + l_2) * 0.5
    l_diff_squared = (l_avg - 50.0)**2
    delta_l = (l_2 - l_1) / (k_l * (1.0 + 0.015 * l_diff_squared / sqrt(20.0 + l_diff_squared)))

    # Step 6: Calculate chroma difference component
    c_sum = c_1 + c_2
    delta_c = (c_2 - c_1) / (k_c * (1.0 + 0.0225 * c_sum))

    # Step 7: Calculate hue difference component
    # T factor for hue difference weighting
    t = (
        1.0
        + 0.24 * sin(2.0 * h_m + pi * 0.5)
        + 0.32 * sin(3.0 * h_m + 8.0 * pi / 15.0)
        - 0.17 * sin(h_m + pi / 3.0)
        - 0.20 * sin(4.0 * h_m + 3.0 * pi / 20.0)
    )
    delta_h = 2.0 * sqrt(c_1 * c_2) * sin(h_d) / (k_h * (1.0 + 0.0075 * c_sum * t))

    # Step 8: Calculate final color difference
    # Returning the square root ensures that dE00 accurately reflects the
    # geometric distance in color space, which can range from 0 to around 185.
    return sqrt(delta_l**2 + delta_h**2 + delta_c**2 + delta_c * delta_h * r_t)

# GitHub Project : https://github.com/michel-leonard/ciede2000-color-matching
#   Online Tests : https://michel-leonard.github.io/ciede2000-color-matching

# L1 = 56.7   a1 = 35.7   b1 = 1.8
# L2 = 56.4   a2 = 30.4   b2 = -1.5
# CIE ΔE00 = 2.9300617784 (Bruce Lindbloom, Netflix's VMAF, ...)
# CIE ΔE00 = 2.9300753540 (Gaurav Sharma, OpenJDK, ...)
# Deviation between implementations ≈ 1.4e-5

# See the source code comments for easy switching between these two widely used ΔE*00 implementation variants.