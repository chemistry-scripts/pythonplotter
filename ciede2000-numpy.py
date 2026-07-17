# This function written in Python is not affiliated with the CIE (International Commission on Illumination),
# and is released into the public domain. It is provided "as is" without any warranty, express or implied.

# The vecorized CIEDE2000 implementation, which operates on two L*a*b* colors, and returns their difference.
# "l" ranges from 0 to 100, while "a" and "b" are unbounded and commonly clamped to the range of -128 to 127.
def ciede2000(l1, a1, b1, l2, a2, b2, kl=1.0, kc=1.0, kh=1.0, canonical=False):
    import numpy as np

    # Working in Python with the CIEDE2000 color-difference formula.
    # kl, kc, kh are parametric factors to be adjusted according to
    # different viewing parameters such as textures, backgrounds...
    m_pi = np.asarray("3.141592653589793238462643383279502884", dtype=l2.dtype)
    pi_interoperability = m_pi + (1e-6 if m_pi.dtype == np.float32 else 1e-14)
    c1 = b1 * b1
    c2 = b2 * b2
    n = np.sqrt(a1 * a1 + c1)
    n += np.sqrt(a2 * a2 + c2)
    n *= 0.5
    n **= 7.0
    n /= n + 6103515625.0
    n = np.sqrt(n)
    n *= 0.5
    n = 1.5 - n  # The chroma correction factor.

    # atan2 is preferred over atan because it accurately computes the angle of
    # a point (x, y) in all quadrants, handling the signs of both coordinates.
    c = a1 * n
    c1 += c * c
    c1 = np.sqrt(c1)
    hm = np.arctan2(b1, c)
    hm[hm < 0.0] += 2.0 * m_pi

    c = a2 * n
    c2 += c * c
    c2 = np.sqrt(c2)
    h = np.arctan2(b2, c)
    h[h < 0.0] += 2.0 * m_pi

    # When the hue angles lie in different quadrants, the straightforward
    # average can produce a mean that incorrectly suggests a hue angle in
    # the wrong quadrant, the next 10 lines handle this issue.
    h -= hm
    n = pi_interoperability < np.abs(h)
    h *= 0.5  # h_delta
    hm += h  # h_mean

    # The part where most programmers get it wrong.
    if canonical:
        # When canonical is set to True, this acts like Gaurav Sharma, OpenJDK, ...
        hm[n] -= (pi_interoperability < hm[n]) * (2.0 * m_pi)
    # When canonical is set to False, this acts like Bruce Lindbloom, Netflix’s VMAF, ...
    hm[n] += m_pi
    h[n] += m_pi
    h = 2.0 * np.sin(h)
    h *= np.sqrt(c1 * c2)

    # Application of the chroma correction factor.
    c = c1 + c2
    n = 0.5 * c
    n **= 7.0
    n /= n + 6103515625.0

    # The hue rotation correction term is designed to account for the
    # non-linear behavior of hue differences in the blue region.
    r_t = -2.0 * np.sqrt(n)
    n = 36.0 * hm
    n -= 55.0 * m_pi
    n /= 5.0 * m_pi
    n *= n
    n = np.exp(-n)
    n *= m_pi / 3.0
    r_t *= np.sin(n)

    # These coefficients adjust the impact of different harmonic
    # components on the hue difference calculation.
    n = 150.0
    n -= 25.5 * np.sin(hm + m_pi / 3.0)
    n += 36.0 * np.sin(2.0 * hm + m_pi * 0.5)
    n += 48.0 * np.sin(3.0 * hm + 8.0 * m_pi / 15.0)
    n -= 30.0 * np.sin(4.0 * hm + 3.0 * m_pi / 20.0)
    hm = False  # Not used anymore.
    n /= 20000.0
    n *= c
    n += 1.0
    n *= kh
    # Hue.
    h /= n

    # Lightness.
    l = l2 - l1
    n = l1 + l2
    n *= 0.5
    n -= 50.0
    n *= n
    n /= np.sqrt(20.0 + n)
    n *= 3.0
    n /= 200.0
    n += 1.0
    n *= kl
    l /= n

    # Chroma.
    n = 9.0 * c
    n /= 400.0
    c = c2 - c1
    n += 1.0
    n *= kc
    c /= n
    c1 = c2 = n = False  # Not used anymore.

    cie00 = l * l
    cie00 += h * h
    h *= r_t
    h += c
    cie00 += c * h
    cie00 = np.sqrt(cie00)
    # The result accurately reflects the geometric distance in the color space.
    # The function allocates no more than 9 temporary vectors at any one time.
    ##############################################################################
    #       Goal         Data type    Speed gain      Accuracy     Correct digits
    ##############################################################################
    #      General        float32         x6           ± 2e-4            3
    #     Scientific      float64         x4          ± 4e-13            12
    #     Metrology       float96         x1          ± 2e-16            15
    return cie00


# If you remove the constant 1E-14, the code will continue to work, but CIEDE2000
# interoperability between all programming languages will no longer be guaranteed.

# Source code tested by Michel LEONARD
# Website: ciede2000.pages-perso.free.fr
