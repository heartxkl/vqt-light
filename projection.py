
import numpy as np

def new_projection(lambda_, phi):
    phi0 = 3 * np.pi / 8
    hprime = (
        12
        / np.pi
        * (
            np.arcsin(1 / np.sqrt(2 - np.cos(phi0) ** 2))
            + np.arcsin(2 * np.sin(phi0) / np.sqrt(3 - np.cos(2 * phi0)))
            - np.pi / 2
        )
    )
    xiprime = np.arctan(
        (np.pi * (hprime - 3) ** 2)
        / (
            np.sqrt(3)
            * (
                np.pi * (hprime ** 2 - 2 * hprime + 45)
                - 96 * np.arcsin(1 / np.sqrt(2 - np.cos(phi0) ** 2))
                - 48 * np.arcsin(2 * np.sin(phi0) / np.sqrt(3 - np.cos(2 * phi0)))
            )
        )
    )

    phi_c = np.abs(phi)
    lambda_c = lambda_ - np.pi / 4

    quadrant = np.floor(2 * lambda_ / np.pi)
    lambda0 = quadrant * np.pi / 2

    theta = np.abs(
        np.arctan2(
            np.cos(phi_c) * np.sin(lambda_c - lambda0),
            np.sin(phi0) * np.cos(phi_c) * np.cos(lambda_c - lambda0)
            - np.cos(phi0) * np.sin(phi_c),
        )
    )
    r = np.arccos(
        np.sin(phi0) * np.sin(phi_c)
        + np.cos(phi0) * np.cos(phi_c) * np.cos(lambda_c - lambda0)
    )

    psi0 = np.arcsin(1 / np.sqrt(2 - np.cos(phi0) ** 2))
    psi1 = np.pi - 2 * psi0

    beta = (psi0 - theta) * (theta <= psi0)
    beta += (theta - psi0) * (psi0 < theta) * (theta <= psi0 + psi1)
    beta += (np.pi - theta) * (psi0 + psi1 < theta)

    idx = theta <= psi0 + psi1
    c = np.arccos(np.cos(phi0) / np.sqrt(2)) * idx
    c += (np.pi / 2 - phi0) * ~idx

    idx = theta <= psi0
    G = psi0 * idx
    idx = (psi0 < theta) * (theta <= psi0 + psi1)
    G += psi1 * idx
    idx = theta > psi0 + psi1
    G += psi0 * idx

    psi0prime = np.arctan(np.sqrt(3) / hprime)
    psi1prime = 7 * np.pi / 6 - psi0prime - xiprime
    psi2prime = xiprime - np.pi / 6

    Gprime = psi0prime * (theta <= psi0)
    Gprime += psi1prime * (psi0 < theta) * (theta <= psi0 + psi1)
    Gprime += psi2prime * (theta > psi0 + psi1)

    F = np.arcsin(2 * np.sin(phi0) / np.sqrt(3 - np.cos(2 * phi0))) * (theta <= psi0)
    F += (
        (np.pi / 2 - np.arcsin(2 * np.sin(phi0) / np.sqrt(3 - np.cos(2 * phi0))))
        * (psi0 < theta)
        * (theta <= psi0 + psi1)
    )
    F += np.pi / 4 * (theta > psi0 + psi1)

    idx = theta <= psi0
    aprime = hprime * idx
    aprime += (
        np.sqrt(hprime ** 2 + 3)
        * np.sin(np.pi / 3 - np.arctan(hprime / np.sqrt(3)))
        / np.sin(xiprime)
        * ~idx
    )

    idx = theta <= psi0 + psi1
    cprime = np.sqrt(hprime ** 2 + 3) * idx
    cprime += (3 - hprime) * ~idx

    x = np.arccos(np.cos(r) * np.cos(c) + np.sin(r) * np.sin(c) * np.cos(beta))

    idx = x > 0
    gamma = (
        np.arcsin(np.sin(beta) * np.sin(r) / (np.sin(x) + ~idx)) * idx
    )  # Avoid divide by zero

    epsilon = np.arccos(
        np.sin(G) * np.sin(gamma) * np.cos(c) - np.cos(G) * np.cos(gamma)
    )

    upupvp = (gamma + G + epsilon - np.pi) / (F + G - np.pi / 2)

    cos_xy = np.sqrt(1 - (np.sin(G) * np.sin(c) / np.sin(epsilon)) ** 2)
    xpxpyp = np.sqrt((1 - np.cos(x)) / (1 - cos_xy))

    uprime = aprime * upupvp
    xpyp = np.sqrt(uprime ** 2 + cprime ** 2 - 2 * uprime * cprime * np.cos(Gprime))
    cos_gammaprime = np.sqrt(1 - (uprime * np.sin(Gprime) / xpyp) ** 2)
    xprime = xpyp * xpxpyp
    yprime = xpyp - xprime
    rprime = np.sqrt(xprime ** 2 + cprime ** 2 - 2 * xprime * cprime * cos_gammaprime)
    # Avoid divide by zero and clamp to < 1 to avoid issues due to rounding
    idx = uprime * rprime <= 0
    alphaprime = (xiprime - np.pi / 6) * idx
    idx = (
        (yprime ** 2 - uprime ** 2 - rprime ** 2) * np.reciprocal(-2 * uprime * rprime)
        < 1
    ) * (uprime * rprime > 0)
    alphaprime += (
        np.arccos(
            (yprime ** 2 - uprime ** 2 - rprime ** 2)
            / (-2 * uprime * rprime + ~idx)
            * idx
        )
        * idx
    )

    # Put theta back in the correct section
    idx = theta <= psi0
    thetaprime = alphaprime * idx
    idx = (psi0 < theta) * (theta <= psi0 + psi1)
    thetaprime += (np.pi - (xiprime - np.pi / 6) - alphaprime) * idx
    idx = psi0 + psi1 < theta
    thetaprime += (np.pi - (xiprime - np.pi / 6) + alphaprime) * idx

    x_c = (1 - 2 * (lambda_c - lambda0 < 0)) * rprime * np.sin(thetaprime)
    y_c = hprime - rprime * np.cos(thetaprime)
    y_h = y_c * np.sign(phi) - 3

    zeta = np.pi / 4 + quadrant * np.pi / 2
    x_m = (
        (x_c * np.cos(zeta) - y_h * np.sin(zeta) / np.sqrt(3))
        * np.sqrt(3)
        / (3 * np.sqrt(2))
    )
    y_m = (
        (x_c * np.sin(zeta) + y_h * np.cos(zeta) / np.sqrt(3))
        * np.sqrt(3)
        / (3 * np.sqrt(2))
    )
    return x_m, y_m


def new_projection_inverse(x_m, y_m):
    phi0 = 3 * np.pi / 8

    quadrant = 0 + (y_m >= 0) + (x_m <= 0) + 2 * (y_m <= 0) * (x_m < 0)

    zeta = np.pi / 4 + quadrant * np.pi / 2

    x_c = np.sqrt(6) * (x_m * np.cos(zeta) + y_m * np.sin(zeta))
    y_c = 3 * np.sqrt(2) * (y_m * np.cos(zeta) - x_m * np.sin(zeta))

    idx = y_c < -3
    sgnphi = -1 * idx + 1 * ~idx
    y_h = (-6 - y_c) * idx + y_c * ~idx

    hprime = (
        12
        / np.pi
        * (
            np.arcsin(1 / np.sqrt(2 - np.cos(phi0) ** 2))
            + np.arcsin(2 * np.sin(phi0) / np.sqrt(3 - np.cos(2 * phi0)))
            - np.pi / 2
        )
    )
    xiprime = np.arctan(
        (np.pi * (hprime - 3) ** 2)
        / (
            np.sqrt(3)
            * (
                np.pi * (hprime ** 2 - 2 * hprime + 45)
                - 96 * np.arcsin(1 / np.sqrt(2 - np.cos(phi0) ** 2))
                - 48 * np.arcsin(2 * np.sin(phi0) / np.sqrt(3 - np.cos(2 * phi0)))
            )
        )
    )

    rprime = np.sqrt(x_c ** 2 + (hprime - 3 - y_h) ** 2)
    thetaprime = np.abs(np.arctan2(x_c, hprime - 3 - y_h))

    psi0prime = np.arctan(np.sqrt(3) / hprime)
    psi1prime = 7 * np.pi / 6 - psi0prime - xiprime
    psi2prime = xiprime - np.pi / 6

    idx = thetaprime <= psi0prime
    alphaprime = thetaprime * idx
    idx = (psi0prime < thetaprime) * (thetaprime <= psi0prime + psi1prime)
    alphaprime += (np.pi - psi2prime - thetaprime) * idx
    idx = psi0prime + psi1prime < thetaprime
    alphaprime += (thetaprime + psi2prime - np.pi) * idx

    idx = thetaprime <= psi0prime + psi1prime
    c = np.arccos(np.cos(phi0) / np.sqrt(2)) * idx
    c += (np.pi / 2 - phi0) * ~idx

    psi0 = np.arcsin(1 / np.sqrt(2 - np.cos(phi0) ** 2))
    psi1 = np.pi - 2 * psi0
    idx = thetaprime <= psi0prime
    G = psi0 * idx
    idx = (psi0prime < thetaprime) * (thetaprime <= psi0prime + psi1prime)
    G += psi1 * idx
    idx = thetaprime > psi0prime + psi1prime
    G += psi0 * idx

    idx = thetaprime <= psi0prime
    Gprime = psi0prime * idx
    idx = (psi0prime < thetaprime) * (thetaprime <= psi0prime + psi1prime)
    Gprime += psi1prime * idx
    idx = thetaprime > psi0prime + psi1prime
    Gprime += psi2prime * idx

    idx = thetaprime <= psi0prime
    F = np.arcsin(2 * np.sin(phi0) / np.sqrt(3 - np.cos(2 * phi0))) * idx
    idx = (psi0prime < thetaprime) * (thetaprime <= psi0prime + psi1prime)
    F += (np.pi / 2 - np.arcsin(2 * np.sin(phi0) / np.sqrt(3 - np.cos(2 * phi0)))) * idx
    idx = thetaprime > psi0prime + psi1prime
    F += np.pi / 4 * idx

    idx = thetaprime <= psi0prime
    aprime = hprime * idx
    aprime += (
        np.sqrt(hprime ** 2 + 3)
        * np.sin(np.pi / 3 - np.arctan(hprime / np.sqrt(3)))
        / np.sin(xiprime)
        * ~idx
    )

    idx = thetaprime <= psi0prime + psi1prime
    cprime = np.sqrt(hprime ** 2 + 3) * idx
    cprime += (3 - hprime) * ~idx

    idx = thetaprime <= psi0prime
    b = np.pi / 4 * idx
    idx = (psi0prime < thetaprime) * (thetaprime <= psi0prime + psi1prime)
    b += np.arctan(np.sqrt(2) * np.tan(phi0)) * idx
    idx = thetaprime > psi0prime + psi1prime
    b += (np.pi / 2 - np.arctan(np.sqrt(2) * np.tan(phi0))) * idx

    betaprime = Gprime - alphaprime
    xprime = np.sqrt(
        rprime ** 2 + cprime ** 2 - 2 * rprime * cprime * np.cos(betaprime)
    )
    gammaprime = np.arcsin(rprime * np.sin(betaprime) / xprime) * (
        xprime > 0
    )  # Avoid divide by zero
    epsilonprime = np.pi - Gprime - gammaprime
    yprime = (
        rprime * np.sin(alphaprime) / np.sin(epsilonprime) * (epsilonprime > 0)
    )  # Avoid divide by zero
    uprime = np.sqrt(
        np.abs(
            cprime ** 2
            + (xprime + yprime) ** 2
            - 2 * cprime * (xprime + yprime) * np.cos(gammaprime)
        )
    )  # Take absolute value to avoid negative numbers due numerical precision issues
    vprime = aprime - uprime

    delta = np.arctan(
        (-np.sin(vprime * (F + G - np.pi / 2) / aprime))
        / (np.cos(b) - np.cos(vprime * (F + G - np.pi / 2) / aprime))
    )

    gamma = F - delta
    cos_xy = 1 / np.sqrt(1 + (np.tan(b) / np.cos(delta)) ** 2)

    x = np.arccos(1 - (xprime / (xprime + yprime)) ** 2 * (1 - cos_xy))

    r = np.arccos(np.cos(x) * np.cos(c) + np.sin(x) * np.sin(c) * np.cos(gamma))
    beta = np.arcsin(np.sin(x) * np.sin(gamma) / np.sin(r))

    idx = thetaprime <= psi0prime
    alpha = (psi0 - beta) * idx
    idx = (psi0prime < thetaprime) * (thetaprime <= psi0prime + psi1prime)
    alpha += (psi0 + beta) * idx
    idx = psi0prime + psi1prime < thetaprime
    alpha += (np.pi - beta) * idx

    phi_h = np.arcsin(
        np.sin(phi0) * np.cos(r) - np.cos(phi0) * np.sin(r) * np.cos(alpha)
    )
    lambda0 = np.pi / 4 + quadrant * np.pi / 2
    phi = sgnphi * phi_h
    lambda_ = lambda0 + np.sign(x_c) * np.arctan(
        np.sin(alpha)
        * np.sin(r)
        * np.cos(phi0)
        / (np.cos(r) - np.sin(phi0) * np.sin(phi_h))
    )
    lambda_ *= np.abs(phi) != np.pi / 2  # Handle edge case

    return lambda_, phi
