import numpy as np
location_of_tip = 0.605  # the location of the middle of the tip, we want the entire tip to move together


def fem_wing_normal_sine(vec, freq_t, freq_s, amp, t):
    """
            take the wing and return the wing in normal sine wave in time t
           Args:
              vec: the wing's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
            Returns:
               vector representing the wing in time t after transform
            """
    transformed = np.copy(vec)
    transformed[2] += amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_normal_sine(vec, freq_t, freq_s, amp, t):
    """
            take the tip and return the tip in normal sine wave in time t
           Args:
              vec: the tip's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
            Returns:
               vector representing the wing in time t after transform
            """
    transformed = np.copy(vec)
    transformed[2] += amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_space(vec, freq_t, freq_s, amp, t, decay_rate_s):
    """
            take the wing and return the wing in decaying in space sine wave in time t
           Args:
              vec: the wing's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate_s: decay rate in space
            Returns:
               vector representing the wing in time t after transform
            """
    transformed = np.copy(vec)
    transformed[2] += np.exp(-(location_of_tip - vec[1]) * decay_rate_s) * amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_sine_decaying_in_space(vec, freq_t, freq_s, amp, t):
    """
            take the tip and return the tip in decaying sine wave in time t
           Args:
              vec: the tip's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time

            Returns:
               vector representing the wing in time t after transform
            """
    transformed = np.copy(vec)
    transformed[2] += amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_time(vec, freq_t, freq_s, amp, t, decay_rate_t):
    """
            take the wing and return the wing in decaying in time sine wave in time t
           Args:
              vec: the wing's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate_t: the decay rate in time
            Returns:
               vector representing the wing in time t after transform
            """
    transformed = np.copy(vec)
    transformed[2] += np.exp(-decay_rate_t * t) * amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_sine_decaying_in_time(vec, freq_t, freq_s, amp, t, decay_rate_t):
    """
            take the tip and return the tip in decaying in time sine wave in time t
           Args:
              vec: the tip's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate_t: the decay rate in time
            Returns:
               vector representing the wing in time t after transform
            """
    transformed = np.copy(vec)
    transformed[2] += np.exp(-decay_rate_t * t) * amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_space_time(vec, freq_t, freq_s, amp, t, decay_rate_s, decay_rate_t):
    """
            take the wing and return the wing in decaying in space and time sine wave in time t
           Args:
              vec: the wing's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate_s: the decay rate in space
              decay_rate_t: the decay rate in time
            Returns:
               vector representing the wing in time t after transform
            """
    transformed = vec
    transformed[2] += np.exp(-(location_of_tip - vec[1]) * decay_rate_s) * \
                      np.exp(-decay_rate_t * t) * amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_sine_decaying_in_space_time(vec, freq_t, freq_s, amp, t, decay_rate_t):
    """
            take the tip and return the tip in decaying in time and space sine wave in time t
           Args:
              vec: the tip's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate_t: the decay rate in time
            Returns:
               vector representing the wing in time t after transform
            """

    transformed = vec
    transformed[2] += np.exp(-decay_rate_t * t) * amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


