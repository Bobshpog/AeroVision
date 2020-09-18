import numpy as np


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
    location_of_tip = 0.605  # the location of the middle of the tip, we want the entire tip to move together
    transformed = np.copy(vec)
    transformed[2] += amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_space(vec, freq_t, freq_s, amp, t):
    """
            take the wing and return the wing in decaying in space sine wave in time t
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
    transformed[2] += vec[1] * amp * np.sin(t * freq_t + freq_s * vec[1])
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
    location_of_tip = 0.605  # the location of the middle of the tip, we want the entire tip to move together
    transformed = np.copy(vec)
    transformed[2] += location_of_tip * amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_time(vec, freq_t, freq_s, amp, t, decay_rate):
    """
            take the wing and return the wing in decaying in time sine wave in time t
           Args:
              vec: the wing's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate: the decay rate in time
            Returns:
               vector representing the wing in time t after transform
            """
    transformed = np.copy(vec)
    transformed[2] += np.exp(decay_rate * t) * amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_sine_decaying_in_time(vec, freq_t, freq_s, amp, t, decay_rate):
    """
            take the tip and return the tip in decaying in time sine wave in time t
           Args:
              vec: the tip's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate: the decay rate in time
            Returns:
               vector representing the wing in time t after transform
            """
    location_of_tip = 0.605  # the location of the middle of the tip, we want the entire tip to move together
    transformed = np.copy(vec)
    transformed[2] += np.exp(decay_rate * t) * amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_space_time(vec, freq_t, freq_s, amp, t, decay_rate):
    """
            take the wing and return the wing in decaying in space and time sine wave in time t
           Args:
              vec: the wing's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate: the decay rate in time
            Returns:
               vector representing the wing in time t after transform
            """
    transformed = vec
    transformed[2] += np.exp(decay_rate * t) * vec[1] * amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_sine_decaying_in_space_time(vec, freq_t, freq_s, amp, t, decay_rate):
    """
            take the tip and return the tip in decaying in time and space sine wave in time t
           Args:
              vec: the tip's vertices
              freq_s: the frequency in space, inverse of wave length
              freq_t: the frequency in time
              amp: amplitude
              t: time
              decay_rate: the decay rate in time
            Returns:
               vector representing the wing in time t after transform
            """
    location_of_tip = 0.605     # the location of the middle of the tip, we want the entire tip to move together
    transformed = vec
    transformed[2] += np.exp(decay_rate * t) * location_of_tip * amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


