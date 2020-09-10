import numpy as np


def fem_wing_normal_sine(vec, freq_t, freq_s, amp, t):
    transformed = np.copy(vec)
    transformed[2] += amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_normal_sine(vec, freq_t, freq_s, amp, t):
    location_of_tip = 0.605
    transformed = np.copy(vec)
    transformed[2] += amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_space(vec, freq_t, freq_s, amp, t):
    transformed = np.copy(vec)
    transformed[2] += vec[1] * amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_sine_decaying_in_space(vec, freq_t, freq_s, amp, t):
    location_of_tip = 0.605
    transformed = np.copy(vec)
    transformed[2] += location_of_tip * amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_time(vec, freq_t, freq_s, amp, t, decay_rate):
    transformed = np.copy(vec)
    transformed[2] += np.exp(decay_rate * t) * amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_sine_decaying_in_time(vec, freq_t, freq_s, amp, t, decay_rate):
    location_of_tip = 0.605
    transformed = np.copy(vec)
    transformed[2] += np.exp(decay_rate * t) * amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


def fem_wing_sine_decaying_in_space_time(vec, freq_t, freq_s, amp, t, decay_rate):
    transformed = vec
    transformed[2] += np.exp(decay_rate * t) * vec[1] * amp * np.sin(t * freq_t + freq_s * vec[1])
    return transformed


def fem_tip_sine_decaying_in_space_time(vec, freq_t, freq_s, amp, t, decay_rate):
    location_of_tip = 0.605
    transformed = vec
    transformed[2] += np.exp(decay_rate * t) * location_of_tip * amp * np.sin(t * freq_t + freq_s * location_of_tip)
    return transformed


