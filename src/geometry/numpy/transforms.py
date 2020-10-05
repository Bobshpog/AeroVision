import numpy as np
location_of_tip = 0.605  # the location of the middle of the tip, we want the entire tip to move together
from src.geometry.numpy.mesh import cord2index

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


def synth_tip_movement(mesh_ver, tip_index, x, y, z, y_t, z_t, tip_table, new_tip_position, t):
    """
               creating the tip's  position in time t.
              Args:
                 mesh_ver: the mesh'es coordinates
                 tip_index: list of the indecies of the tip inside of mesh_ver
                 x: displacement in X, U1 in dinallea's format
                 y: displacement in Y, U2 in dinallea's format
                 z: displacement in Z, U3 in dinallea's format
                 y_t: the difference in y cord for our tip and the original tip, size (30) np..array
                 z_t: the difference in z cord for our tip and the original tip, size (30) np..array
                 tip_table: the tip's table
                 new_tip_position: the np.array we change, size (|tip.meshes|,3)
                 t: time
               Returns:
                  nothing, will fill new_tip_position
               """
    for idx in tip_index:
        for i in range(30):
            cord = mesh_ver[idx]
            vec = np.array((cord[0] + x[idx, t], cord[1] + y[idx, t] + y_t[i],
                            cord[2] + z[idx, t] + z_t[i]))
            new_tip_position[tip_table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec

def tip_arr_creation(mesh_ver, line=0.605):
    """
            creating list with the tip indices
           Args:
              mesh_ver: the mesh's vertices
              line: the threshold of where is the tip
            Returns:
               list [N] of indices of the tip
            """
    tip_index_arr = []
    for idx, cord in enumerate(mesh_ver):
        if cord[1] >= line:
            tip_index_arr.append(idx)
    return tip_index_arr
