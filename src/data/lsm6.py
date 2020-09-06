import enum

from smbus2 import SMBus


class RegAddr(enum.Enum):
    FUNC_CFG_ACCESS = 0x01
    FIFO_CTRL1 = 0x06
    FIFO_CTRL2 = 0x07
    FIFO_CTRL3 = 0x08
    FIFO_CTRL4 = 0x09
    FIFO_CTRL5 = 0x0A
    ORIENT_CFG_G = 0x0B
    INT1_CTRL = 0x0D
    INT2_CTRL = 0x0E
    WHO_AM_I = 0x0F
    CTRL1_XL = 0x10
    CTRL2_G = 0x11
    CTRL3_C = 0x12
    CTRL4_C = 0x13
    CTRL5_C = 0x14
    CTRL6_C = 0x15
    CTRL7_G = 0x16
    CTRL8_XL = 0x17
    CTRL9_XL = 0x18
    CTRL10_C = 0x19
    WAKE_UP_SRC = 0x1B
    TAP_SRC = 0x1C
    D6D_SRC = 0x1D
    STATUS_REG = 0x1E
    OUT_TEMP_L = 0x20
    OUT_TEMP_H = 0x21
    OUTX_L_G = 0x22
    OUTX_H_G = 0x23
    OUTY_L_G = 0x24
    OUTY_H_G = 0x25
    OUTZ_L_G = 0x26
    OUTZ_H_G = 0x27
    OUTX_L_XL = 0x28
    OUTX_H_XL = 0x29
    OUTY_L_XL = 0x2A
    OUTY_H_XL = 0x2B
    OUTZ_L_XL = 0x2C
    OUTZ_H_XL = 0x2D
    FIFO_STATUS1 = 0x3A
    FIFO_STATUS2 = 0x3B
    FIFO_STATUS3 = 0x3C
    FIFO_STATUS4 = 0x3D
    FIFO_DATA_OUT_L = 0x3E
    FIFO_DATA_OUT_H = 0x3F
    TIMESTAMP0_REG = 0x40
    TIMESTAMP1_REG = 0x41
    TIMESTAMP2_REG = 0x42
    STEP_TIMESTAMP_L = 0x49
    STEP_TIMESTAMP_H = 0x4A
    STEP_COUNTER_L = 0x4B
    STEP_COUNTER_H = 0x4C
    FUNC_SRC = 0x53
    TAP_CFG = 0x58
    TAP_THS_6D = 0x59
    INT_DUR2 = 0x5A
    WAKE_UP_THS = 0x5B
    WAKE_UP_DUR = 0x5C
    FREE_FALL = 0x5D
    MD1_CFG = 0x5E
    MD2_CFG = 0x5F


def _calc_value(data):
    data = data[1] << 8 | data[0]  # TODO: unclear if data[0] and [1] are strings or int
    return data - 65536 if data > 32768 else data


class LSM6:
    """
    Usage is:
            with LSM6() as imu:
    """
    DS33_SA0_HIGH_ADDRESS = 0b1101011
    DS33_SA0_LOW_ADDRESS = 0b1101010

    def __init__(self):
        self.bus = SMBus(1)
        self.address = self.DS33_SA0_HIGH_ADDRESS  # TODO: high address is assumed, SA0 connected to supply voltage
        # Enable Accelerometer high performance mode, ODR = 1000, (+-2g)
        self.bus.write_byte_data(self.address, RegAddr['CTRL1_XL'], 0x80)
        # Enable Gyro high performance mode, ODR = 1000, (245 dps)
        self.bus.write_byte_data(self.address, RegAddr['CTRL2_G'], 0x80)
        # Enable auto increment
        self.bus.write_byte_data(self.address, RegAddr['CTRL3_C'], 0x04)

    def read_values(self):
        """
        This function reads the IMUs measurements

        Returns:
            The sensors measurements in the following format
            (Gyro X, Gyro Y, Gyro Z, Acceleration X, Acc Y, Acc Z)
        """
        value_list = self.bus.read_i2c_block_data(self.address, RegAddr['OUTX_L_G'],
                                                  RegAddr['OUTZ_H_XL'] - RegAddr['OUTX_L_G'])
        value_list = [(value_list[i], value_list[i + 1]) for i in range(0, len(value_list), 2)]
        value_list = map(_calc_value, value_list)
        return tuple(value_list)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.bus.close()
        return True
