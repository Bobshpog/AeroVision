import platform
from shutil import disk_usage

import cpuinfo
import torch
from psutil import virtual_memory


def identify_system():
    print(f"Computer Name: {platform.uname()[1]}")
    print(f"Python {platform.python_version()}, Pytorch {torch.__version__}, CuDNN {torch.backends.cudnn.version()}")
    cpu_info = cpuinfo.get_cpu_info()
    print(
        f"CPU: {cpu_info['brand_raw']} :: {cpu_info['count']} Cores ::"
        f" {virtual_memory().total / (1024. ** 3):.2f} GiB Memory ::"
        f"{disk_usage('/')[2] / 1024. ** 3:.2f} GiB Free Hard Disk Space")

    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU Devices:")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(
            f"GPU {i}: {props.name} [{props.multi_processor_count} SMPs , {props.total_memory / 1024. ** 3:.2f} GiB Memory]")

if __name__=='__main__':
    identify_system()