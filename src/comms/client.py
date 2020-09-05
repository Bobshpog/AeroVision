import enum
import logging
import os
import subprocess
import sys
from datetime import datetime

import paho.mqtt.client as mqtt
import pause

logger = logging.getLogger(__name__)


class Devices(enum.Enum):
    cam = 0
    imu = 1


class Const:
    device_name = 'cam0' # 4 letters
    device_type = Devices.cam
    record_func_dict = {}  # initialized in main
    broker = "localhost"  # "server" ip
    port = 1883
    timeout = 300
    root_folder = os.path.expanduser('~/cam_script/')
    data_folder = root_folder + 'data/'
    folders = [root_folder, data_folder]
    time_format = "%Y_%m_%d_%H_%M_%S_%f"  # <Year>_<Month>_<Day>_<Hour>_<Minute>_<Sec>_<MilliSec>
    sub_topics = ["broadcast/instructions", device_name + "/instructions"]
    pub_messages = device_name + "/feedback"
    pub_data = device_name + "/data"
    delim = ' '


def record_camera(length):
    """
    This functions records a length long video and saves it

    Args:
        length: The length of the video in seconds
    Returns:
        start_time: datetime object of the record start
        filename: location of saved file
    """
    # TODO fix function with actual cam api
    # TODO make sure files are deleted at some point
    start_time = datetime.now()
    filename = Const.data_folder + Const.device_name + Const.delim + start_time.strftime(Const.time_format) + '.mp4'
    file = open(filename, "w")
    file.close()
    logger.info(filename + " created")
    return start_time, filename


def record_imu(length):
    """
       This functions records a length imu measurement and saves it

       Args:
           length: The length of the measurement in seconds
       Returns:
           start_time: datetime object of the record start
           filename: location of saved file
       """
    # TODO: write function
    return None, None


def on_connect(client, userdata, flags, rc):
    """
    Runs when connection to broker is established
    """
    logger.info("Connected with result code " + str(rc))
    for i in Const.sub_topics:
        client.subscribe(i)
        logger.info("Successfully subscribed to " + i)


def on_message(client, userdata, msg):
    """
    Runs when message is received, supported messages are:
    "PING": publishes a reply with device ip and system time
    "RECORD <time in seconds since epoch> <video length>":
        Begins a recording at provided time and publishes
        "RECORD <start_time>" to feedback channel
        and
        "RECORD <image>" to data channel
    "SAMPLE": Publishes a data sample
    "SHUTDOWN": Publishes "SHUTDOWN" and Shuts down gracefully
    """
    message = msg.payload.decode("utf-8")
    logger.info(f"Received message: {message} from {msg.topic}")
    reply = "Unknown command"
    if message == "PING":
        ip = subprocess.check_output(["hostname", "-I"]).decode("utf-8")[:-2]
        reply = "PONG" + Const.delim + ip + Const.delim + datetime.now().strftime(Const.time_format)

    elif message[:6] == "RECORD":
        record_fun = Const.record_func_dict[Const.device_type]
        start_recording, length = message[7:].split(' ')
        start_recording, length = float(start_recording), int(length)
        pause.until(start_recording)
        start_time, filename = record_fun(length)
        start_time = str(start_time)
        reply = f"RECORD" + Const.delim + start_time
        image_content=open(filename,"rb").read()
        client.publish(Const.pub_data,bytes(image_content)) # works only for files of size <256MB

    elif message == "SAMPLE":
        pass
    # TODO create sample image/imu data and send it

    elif message == "SHUTDOWN":
        client.publish(Const.pub_messages, "SHUTDOWN")
        client.disconnect()
    else:
        logger.error(f"Unknown command: {message}")
    client.publish(Const.pub_messages, reply)
    logger.info("Sent Message: \"" + reply + "\"")


def init_logger():
    handler_file = logging.FileHandler(Const.root_folder + Const.device_name + ".log")
    handler_stdout = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stdout)
    logger.addHandler(handler_file)
    handler_file.setFormatter(logging.Formatter(Const.device_name + '|%(asctime)s|%(levelname)s|%(message)s'))
    handler_stdout.setFormatter(logging.Formatter(Const.device_name + '|%(asctime)s|%(levelname)s|%(message)s'))
    logger.setLevel(logging.INFO)
    logger.info("Starting...")


def init_folder_struct():
    for folder in Const.folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def sync_clocks():
    pass  # TODO: maybe use external tools


def main():
    sync_clocks()
    init_folder_struct()
    init_logger()
    Const.record_func_dict = {Devices.cam: record_camera, Devices.imu: record_imu}
    client = mqtt.Client(Const.device_name)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(Const.broker, Const.port, Const.timeout)
    client.loop_forever()  # blocking


if __name__ == "__main__":
    main()
