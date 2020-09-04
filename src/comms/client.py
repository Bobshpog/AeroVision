import logging
import os
import subprocess
from datetime import datetime
from time import sleep

import paho.mqtt.client as mqtt
import pause as pause

logger = logging.getLogger(__name__)


class Const:
    device_name = 'cam0'
    broker = "192.168.0.1"  # "server" ip
    root_folder = os.path.expanduser('~/cam_script/')
    data_folder = root_folder + 'data/'
    folders = [root_folder, data_folder]
    time_format = "%Y_%m_%d_%H_%M_%S_%f"  # <Year>_<Month>_<Day>_<Hour>_<Minute>_<Sec>_<MilliSec>
    port = 1883
    timeout = 300
    sub_topics = ["broadcast/instructions", device_name + "/instructions"]
    pub_topic = device_name + "/feedback"


def take_video(length):
    """
    This functions records a length long video and saves it

    Args:
        length: The length of the video in seconds
    Returns:
        start_time: datetime object of the record start
        filename: location of saved file
    """
    # TODO fix function with actual cam api
    start_time = datetime.now()
    filename = Const.data_folder + Const.device_name + '::' + start_time.strftime(Const.time_format) + '.mp4'
    file = open(filename, "w")
    file.close()
    logger.info(filename + "created")
    return start_time, filename


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
    "PING": publishes a reply with device ip
    "RECORD <time in seconds since epoch> <video length>":
        Begins a recording at provided time and publishes
        "Video started at <start_time>, Saved at <filename>"
    "SAMPLE": Publishes a data sample
    "SHUTDOWN": Publishes "Shutting Down" and Shuts down gracefully
    """
    message = str(msg.payload)
    logger.info(f"Received message: {message} from {msg.topic}")
    reply = "Unknown command"
    if message == "PING":
        ip = subprocess.check_output(["hostname", "-I"]).decode("utf-8")[:-2]
        reply = ip
    elif message[:6] == "RECORD":
        start_recording, length = message[7:].split(' ')
        start_recording, length = float(start_recording), int(length)
        pause.until(start_recording)
        start_time, filename = take_video()
        start_time = str(start_time)
        reply = f"Video started at {start_time}, Saved at {filename}"
        # TODO wait until message[7:], run take_video(), send message finished
    elif message == "SAMPLE":
        pass
    # TODO create sample image/imu data and send it
    elif message == "SHUTDOWN":
        client.publish(Const.pub_topic, "Shutting Down...")
        client.loop_stop()
        pass  # TODO:send ACK,close connections and end script
    else:
        logger.error(f"Unknown command: {message}")
    client.publish(Const.pub_topic, reply)
    logger.info("Sent Message" + reply)


def init_logger():
    handler = logging.FileHandler(Const.root_folder + Const.device_name + ".log")
    logger.addHandler(handler)
    handler.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(message)s'))
    logger.setLevel(logging.INFO)
    logger.info("Starting...")


def init_folder_struct():
    for folder in Const.folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def sync_clocks():
    pass  # TODO


def main():
    sync_clocks()
    init_folder_struct()
    init_logger()
    client = mqtt.Client(Const.device_name)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(Const.broker, Const.port, Const.timeout)
    client.loop_start()  # non-blocking
    while 1:
        sleep(1)


if __name__ == "__main__":
    main()
