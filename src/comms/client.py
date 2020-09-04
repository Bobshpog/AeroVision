# Status/ device working?
# Begin recording @time for x seconds
# Sync clocks
# Stop recording
# Shutdown
import logging
import os
import subprocess
from datetime import datetime
from time import sleep

import paho.mqtt.client as mqtt
import pause as pause

logger = logging.getLogger(__name__)


class Const:
    video_length = 20  # sec
    root_folder = os.path.expanduser('~/cam_script/')
    data_folder = root_folder + 'data/'
    folders = [root_folder, data_folder]
    device_name = 'cam0'
    time_format = "%Y_%m_%d_%H_%M_%S_%f"  # <Year>_<Month>_<Day>_<Hour>_<Minute>_<Sec>_<MilliSec>
    broker = "192.168.0.1"  # "server" ip
    port = 1883
    timeout = 300
    sub_topics = []  # TODO
    pub_topic = ""  # TODO


def take_video():
    # TODO fix function with actual cam api
    start_time = datetime.now()
    filename = Const.data_folder + Const.device_name + '::' + start_time.strftime(Const.time_format) + '.mp4'
    file = open(filename, "w")
    file.close()
    logger.info(filename + "created")
    return start_time, filename


def on_connect(client, userdata, flags, rc):
    logger.info("Connected with result code " + str(rc))
    for i in Const.sub_topics:
        client.subscribe(i)
        logger.info("Successfully subscribed to " + i)


def on_message(client, userdata, msg):
    message = str(msg.payload)
    logger.info(f"Received message: {message} from {msg.topic}")
    if message == "PING":
        ip = subprocess.check_output(["hostname", "-I"]).decode("utf-8")[:-2]
        reply = ip
    elif message[:6] == "RECORD":
        start_recording = float(message[7:])
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
        reply = "Unknown command"
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
    for dir in Const.folders:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if not os.path.exists(Const.root_folder):
        os.makedirs(Const.root_folder)


def main():
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
