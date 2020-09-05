# Messages to server
# Successfully connected to ...
# Failed to connect to ...
# ... began recording
# ... finished recording, ... file created
#  ... started sending file ... to server
# file ... from ... saved as ...
# begin unifying? yes/no
# unified file saved as ...
import cmd
import logging
import os
import sys

import paho.mqtt.client as mqtt

logger = logging.Logger(__name__)


class Const:
    num_cam = 2
    num_imu = 0
    broker = "localhost"  # "server" ip
    port = 1883
    timeout = 300
    log_file = 'log.txt'
    data_folder = 'data/'
    raw_folder = data_folder + 'raw/'
    folders = [data_folder, raw_folder]
    device_name = "server"
    devices = ['cam' + str(i) for i in range(num_cam)] + \
              ['imu' + str(i) for i in range(num_imu)]
    broadcast = 'broadcast/instructions'
    sub_topics = [i + '/feedback' for i in devices] + [i + '/data' for i in devices]
    time_format = "%Y_%m_%d_%H_%M_%S_%f"  # <Year>_<Month>_<Day>_<Hour>_<Minute>_<Sec>_<MilliSec>
    delim = ' '


class Interface(cmd.Cmd):
    def do_ping(self, dev):
        pass

    def do_record(self, dev):
        #sets client.start_time
        pass

    def do_sample(self, dev):
        pass

    def do_shutdown(self, dev):
        pass


def init_logger():
    handler_file = logging.FileHandler(Const.log_file)
    handler_stdout = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stdout)
    logger.addHandler(handler_file)
    handler_file.setFormatter(logging.Formatter(Const.device_name + '|%(asctime)s|%(levelname)s|%(message)s'))
    handler_stdout.setFormatter(logging.Formatter(Const.device_name + '|%(asctime)s|%(levelname)s|%(message)s'))
    logger.setLevel(logging.INFO)
    logger.info("Starting...")


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
    message = msg.payload.decode("utf-8")# TODO remove this, odd behaviour for recieving data
    logger.info(f"Received message: {message} from {msg.topic}")
    reply = "Unknown command"
    if message[:4] == "PONG":
        pass
    elif message[:6] == "RECORD":
        pass # treats both data and feedback

    elif message == "SAMPLE":
        pass
    # TODO create sample image/imu data and send it

    elif message == "SHUTDOWN":
        pass # print shutdown successful
    else:
        logger.error(f"Unknown command: {message}")


def init_folder_struct():
    for folder in Const.folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def main():
    init_folder_struct()
    init_logger()
    client = mqtt.Client(Const.device_name)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(Const.broker, Const.port, Const.timeout)
    client.loop_start()  # -non-blocking
    Interface().cmdloop()


if __name__ == "__main__":
    main()
