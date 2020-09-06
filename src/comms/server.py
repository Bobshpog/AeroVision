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
import threading
from datetime import datetime

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
    sub_topics = [i + '/feedback' for i in devices] + [i + '/data' for i in devices] + [i + '/sample' for i in devices]
    time_format = "%Y_%m_%d_%H_%M_%S_%f"  # <Year>_<Month>_<Day>_<Hour>_<Minute>_<Sec>_<MilliSec>
    delim = ' '


class Interface(cmd.Cmd):
    ip_map = {}  # currently unused
    samples = {}
    data = {}
    start_times = {}
    client = None
    messages_left = 0

    def do_ping(self, dev):
        success = False

        def ping(d):
            if d not in Const.devices:
                logger.info("Bad device name")
                return
            self.ip_map.pop(d)
            self.start_times.pop(d)
            pub_topic = d + "/instructions"
            self.client.publish(pub_topic, "PING")
            logger.debug("SENT|" + d + "|PING")

        if dev:
            ping(dev)
        else:
            self.ip_map.clear()
            self.start_times.clear()
            threads = []
            for d in Const.devices:
                threads.append(threading.Thread(target=ping, args=(d,)))
            self.messages_left = len(threads)
            for t in threads:
                t.start()
            if (threading.Condition().wait_for(lambda: self.messages_left == 0, 10)):
                logger.info("All Pings Successful")
            else:
                logger.info("Some Pings Failed")

    def do_record(self, dev):
        # sets client.start_time
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
    handler_file.setLevel(logging.DEBUG)
    logger.info("Starting...")


def on_connect(client, interface, flags, rc):
    """
    Runs when connection to broker is established
    """
    logger.info("Connected with result code " + str(rc))
    for i in Const.sub_topics:
        client.subscribe(i)
        logger.info("Successfully subscribed to " + i)


def on_message(client, interface, msg):
    """
    Runs when message is received, supported messages are:
    "PING": publishes a reply with device ip and system time
    "RECORD <time in seconds since epoch> <video length>":
        Begins a recording at provided time and publishes
        "RECORD <start_time>" to feedback channel
        and
        "<image>" to data channel
    "SAMPLE": Publishes a data sample
    "SHUTDOWN": Publishes "SHUTDOWN" and Shuts down gracefully
    """
    message = msg.payload
    msg_type = msg.topic.split("/")[-1]
    sender = msg.topic.split("/")[0]
    if msg_type == "feedback":
        message = msg.decode("utf-8").split(Const.delim)
        logger.debug(f"Received message: {message} from {msg.topic}")
        reply = "Unknown command"
        if message[0] == "PONG":
            interface.ip_map[sender] = message[1]
            interface.start_times[sender] = datetime.strptime(message[2], Const.time_format)
            logger.info(sender + "|Connected|" + message[1] + "|" + str(interface.start_times[sender]))
            interface.messages_left -= 1
        elif message[0] == "RECORD":
            pass  # treats both data and feedback

        elif message[0] == "SAMPLE":
            pass
        # TODO create sample image/imu data and send it

        elif message[0] == "SHUTDOWN":
            logger.info()
            pass  # print shutdown successful
        else:
            logger.error(f"Unknown command: {message}")
    if msg_type == "data":
        pass
    if msg_type == "sample":
        pass


def init_folder_struct():
    for folder in Const.folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def main():
    init_folder_struct()
    init_logger()
    interface = Interface()
    client = mqtt.Client(Const.device_name, userdata=interface)
    interface.client = client
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(Const.broker, Const.port, Const.timeout)
    client.loop_start()  # -non-blocking
    interface.cmdloop()


if __name__ == "__main__":
    main()
