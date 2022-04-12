import sys
import os
import logging
from datetime import datetime
class Logger(object):

    def __init__(self, input_path, type, terminal):
        self.terminal = terminal


        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m_%d_%H%M%S")
        self.path = f"Experiments_Final/{input_path}/{dt_string}_{type}/"
        filename = "Log.log"

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.path + filename):
            open(self.path + filename, 'w').close()
        LOG_FORMAT = "%(message)s"
        logging.basicConfig(filename = self.path + filename, level = logging.DEBUG, format = LOG_FORMAT)
        self.logger = logging.getLogger()
        #self.logger.path = self.path + filename
        for hdlr in self.logger.handlers[:]:  # remove all old handlers
            self.logger.removeHandler(hdlr)
        fileh = logging.FileHandler(self.path + filename, 'a')
        formatter = logging.Formatter(LOG_FORMAT)
        fileh.setFormatter(formatter)
        self.logger.addHandler(fileh)
        self.logger.handlers[0].terminator = ""

    def close(self):
        logging.shutdown()
    def write(self, message):
        self.terminal.write(message)
        #self.log.write(message)
        self.logger.info(message)
    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


