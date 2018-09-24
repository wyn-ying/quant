# coding:utf-8
import os
import logging
import time
import sys
from logging.handlers import TimedRotatingFileHandler

class LogTxt(object):
    '''
    classdocs
    '''


    def __init__(self, log_dir='default_log'):
        '''
        Constructor
        '''
        self.log_root_path = os.path.join(os.environ['HOME'], 'yingge_log', log_dir)
        if not os.path.exists(self.log_root_path):
            os.makedirs(self.log_root_path)
        
        self.log_file_prefix_above_info = os.path.join(self.log_root_path, 'above_info.log')
        self.log_file_prefix_debug_only = os.path.join(self.log_root_path, 'debug_only.log')
        
        self.logger_name = 'yingge'
       
       # define the logger with given name 'yidu' and debug level of 'DEBUG'
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        if not len(self.logger.handlers):
            # define one stream handler with debug level of 'INFO' for the logger
            streamHandler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)-8s %(asctime)s %(filename)s(%(lineno)d):%(funcName)s() %(message)s', datefmt='%Y%m%d-%H:%M:%S')
            streamHandler.setFormatter(formatter)
            streamHandler.setLevel(logging.INFO)
            self.logger.addHandler(streamHandler) 

            # define one TimedRotatingFileHandler with debug level of only 'DEBUG' for the logger
            timedRotatingFileHandler_debug_only = TimedRotatingFileHandler(self.log_file_prefix_debug_only, when='H', interval=24, backupCount=3)
            timedRotatingFileHandler_debug_only.suffix = "%Y_%m_%d-%H_%M_%S.log"
            timedRotatingFileHandler_debug_only.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)-8s %(asctime)s %(filename)s(%(lineno)d):%(funcName)s() %(message)s', datefmt='%Y%m%d-%H:%M:%S')
            timedRotatingFileHandler_debug_only.setFormatter(formatter)
            self.logger.addHandler(timedRotatingFileHandler_debug_only)
            
            # define one TimedRotatingFileHandler with debug level of 'INFO' for the logger
            timedRotatingFileHandler_above_info = TimedRotatingFileHandler(self.log_file_prefix_above_info, when='H', interval=24, backupCount=3)
            timedRotatingFileHandler_above_info.suffix = "%Y_%m_%d-%H_%M_%S.log"
            timedRotatingFileHandler_above_info.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)-8s %(asctime)s %(filename)s(%(lineno)d):%(funcName)s() %(message)s', datefmt='%Y%m%d-%H:%M:%S')
            timedRotatingFileHandler_above_info.setFormatter(formatter)
            self.logger.addHandler(timedRotatingFileHandler_above_info)
        self.logger.info('-'*20)
        
if __name__ == '__main__':
#     log_root_path = os.path.join(os.environ['HOME'], 'yidu_log_root')
#     if not os.path.exists(log_root_path):
#         os.makedirs(log_root_path)
    
    
    #logTxt = LogTxt()
    cur_logger = LogTxt().logger
    print(__file__)
    #cur_logger = logging.getLogger(logTxt.logger_name)
    for i in range(0, 1):
        cur_logger.debug('This is debug message')
        cur_logger.info('This is info message')
        cur_logger.warning('This is warning message')
        
#         
#     p = os.path.join(os.getcwd(), 'tst', 'test.txt')
#     print os.getcwd()
#     print os.environ['HOME']
#     print p
#     print os.path.basename(__file__)
    #print os.getcwd()
    #os.makedirs(p)
