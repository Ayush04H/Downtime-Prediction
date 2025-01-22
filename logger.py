import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_file=None):
        if log_file is None:
            log_file = f'ml_manufacturing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        log_path = os.path.join('logs', log_file)
        
        # Configure logging
        self.logger = logging.getLogger('ManufacturingML')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to the handlers
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def debug(self, message):
        self.logger.debug(message)