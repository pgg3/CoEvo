import logging

class BaseLogger(object):
    level_relations = {
        'DEBUG':logging.DEBUG,
        'INFO':logging.INFO,
        'WARNING':logging.WARNING,
        'ERROR':logging.ERROR,
        'CRITICAL':logging.CRITICAL
    }
    def __init__(self,log_file, cmd_log_level, file_log_level):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        file_fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        file_format_str = logging.Formatter(file_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(log_file, mode='w+', encoding='utf-8')
        file_handler.setLevel(self.level_relations.get(file_log_level))
        file_handler.setFormatter(file_format_str)
        self.logger.addHandler(file_handler)

        cmd_format = '%(asctime)s - %(levelname)s: %(message)s'
        cmd_format_str = logging.Formatter(cmd_format, datefmt="%Y-%m-%d %H:%M:%S")
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.level_relations.get(cmd_log_level))
        stream_handler.setFormatter(cmd_format_str)
        self.logger.addHandler(stream_handler)

if __name__ == '__main__':
    logger = Logger('all.log',cmd_log_level='INFO',file_log_level='DEBUG')
    logger.logger.info('DEBfdasfdsUG')
    logger.logger.error('hhh')
