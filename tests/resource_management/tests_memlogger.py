import unittest, logging, io, sys, os
from unittest.mock import patch, MagicMock, call
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..','src')))
from HiTMicTools.resource_management.memlogger import MemoryLogger
MEMLOGGER_MODULE_PATH = 'HiTMicTools.resource_management.memlogger'

class TestMemoryLoggerConcise(unittest.TestCase):
    def setUp(self):
        self.log_stream = io.StringIO()
        self.original_logger_class = logging.getLoggerClass()
        logging.setLoggerClass(MemoryLogger)
        self.logger = logging.getLogger(f'TestMemLoggerConcise_{self.id()}')
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.handlers = [self.handler] # Replace any existing handlers

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        logging.setLoggerClass(self.original_logger_class)
        if self.logger.name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[self.logger.name]

    def test_log_basic(self): # Test basic message
        self.logger.info("Msg")
        self.assertIn("Msg", self.log_stream.getvalue())
        self.assertNotIn("| RAM:", self.log_stream.getvalue())
        self.assertNotIn("| VRAM:", self.log_stream.getvalue())

    @patch(f'{MEMLOGGER_MODULE_PATH}.get_memory_usage')
    def test_log_ram_only(self, mock_get_mem): # Test RAM info
        mock_get_mem.return_value = 1.5
        self.logger.info("M", show_memory=True)
        self.assertIn("M | RAM: 1.50 GB used", self.log_stream.getvalue())
        mock_get_mem.assert_called_once_with(unit="GB", as_string=False)

    @patch(f'{MEMLOGGER_MODULE_PATH}.GPUTIL_AVAILABLE', True)
    @patch(f'{MEMLOGGER_MODULE_PATH}.get_memory_usage')
    @patch(f'{MEMLOGGER_MODULE_PATH}.torch.device')
    def test_log_vram_gpu_available(self, mock_dev, mock_get_mem): # VRAM, GPU on
        mock_dev.return_value = MagicMock()
        mock_get_mem.side_effect = [2.5, 5.5] # VRAM used, VRAM free
        self.logger.info("M", cuda=True)
        self.assertIn("M | VRAM: 2.50 GB used / 5.50 GB free", self.log_stream.getvalue())
        mock_get_mem.assert_has_calls([call(device=mock_dev(), unit="GB", as_string=False),
                                       call(device=mock_dev(), free=True, unit="GB", as_string=False)])

    @patch(f'{MEMLOGGER_MODULE_PATH}.GPUTIL_AVAILABLE', False)
    @patch(f'{MEMLOGGER_MODULE_PATH}.get_memory_usage')
    def test_log_vram_gpu_not_available(self, mock_get_mem): # VRAM, GPU off
        self.logger.info("M", cuda=True)
        self.assertIn("M | VRAM: Not available", self.log_stream.getvalue())
        mock_get_mem.assert_not_called()

    @patch(f'{MEMLOGGER_MODULE_PATH}.GPUTIL_AVAILABLE', True)
    @patch(f'{MEMLOGGER_MODULE_PATH}.get_memory_usage')
    @patch(f'{MEMLOGGER_MODULE_PATH}.torch.device')
    def test_log_ram_and_vram(self, mock_dev, mock_get_mem): # RAM and VRAM
        mock_dev.return_value = MagicMock()
        mock_get_mem.side_effect = [1.5, 2.5, 5.5] # RAM, VRAM used, VRAM free
        self.logger.info("M", show_memory=True, cuda=True)
        self.assertIn("M | RAM: 1.50 GB used | VRAM: 2.50 GB used / 5.50 GB free", self.log_stream.getvalue())

    @patch(f'{MEMLOGGER_MODULE_PATH}.get_memory_usage')
    def test_log_ram_error(self, mock_get_mem): # RAM fetch error
        mock_get_mem.side_effect = Exception("RErr")
        self.logger.info("M", show_memory=True)
        self.assertIn("M | RAM: Error (RErr)", self.log_stream.getvalue())

if __name__ == '__main__':
    unittest.main()