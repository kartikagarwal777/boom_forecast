
from . import statsforecast_predictor
from . import data_utils
from . import lstm_predictor
from . import tft_predictor
import sys

sys.modules.setdefault("statsforecast_predictor", statsforecast_predictor)
sys.modules.setdefault("data_utils", data_utils)
sys.modules.setdefault("lstm_predictor", lstm_predictor)
sys.modules.setdefault("tft_predictor", tft_predictor)
