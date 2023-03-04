import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{pwd}/voxceleb_trainer")
from .wrapper import prepare_model_clova