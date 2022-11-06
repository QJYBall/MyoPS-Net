import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import warnings
warnings.filterwarnings("ignore")

from utils.config import config
from train import CrossModalSegNetTrain
from pytorch_lightning import seed_everything


if __name__ == '__main__':
    seed_everything(233)
    args = config()    
    CrossModalSegNetTrain(args)
