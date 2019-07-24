from vocab import Vocabulary
import evaluation as eval
import os

# Allocate GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

eval.evalrank("./runs/f30k_scan/log/model_best.pth.tar", data_path="/home/ivy/scan/data", split="test")
