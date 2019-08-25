from vocab import Vocabulary
import evaluation as eval
import os
import time

# Allocate GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


start_time = time.time()
eval.evalrank("runs/coco_scan/log/model_best.pth.tar", data_path="/home/ivy/scan/data", split="test", fold5=True)
end_time = time.time()
print("---%s seconds taken to process" %(end_time - start_time))
