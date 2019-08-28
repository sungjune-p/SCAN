from vocab import Vocabulary
import evaluation as eval
import os
import time
import json

# Allocate GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def execute(input_string, n=1000):
    final_top_n = eval.evalrank(input_string, n, "runs/coco_scan/log/model_best.pth.tar", data_path="./data", split="test", fold5=False)
    
    ids = dict(result = None)
    ids['result'] = final_top_n
    jsonFile = json.dumps(ids)
    requests.post("http://localhost:5000/getScanResult", data={'Results': jsonFile})
    




# if __name__ == '__main__':
#     start_time = time.time()

    # local dir    
    # eval.evalrank("runs/coco_scan/log/model_best.pth.tar", data_path="/mnt/hard2/scan_data", split="test", fold5=False)
    # docker dir
#     eval.evalrank("runs/coco_scan/log/model_best.pth.tar", data_path="./data", split="test", fold5=False)
    
#    execute()

#     end_time = time.time()
#     print("---%s seconds taken to process" %(end_time - start_time))
    
