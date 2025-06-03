import torch
import time

time.sleep(5)  
print(torch.cuda.is_available(), flush=True)
print(torch.cuda.device_count(), flush=True)
time.sleep(5)