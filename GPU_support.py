import humanize,psutil,GPUtil
import jax



def mem_report(num, gpu_idx):
            print(f"-{num}-CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
            
            GPUs = GPUtil.getGPUs()
            gpu = GPUs[gpu_idx]
            # for i, gpu in enumerate(GPUs):
            print('---GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%\n'.format(gpu_idx, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

def mem_report_0(gpu_idx):
            print(f"---CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
            
            GPUs = GPUtil.getGPUs()
            gpu = GPUs[gpu_idx]
            # for i, gpu in enumerate(GPUs):
            print('---GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%\n'.format(gpu_idx, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
