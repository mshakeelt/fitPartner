from threading import Thread
from queue import Queue

class Worker(Thread):
    def __init__(self, fn, queue_in, queue_out, *args, **kwargs):
        Thread.__init__(self)
        self.fn = fn
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.args = args
        self.kwargs = kwargs

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            input_file_path, total_files, useGPU = self.queue_in.get()
            try:
                results = self.fn(input_file_path, total_files, use_gpu=useGPU, return_results = True)
                self.queue_out.put(results)
            except:
                pass
            finally:
                self.queue_in.task_done()