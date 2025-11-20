import os
os.environ["CAMBRIAN_LAUNCHER"] = "TORCHXLA_MP"

# ! NOTE: debug only
# def _next_data(self):
#     index = self._next_index()  # may raise StopIteration
    
#     import accelerate
#     state = accelerate.state.AcceleratorState()
#     print(state.num_processes, state.process_index, self._index_sampler, index, flush=True)

#     data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
#     if self._pin_memory:
#         data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
#     return data

# import torch
# torch.utils.data.dataloader._SingleProcessDataLoaderIter._next_data = _next_data


from cambrian.train.train_fsdp import train

if __name__ == "__main__":

    import multiprocessing as mp
    import torch_xla.distributed.xla_multiprocessing as xmp
    mp.set_start_method('spawn', force=True)

    import os
    if os.getenv('CAMBRIAN_DEBUG', None) == '1':
        xmp.spawn(train, args=(None,), nprocs=1)
    else:
        xmp.spawn(train, args=(None,))
