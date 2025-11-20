import os
import random
import functools
import numpy as np
from datetime import timedelta

_LOCAL_PROCESS_GROUP = None

# ! NOTE: setup logger
import logging
@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)

@functools.lru_cache(None)
def info_once(self, *args, **kwargs):
    """
    This method is identical to `logger.info()`, but will emit the info with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.info(*args, **kwargs)

logging.Logger.warning_once = warning_once
logging.Logger.info_once = info_once

logger = logging.getLogger(__name__)

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl
import torch.distributed as dist

# Import to register the `xla://` init_method
import torch_xla.distributed.xla_backend

# for checkpointing
import multiprocessing as mp
import torch.distributed.checkpoint as dist_cp
import torch_xla.experimental.distributed_checkpoint as xc

from io import BytesIO
import zstandard as zstd

################################################################################
# ! NOTE: please always keep accelerate's monkey patch at the top
# ! NOTE: the following code is monkey-patched to accelerate's source code
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

class MpDeviceLoaderWrapper(pl.MpDeviceLoader):
    def __init__(self, dataloader, device):
        logger.info("Calling monkey patch for MpDeviceLoaderWrapper...")
        input_sharding = {
            "input_ids": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "labels": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "attention_mask": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "images": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None, None, None), minibatch=True),
            "si_token_indices": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "miv_token_indices": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
        }

        super().__init__(
            dataloader,
            device=xm.xla_device(),
            input_sharding=input_sharding,
            loader_prefetch_size=1,
            device_prefetch_size=1,
        )
        self._rng_types = self._loader.rng_types
        self._loader.rng_types = None

    def __iter__(self):
        from accelerate.utils import synchronize_rng_states
        if self._rng_types is not None:
            synchronize_rng_states(self._rng_types, self._loader.synchronized_generator)

        return super().__iter__()

    @property
    def total_batch_size(self):
        return self._loader.total_batch_size

    @property
    def total_dataset_length(self):
        return self._loader.total_dataset_length

def skip_first_batches(dataloader, num_batches=0):
    """
    Creates a `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`.
    """
    logger.info(f"Calling monkey patch for skip_first_batches with num_batches={num_batches}...")
    #################################### ! CHANGES HERE ! ####################################
    import accelerate
    if isinstance(dataloader, MpDeviceLoaderWrapper):
        return MpDeviceLoaderWrapper(skip_first_batches(dataloader._loader, num_batches), None)
    #################################### ! CHANGES HERE ! ####################################

    dataset = dataloader.dataset
    sampler_is_batch_sampler = False
    if isinstance(dataset, IterableDataset):
        new_batch_sampler = None
    else:
        sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
        batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
        new_batch_sampler = accelerate.data_loader.SkipBatchSampler(batch_sampler, skip_batches=num_batches)

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    kwargs = {
        k: getattr(dataloader, k, accelerate.data_loader._PYTORCH_DATALOADER_KWARGS[k])
        for k in accelerate.data_loader._PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = dataloader.batch_size

    if isinstance(dataloader, accelerate.data_loader.DataLoaderDispatcher):
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            kwargs["skip_batches"] = num_batches
        dataloader = accelerate.data_loader.DataLoaderDispatcher(
            dataset,
            split_batches=dataloader.split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader._drop_last,
            **kwargs,
        )
    elif isinstance(dataloader, accelerate.data_loader.DataLoaderShard):
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            kwargs["skip_batches"] = num_batches
        elif sampler_is_batch_sampler:
            kwargs["sampler"] = new_batch_sampler
            kwargs["batch_size"] = dataloader.batch_size
        else:
            kwargs["batch_sampler"] = new_batch_sampler
        dataloader = accelerate.data_loader.DataLoaderShard(
            dataset,
            device=dataloader.device,
            rng_types=dataloader.rng_types,
            synchronized_generator=dataloader.synchronized_generator,
            **kwargs,
        )
    else:
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            dataloader = accelerate.data_loader.SkipDataLoader(dataset, skip_batches=num_batches, **kwargs)
        else:
            dataloader = DataLoader(dataset, batch_sampler=new_batch_sampler, **kwargs)

    return dataloader

def prepare_data_loader(
    dataloader=None,
    device=None,
    num_processes=None,
    process_index=None,
    split_batches=False,
    put_on_device=False,
    rng_types=None,
    dispatch_batches=None,
    even_batches=True,
    slice_fn_for_dispatch=None,
):
    logger.info("Calling monkey patch for prepare_data_loader...")
    import accelerate
    import torch

    if dispatch_batches is None:
        if not put_on_device:
            dispatch_batches = False
        else:
            dispatch_batches = isinstance(dataloader.dataset, IterableDataset)

    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`.")
    # Grab defaults from AcceleratorState
    state = accelerate.state.AcceleratorState()
    if num_processes is None:
        num_processes = state.num_processes
    if process_index is None:
        process_index = state.process_index

    #################################### ! CHANGES HERE ! ####################################
    num_processes = torch.distributed.get_world_size()
    process_index = torch.distributed.get_rank()
    #################################### ! CHANGES HERE ! ####################################

    # Sanity check
    if split_batches and dataloader.batch_size > 1 and dataloader.batch_size % num_processes != 0:
        raise ValueError(
            f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
            f"needs to be a round multiple of the number of processes ({num_processes})."
        )

    new_dataset = dataloader.dataset
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    sampler_is_batch_sampler = False
    synchronized_generator = None
    # No change if no multiprocess

    if (num_processes != 1 or state.distributed_type == accelerate.state.DistributedType.MEGATRON_LM) and not dispatch_batches:
        if isinstance(new_dataset, IterableDataset):
            if getattr(dataloader.dataset, "generator", None) is not None:
                synchronized_generator = dataloader.dataset.generator
            new_dataset = accelerate.data_loader.IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
        else:
            # New batch sampler for the current process.
            sampler_is_batch_sampler = isinstance(dataloader.sampler, torch.utils.data.BatchSampler)
            if sampler_is_batch_sampler:
                sampler = dataloader.sampler.sampler
            else:
                sampler = dataloader.batch_sampler.sampler
            if hasattr(sampler, "generator"):
                if sampler.generator is None:
                    sampler.generator = torch.Generator()
                synchronized_generator = sampler.generator

            batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
            new_batch_sampler = accelerate.data_loader.BatchSamplerShard(
                batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
                even_batches=even_batches,
            )

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    if rng_types is not None and synchronized_generator is None and "generator" in rng_types:
        rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, accelerate.data_loader._PYTORCH_DATALOADER_KWARGS[k])
        for k in accelerate.data_loader._PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = (
            dataloader.batch_size // num_processes if split_batches and not dispatch_batches else dataloader.batch_size
        )

    if dispatch_batches:
        kwargs.pop("generator")
        dataloader = accelerate.data_loader.DataLoaderDispatcher(
            new_dataset,
            split_batches=split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader.drop_last,
            slice_fn=slice_fn_for_dispatch,
            **kwargs,
        )
    elif sampler_is_batch_sampler:
        dataloader = accelerate.data_loader.DataLoaderShard(
            new_dataset,
            device=device if put_on_device and state.distributed_type != accelerate.state.DistributedType.TPU else None,
            sampler=new_batch_sampler,
            batch_size=dataloader.batch_size,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            **kwargs,
        )
    else:
        dataloader = accelerate.data_loader.DataLoaderShard(
            new_dataset,
            device=device if put_on_device and state.distributed_type != accelerate.state.DistributedType.TPU else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            **kwargs,
        )

    if state.distributed_type == accelerate.state.DistributedType.TPU:
        return MpDeviceLoaderWrapper(dataloader, device)
    return dataloader

def accelerator_prepare_data_loader(
    self, data_loader, device_placement=None, slice_fn_for_dispatch=None
):
    logger.info("Calling monkey patch for Accelerator.prepare_data_loader...")
    import accelerate
    # Ensure we can't double wrap a DataLoader due to `find_batch_size`
    if getattr(data_loader, "_is_accelerate_prepared", False):
        if data_loader not in self._dataloaders:
            self._dataloaders.append(data_loader)
        return data_loader
    if device_placement is None:
        device_placement = self.device_placement if self.distributed_type != accelerate.state.DistributedType.TPU else False

    #################################### ! CHANGES HERE ! ####################################
    prepared_data_loader = prepare_data_loader(
        data_loader,
        self.device,
        num_processes=self.num_processes,
        process_index=self.process_index,
        split_batches=self.split_batches,
        put_on_device=device_placement,
        rng_types=self.rng_types.copy(),
        dispatch_batches=self.dispatch_batches,
        even_batches=self.even_batches,
        slice_fn_for_dispatch=slice_fn_for_dispatch,
    )
    #################################### ! CHANGES HERE ! ####################################

    self._dataloaders.append(prepared_data_loader)
    return prepared_data_loader

import accelerate
import accelerate.data_loader
accelerate.skip_first_batches = skip_first_batches
accelerate.data_loader.skip_first_batches = skip_first_batches
accelerate.data_loader.MpDeviceLoaderWrapper = MpDeviceLoaderWrapper
accelerate.accelerator.Accelerator.prepare_data_loader = accelerator_prepare_data_loader
# ! NOTE: the above code is monkey-patched to accelerate's source code
################################################################################

################################################################################
# ! NOTE: the following code is monkey-patched to transformers' source code
import transformers
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    logger.info_once("Calling monkey patch for _get_cosine_schedule_with_warmup_lr_lambda...")
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

    min_lr_ratio = float(os.getenv("CAMBRIAN_MIN_LR_RATIO", "0."))

    scale_ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    if min_lr_ratio > 0.0:
        logger.info_once(f"Using min_lr_ratio {min_lr_ratio} to scale the learning rate...")
        scale_ratio = min_lr_ratio + (1.0 - min_lr_ratio) * scale_ratio
    return scale_ratio

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    logger.info("Calling monkey patch for get_cosine_schedule_with_warmup...")
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

transformers.optimization.TYPE_TO_SCHEDULER_FUNCTION = {transformers.trainer_utils.SchedulerType.COSINE: get_cosine_schedule_with_warmup}

from transformers import Trainer
import time
import torch
import functools
from torch import nn
from packaging import version

from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import (
    get_module_class_from_name,
)
from transformers.training_args import ParallelMode
from transformers.utils import (
    logging,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_neuroncore_available,
)
from accelerate.utils import DistributedDataParallelKwargs

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

def _wrap_model(self, model, training=True, dataloader=None):
    logger.info("Calling monkey patch for Trainer._wrap_model...")
    self.is_fsdp_xla_v2_enabled = True

    if self.args.use_ipex:
        dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
        model = self.ipex_optimize_model(model, training, dtype=dtype)

    if is_sagemaker_mp_enabled():
        # Wrapping the base model twice in a DistributedModel will raise an error.
        if isinstance(self.model_wrapped, smp.model.DistributedModel):
            return self.model_wrapped
        return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

    # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
    if self.accelerator.unwrap_model(model) is not model:
        return model

    # Mixed precision training with apex (torch < 1.6)
    if self.use_apex and training:
        model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
    if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
        model = nn.DataParallel(model)

    if self.args.jit_mode_eval:
        start_time = time.time()
        model = self.torch_jit_model_eval(model, dataloader, training)
        self.jit_compilation_time = round(time.time() - start_time, 4)

    # Note: in torch.distributed mode, there's no point in wrapping the model
    # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
    if not training:
        return model

    # Distributed training (should be after apex fp16 initialization)
    # Distributed training using PyTorch FSDP
    if self.is_fsdp_xla_enabled:
        try:
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
            from torch_xla.distributed.fsdp import checkpoint_module
            from torch_xla.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
            )

            if self.is_fsdp_xla_v2_enabled:
                from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
                    SpmdFullyShardedDataParallel as FSDPv2,
                )
        except ImportError:
            raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
        auto_wrap_policy = None
        auto_wrapper_callable = None
        default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = self.args.fsdp_config.get(
            "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
        )

        if self.args.fsdp_config["min_num_params"] > 0:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["min_num_params"]
            )
        elif fsdp_transformer_layer_cls_to_wrap is not None:
            transformer_cls_to_wrap = set()
            for layer_class in fsdp_transformer_layer_cls_to_wrap:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise Exception("Could not find the transformer layer class to wrap in the model.")
                else:
                    transformer_cls_to_wrap.add(transformer_cls)

            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                # Transformer layer class to wrap
                transformer_layer_cls=transformer_cls_to_wrap,
            )
        fsdp_kwargs = self.args.xla_fsdp_config
        if self.args.fsdp_config["xla_fsdp_grad_ckpt"]:
            if model.config.use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                model.config.use_cache = False

            # Apply gradient checkpointing to auto-wrapped sub-modules if specified
            def auto_wrapper_callable(m, *args, **kwargs):
                target_cls = FSDP if not self.is_fsdp_xla_v2_enabled else FSDPv2
                return target_cls(checkpoint_module(m), *args, **kwargs)

        #################################### ! CHANGES HERE ! ####################################
        # Wrap the base model with an outer FSDP wrapper
        if self.is_fsdp_xla_v2_enabled:

            def shard_output(output, mesh):
                from transformers.modeling_outputs import CausalLMOutputWithPast

                real_output = None
                if isinstance(output, torch.Tensor):
                    real_output = output
                elif isinstance(output, tuple):
                    real_output = output[0]
                elif isinstance(output, CausalLMOutputWithPast):
                    real_output = output.logits

                if real_output is None:
                    raise ValueError("Something went wrong, the output of the model shouldn't be `None`")
                xs.mark_sharding(real_output, mesh, ("fsdp", None, None))

            self.model = model = FSDPv2(
                model,
                shard_output=shard_output,
                auto_wrap_policy=auto_wrap_policy,
                auto_wrapper_callable=auto_wrapper_callable,
            )

            from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
            # Use a patch to `nn.Linear` (`torch.nn.functional.linear`) in XLA so that its
            # backward pass will use its weight parameter rather than an intermediate result.
            # (see https://github.com/pytorch/xla/issues/3811 for details)
            self.model = apply_xla_patch_to_nn_linear(self.model, xs.xla_patched_nn_linear_forward)
            #################################### ! CHANGES HERE ! ####################################
            # from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear
            # from cambrian.train.shard_model import shard_torch_xla_model_from_config, wrap_module

            # def maybe_checkpoint(mod, _name):
            #     if isinstance(mod, tuple(transformer_cls_to_wrap)):
            #         logger.info(f"Applying gradient checkpointing to {_name}...")
            #         return checkpoint_module(mod)
            #     return mod

            # def maybe_add_barrier(mod, _name):
            #     if isinstance(mod, tuple(transformer_cls_to_wrap)):
            #         # Register a backward hook to place optimization barrier to prevent
            #         # gigantic fusions on syncing the gradients.
            #         logger.info(f"Adding optimization barrier to {_name}...")
            #         xs.apply_backward_optimization_barrier(mod)
            #         return mod
            #     return mod

            # model = model.to("xla")
            # model = apply_xla_patch_to_nn_linear(model)
            
            # for name, param in model.named_parameters():
            #     print(name, param.shape)

            # sharding_config = {
            #     # language model
            #     "model.embed_tokens.weight": ["fsdp", None],
            #     "model.layers.*.self_attn.q_proj.weight": ["fsdp", None],
            #     "model.layers.*.self_attn.q_proj.bias": ["fsdp",],
            #     "model.layers.*.self_attn.k_proj.weight": [None, "fsdp"],
            #     "model.layers.*.self_attn.k_proj.bias": ["fsdp",],
            #     "model.layers.*.self_attn.v_proj.weight": [None, "fsdp"],
            #     "model.layers.*.self_attn.v_proj.bias": ["fsdp",],
            #     "model.layers.*.self_attn.o_proj.weight": ["fsdp", None],
            #     "model.layers.*.mlp.gate_proj.weight": ["fsdp", None],
            #     "model.layers.*.mlp.up_proj.weight": ["fsdp", None],
            #     "model.layers.*.mlp.down_proj.weight": [None, "fsdp"],
            #     "model.layers.*.input_layernorm.weight": ["fsdp",],
            #     "model.layers.*.post_attention_layernorm.weight": ["fsdp",],
            #     "model.norm.weight": ["fsdp",],
            #     "lm_head.weight": ["fsdp", None],

            #     # mm_projector
            #     "model.mm_projector.*.weight": ["fsdp", None],
            #     "model.mm_projector.*.bias": ["fsdp",],
                
            #     # activations
            #     "model.layers.*[0]": ["fsdp", None, None],
            #     "lm_head": ["fsdp", None, None],
            # }

            # model = shard_torch_xla_model_from_config(model, config=sharding_config)

            # model = wrap_module(model, maybe_checkpoint)
            # model = wrap_module(model, maybe_add_barrier)
            # self.model = model
            #################################### ! CHANGES HERE ! ####################################
        else:
            self.model = model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                auto_wrapper_callable=auto_wrapper_callable,
                **fsdp_kwargs,
            )

        # Patch `xm.optimizer_step` should not reduce gradients in this case,
        # as FSDP does not need gradient reduction over sharded parameters.
        def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
            loss = optimizer.step(**optimizer_args)
            if barrier:
                xm.mark_step()
            return loss

        xm.optimizer_step = patched_optimizer_step
    elif is_sagemaker_dp_enabled():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
        )
    elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
        if is_torch_neuroncore_available():
            return model
        kwargs = {}
        if self.args.ddp_find_unused_parameters is not None:
            kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
        elif isinstance(model, PreTrainedModel):
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
            kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
        else:
            kwargs["find_unused_parameters"] = True

        if self.args.ddp_bucket_cap_mb is not None:
            kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

        if self.args.ddp_broadcast_buffers is not None:
            kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

        self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)

    # model.to(torch.bfloat16) # ! NOTE: this helps to reduce memory usage, but the impact on performance remains to be seen
    from cambrian.utils import inspect_tensor_sharding
    print(model, flush=True)
    for name, param in model.named_parameters():
        print(name, inspect_tensor_sharding(param), flush=True)
    return model

Trainer._wrap_model = _wrap_model
# ! NOTE: the above code is monkey-patched to transformers' source code
################################################################################

################################################################################
# ! NOTE: the following code is monkey-patched to transformers' qwen2 implementation
from torch_xla.experimental.custom_kernel import flash_attention
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache
from cambrian.utils import IS_XLA_AVAILABLE
import warnings
from typing import Optional, Tuple
import math

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    logger.info_once("Calling monkey patch for Qwen2Attention.forward...")
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if not IS_XLA_AVAILABLE:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

    else:
        attn_output = flash_attention(
            query_states, key_states, value_states, causal=True,
            sm_scale=1. / math.sqrt(self.head_dim),
            partition_spec=('fsdp', None, None, None))

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

Qwen2Attention.forward = forward
# ! NOTE: the above code is monkey-patched to transformers' qwen2 implementation
################################################################################

################################################################################
# ! NOTE: the following code is monkey-patched to siglip's implementation
from cambrian.model.multimodal_encoder.llava_next_siglip_encoder import SigLipAttention, SigLipVisionTransformer
def _siglip_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    logger.info_once("Calling monkey patch for SigLipAttention.forward...")

    batch_size, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if not IS_XLA_AVAILABLE:
        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")
    else:
        attn_output = flash_attention(
            query_states, key_states, value_states, causal=False,
            ab=attention_mask,
            sm_scale=self.scale,
            partition_spec=('fsdp', None, None, None))
        attn_weights = None

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights

def _siglip_vit_forward(
    self,
    pixel_values,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    logger.info_once("Calling monkey patch for SigLipVisionTransformer.forward...")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    hidden_states = self.embeddings(pixel_values)

    #################################### ! CHANGES HERE ! ####################################
    hidden_states_padded = False
    hidden_states_seqlen = None
    attention_mask = None
    if hidden_states.size(1) % 512 != 0:
        hidden_states_padded = True
        hidden_states_seqlen = hidden_states.size(1)
        hidden_states = torch.cat([hidden_states, torch.zeros(hidden_states.size(0), 512 - hidden_states.size(1) % 512, hidden_states.size(2)).to(hidden_states.device)], dim=1)

        if not hasattr(self, "cached_attention_mask"):
            attention_mask = torch.zeros(hidden_states.size(0), self.config.num_attention_heads, hidden_states.size(1), hidden_states.size(1)).to(hidden_states.dtype)
            attention_mask[..., hidden_states_seqlen:] = torch.finfo(attention_mask.dtype).min
            attention_mask = attention_mask.to(hidden_states.device)
            self.cached_attention_mask = attention_mask
        else:
            attention_mask = self.cached_attention_mask
    #################################### ! CHANGES HERE ! ####################################

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask, # ! NOTE: CHANGE HERE !
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    #################################### ! CHANGES HERE ! ####################################
    if hidden_states_padded:
        # Change tuple to list to allow modification
        encoder_outputs.hidden_states = list(encoder_outputs.hidden_states)
        # Trunc the last hidden state to the original length
        encoder_outputs.hidden_states[-1] = encoder_outputs.hidden_states[-1][:, :hidden_states_seqlen, :].clone()
    #################################### ! CHANGES HERE ! ####################################

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.post_layernorm(last_hidden_state)

    pooled_output = self.head(last_hidden_state)

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    from transformers.modeling_outputs import BaseModelOutputWithPooling
    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

# SigLipAttention.forward = _siglip_attention_forward
# SigLipVisionTransformer.forward = _siglip_vit_forward
# ! NOTE: not use because it does not help to reduce memory usage or speed up
# ! NOTE: the above code is monkey-patched to siglip's implementation
# ! NOTE: siglip's seq len is 729, not divisible by 512 so we cannot use torchxla's flash attention here
################################################################################

from fsspec.core import url_to_fs
from torch_xla.experimental.distributed_checkpoint import CheckpointManager

# def _load_tracked_chkpts(self):
#     logger.info("Calling monkey patch for CheckpointManager._load_tracked_chkpts...")
#     return []
# CheckpointManager._load_tracked_chkpts = _load_tracked_chkpts

################################################################################
# ! NOTE: the following code is monkey-patched to Cambrian Trainer
from cambrian.train.cambrian_trainer import CambrianTrainer
from transformers import Trainer
import dataclasses
import json

def __init__(self, *args, **kwargs):
    logger.info("Calling spmd monkey patch for CambrianTrainer.__init__...")
    super(CambrianTrainer, self).__init__(*args, **kwargs)
    logger.info("Finish super init for CambrianTrainer.__init__...")

    self.consolidate_counter = 0
    logger.info(f"Create checkpoint manager with output_dir {self.args.output_dir.replace('/mnt/', 'gs://')}...")

    # dist.barrier()
    global _LOCAL_PROCESS_GROUP
    if dist.get_rank() == 0:
        logger.info("Create checkpoint folder if not exist on rank 0...")
        os.makedirs(self.args.output_dir.replace("gs://", "/mnt/"), exist_ok=True)
    # dist.barrier()
    self.checkpoint_manager = CheckpointManager(
        # path=self.args.output_dir.replace("gs://", "/mnt/"), # ! NOTE: hard code to gcsfuse
        path=self.args.output_dir.replace("/mnt/", "gs://"), # ! NOTE: hard code to gs (0622 for testing)
        save_interval=self.args.save_steps,
        max_to_keep=5, # ! NOTE: hard code to 10 but need to be configurable
        process_group=_LOCAL_PROCESS_GROUP,
    )
    # dist.barrier()
    logger.info("Finish create checkpoint manager...")
    logger.info(f"Create fs with output_dir {self.args.output_dir.replace('gs://', '/mnt/')}")
    self.fs, _ = url_to_fs(self.args.output_dir.replace("gs://", "/mnt/"))
    # dist.barrier()
    logger.info("Finish create fs...")

CambrianTrainer.__init__ = __init__

def get_train_dataloader(self):
    logger.info("Calling spmd monkey patch for CambrianTrainer.get_train_dataloader...")
    out = super(CambrianTrainer, self).get_train_dataloader()
    return out

CambrianTrainer.get_train_dataloader = get_train_dataloader

def _save_checkpoint(self, model, trial, metrics=None):
    logger.info("Calling spmd monkey patch for CambrianTrainer._save_checkpoint...")
    xm.mark_step()

    logger.info("Creating state dict...")
    state_dict = {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
    }
    logger.info("Creating state dict done.")

    if self.checkpoint_manager.save(
        self.state.global_step, state_dict, force=True
    ):  # NOTE: use save instead of save_async to make sure the checkpoint is saved
        checkpoint_dir = self.checkpoint_manager._get_path(self.state.global_step)
        # ! NOTE: hard code to replace gs:// with /mnt/
        logger.info(f"Saving checkpoint at {checkpoint_dir}")
    else:
        checkpoint_dir = None

    checkpoint_dir = checkpoint_dir.replace("gs://", "/mnt/") # ! NOTE: hard code to replace gs:// with /mnt/ (0622 for testing)
    logger.info("Saving checkpoint done.")
    # dist.barrier() # ! NOTE

    logger.info("Create random number generator state...")
    # save rng state
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torchxla": xm.get_rng_state(),
    }

    rank = dist.get_rank()

    logger.info("Saving random number generator state...")
    with self.fs.open(os.path.join(checkpoint_dir, f"rng_state_rank{rank}.pth"), "wb") as f:
        torch.save(rng_state, f)
    logger.info("Saving random number generator state done.")

    logger.info("Saving lr_scheduler state and trainer state and model config...")
    if rank == 0:
        # save lr_scheduler state
        lr_state = self.lr_scheduler.state_dict()
        with self.fs.open(os.path.join(checkpoint_dir, "lr_scheduler.pth"), "wb") as f:
            torch.save(lr_state, f)
        logger.info("Saving lr_scheduler state done.")

        # save trainer state
        json_string = json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True) + "\n"
        with self.fs.open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
            f.write(json_string)
        logger.info("Saving trainer state done.")

        # save model config
        self.model.config.save_pretrained(checkpoint_dir.replace("gs://", "/mnt/"))
        logger.info("Saving model config done.")

    # dist.barrier()

    # self.consolidate_counter += 1
    # if self.consolidate_counter % self.args.consolidate_interval == 0:
    #     self._save(checkpoint_dir)

    # dist.barrier()
    xm.rendezvous("_save_checkpoint")

CambrianTrainer._save_checkpoint = _save_checkpoint

def _load_rng_state(self, resume_from_checkpoint):
    logger.info("Calling spmd monkey patch for CambrianTrainer._load_rng_state...")
    if resume_from_checkpoint is None:
        logger.info("resume_from_checkpoint is None, skip loading rng state")
        return
    resume_from_checkpoint = resume_from_checkpoint.replace("gs://", "/mnt/")
    logger.info(f"Loading RNG states from {resume_from_checkpoint}")

    with self.fs.open(os.path.join(resume_from_checkpoint, f"rng_state_rank{dist.get_rank()}.pth"), "rb") as f:
        rng_state = torch.load(f, map_location="cpu", weights_only=False)

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    xm.set_rng_state(rng_state["torchxla"])

    logger.info(f"Loading RNG states from {resume_from_checkpoint} successfully")
    xm.rendezvous("_load_rng_state")

def _load_optimizer_and_scheduler(self, resume_from_checkpoint):
    logger.info("Calling spmd monkey patch for CambrianTrainer._load_optimizer_and_scheduler...")
    if resume_from_checkpoint is None:
        logger.info("resume_from_checkpoint is None, skip loading optimizer and scheduler states")
        return
    resume_from_checkpoint = resume_from_checkpoint.replace("gs://", "/mnt/")
    logger.info(f"Loading optimizer and scheduler states from {resume_from_checkpoint}")

    with self.fs.open(os.path.join(resume_from_checkpoint, "lr_scheduler.pth"), "rb") as f:
        lr_state = torch.load(f, map_location="cpu", weights_only=False)
    self.lr_scheduler.load_state_dict(lr_state)

    logger.info(f"Loading optimizer and scheduler states from {resume_from_checkpoint} successfully")
    xm.rendezvous("_load_optimizer_and_scheduler")

def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
    logger.info("Calling spmd monkey patch for CambrianTrainer._load_from_checkpoint...")
    if resume_from_checkpoint is None:
        logger.info("resume_from_checkpoint is None, skip loading model checkpoint")
        return
    resume_from_checkpoint = resume_from_checkpoint.replace("gs://", "/mnt/")
    logger.info(f"Loading model checkpoint from {resume_from_checkpoint}")

    # Before restoring the checkpoint, the optimizer state must be primed
    # to allow state to be loaded into it.
    from torch_xla.experimental.distributed_checkpoint import prime_optimizer
    prime_optimizer(self.optimizer)

    resume_step = int(resume_from_checkpoint.rstrip('/').split('/')[-1])

    state_dict = {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
    }
    self.checkpoint_manager.restore(resume_step, state_dict)

    self.model.load_state_dict(state_dict["model"])
    self.optimizer.load_state_dict(state_dict["optimizer"])

    logger.info(f"Loaded checkpoint from {resume_from_checkpoint} successfully")
    xm.rendezvous("_load_from_checkpoint")

CambrianTrainer._load_rng_state = _load_rng_state
CambrianTrainer._load_optimizer_and_scheduler = _load_optimizer_and_scheduler
CambrianTrainer._load_from_checkpoint = _load_from_checkpoint

def sync_to_cpu(state_dict):
    def convert_fn(item):
        if isinstance(item, torch.Tensor):
            item = xm._maybe_convert_to_cpu(item).to(torch.float32)
            return item
        elif isinstance(item, dict):
            return {k: convert_fn(v) for k,v in item.items()}
        elif isinstance(item, list):
            return [convert_fn(v) for v in item]
        elif isinstance(item, tuple):
            return tuple(convert_fn(v) for v in item)
        else:
            return item
    state_dict = {
        k: convert_fn(v) for k,v in state_dict.items()
    }
    return state_dict

def _save(self, output_dir, state_dict=None):
    logger.info("Calling spmd monkey patch for CambrianTrainer._save...")
    output_dir = output_dir.replace("gs://", "/mnt/")
    logger.info(f"Saving model checkpoint to {output_dir}")

    state_dict = {"model": self.model.state_dict()}
    state_dict = sync_to_cpu(state_dict)
    logger.info("Syncing model state dict to CPU")

    if dist.get_rank() == 0:
        logger.info("Saving model state dict to CPU")
        if output_dir.startswith("/mnt/") and not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with BytesIO() as buffer, self.fs.open(os.path.join(output_dir, "model.pth.zstd"), 'wb') as f:
            torch.save(state_dict, buffer)
            f.write(zstd.compress(buffer.getvalue()))
        logger.info("Saving model state dict to CPU successfully")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir.replace("gs://", "/mnt/"))
        logger.info("Saving tokenizer to CPU successfully")
        with self.fs.open(os.path.join(output_dir, "training_args.bin"), 'wb') as f:
            torch.save(self.args, f)
        logger.info("Saving training args to CPU successfully")
        # ! NOTE: make sure model config is the last one to be saved
        self.model.config.save_pretrained(output_dir.replace("gs://", "/mnt/"))
        logger.info("Saving model config to CPU successfully")
    dist.barrier()

    logger.info(f"Saved model checkpoint to {output_dir} successfully")
    xm.rendezvous("_save")

CambrianTrainer._save = _save

def is_world_process_zero(self):
    return dist.get_rank() == 0

CambrianTrainer.is_world_process_zero = is_world_process_zero

def train(self, resume_from_checkpoint=None, *args, **kwargs):
    logger.info("Calling spmd monkey patch for CambrianTrainer.train...")
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == "true":
        logger.info("resume_from_checkpoint is set to True, resuming from the latest checkpoint")
        tracked_steps = self.checkpoint_manager.all_steps()
        logger.info(f"tracked_steps: {tracked_steps}")
        if tracked_steps:
            max_step = max(tracked_steps)
            resume_from_checkpoint = self.checkpoint_manager._get_path(max_step).replace("gs://", "/mnt/")
            logger.info(f"Max step detected, resuming from checkpoint {resume_from_checkpoint}")
        else:
            logger.warning("No checkpoint found, starting from scratch")
            resume_from_checkpoint = None
    logger.info(f"Start training with resume_from_checkpoint set to: {resume_from_checkpoint}")
    super(CambrianTrainer, self).train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)
CambrianTrainer.train = train
# ! NOTE: the above code is monkey-patched to Cambrian Trainer
################################################################################

################################################################################
# ! NOTE: the following code is monkey-patched to train_fsdp
from cambrian.train import train_fsdp
from cambrian.train.train_fsdp import get_mm_adapter_state_maybe_zero_3
def safe_save_model_for_hf_trainer(trainer, output_dir):
    logger.info("Calling spmd monkey patch for safe_save_model_for_hf_trainer...")

    output_dir = trainer._get_output_dir(None).rstrip('/') + "-last"
    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        output_dir = output_dir.replace("gs://", "/mnt/")

        keys_to_match = ['mm_projector', 'pos_emb', 'vision_sampler', 'vision_sampler_layers', 'vision_query', 'image_newline']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if dist.get_rank() == 0:
            trainer.model.config.save_pretrained(output_dir.replace("gs://", "/mnt/"))

        if not IS_XLA_AVAILABLE:
            raise NotImplementedError("Only XLA is supported for now.")

        ckpt = {"model": weight_to_save}

        logger.info(f"Saving mm_mlp_adapter to {output_dir}")
        if dist.get_rank() == 0:
            with trainer.fs.open(os.path.join(output_dir, "mm_projector.pth"), 'wb') as f:
                torch.save(ckpt, f)
        # dist.barrier()
        xm.rendezvous("save_mm_mlp_adapter")
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    trainer._save(output_dir)

train_fsdp.safe_save_model_for_hf_trainer = safe_save_model_for_hf_trainer
# ! NOTE: the above code is monkey-patched to train_fsdp
################################################################################

################################################################################
# ! NOTE: the following code is monkey-patched to pytorch 2.5.1
from torch.distributed.checkpoint.filesystem import _FileSystemWriter, _TensorLoader, _OverlappingCpuLoader, _SerialCpuLoader, _item_size, _write_item, _metadata_fn
from torch.distributed.checkpoint import filesystem
from torch.distributed.checkpoint.planner import SavePlanner, WriteItemType
from typing import Callable, List, cast
import queue
from torch.distributed.checkpoint.storage import WriteResult
from pathlib import Path
from torch.distributed.checkpoint.metadata import Metadata
import pickle

from io import UnsupportedOperation

def _write_files_from_queue(
    create_stream: Callable,
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
) -> None:
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            custom_backend_name = torch._C._get_privateuse1_backend_name()
            custom_device_mod = getattr(torch, custom_backend_name, None)

            # TODO: Using the OverlappingCpuLoader with multiple threads creates significant
            # performance degredation, observed as being related to cuda stream syncs. We
            # should try to fix this and use _OverlappingCpuLoader for all threaded cases
            if (
                thread_count == 1
                and (
                    torch.cuda.is_available()
                    or (custom_device_mod and custom_device_mod.is_available())
                )
                and inflight_threshhold > 0
            ):
                loader = _OverlappingCpuLoader(
                    planner.resolve_data,
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                loader = _SerialCpuLoader(
                    planner.resolve_data,
                )

            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

            with create_stream(file_name, "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(stream, data, write_item, storage_key)
                    )

                for tensor, write_item in loader.values():
                    assert tensor.is_cpu
                    write_results.append(
                        _write_item(stream, tensor, write_item, storage_key)
                    )

                if use_fsync:
                    try:
                        os.fsync(stream.fileno())
                    except (AttributeError, UnsupportedOperation):
                        os.sync()
            result_queue.put(write_results)
    except queue.Empty:
        pass

filesystem._write_files_from_queue = _write_files_from_queue

def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
    storage_md = {}
    for wr_list in results:
        storage_md.update({wr.index: wr.storage_data for wr in wr_list})
    metadata.storage_data = storage_md

    metadata.storage_meta = self.storage_meta()

    tmp_path = cast(Path, self.fs.concat_path(self.path, f"{_metadata_fn}.tmp"))
    with self.fs.create_stream(tmp_path, "wb") as metadata_file:
        pickle.dump(metadata, metadata_file)
        if self.sync_files:
            try:
                os.fsync(metadata_file.fileno())
            except (AttributeError, UnsupportedOperation):
                os.sync()

    # delete in-case other checkpoints were present.
    if self.fs.exists(self.metadata_path):
        self.fs.rm_file(self.metadata_path)

    self.fs.rename(tmp_path, self.metadata_path)

_FileSystemWriter.finish = finish
# ! NOTE: the above code is monkey-patched to pytorch 2.5.1
################################################################################

if __name__ == '__main__':
    
    os.environ["CAMBRIAN_LAUNCHER"] = "TORCHXLA_SPMD"
    xr.use_spmd()

    # https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_distributed_checkpoint.md#process-groups
    # The `xla://` init_method will automatically discover master worker IP, rank,
    # and global world size without requiring environment configuration on TPUs.
    dist.init_process_group(backend="gloo", init_method="xla://")

    assert _LOCAL_PROCESS_GROUP is None
    _LOCAL_PROCESS_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    logger.info("Init process group done.")

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.arange(num_devices)
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "tp"))
    xs.set_global_mesh(mesh)

    from cambrian.train.train_fsdp import train
    train(dist.get_rank())
