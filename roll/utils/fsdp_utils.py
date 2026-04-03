from abc import ABC
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import Shard

from roll.platforms import current_platform

try:
    from torch.distributed.device_mesh import DeviceMesh
except ImportError:
    DeviceMesh = None

fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


def get_shard_placement_fn(fsdp_size):
    """
    Choose the dimension that can divide fsdp_size to avoid padding
    Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py

    """

    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def apply_fsdp2(model, fsdp_kwargs, config, is_lora=False):
    """
    model: AutoModelForCausalLM

    Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    and LoRA Patch: https://github.com/volcengine/verl/issues/3470

    """
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap",
        default_transformer_cls_names_to_wrap,
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    lora_modules = []
    modules = []
    for name, module in model.named_modules():
        if is_lora and (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            lora_modules.append(module)

        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(lora_modules):
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    for idx, module in enumerate(modules):
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def fsdp2_load_full_state_dict(
    model: torch.nn.Module,
    full_state: dict,
    device_mesh=None,
    cpu_offload=None,
):
    """
    Reference: https://github1s.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py

    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """

    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    device_id = current_platform.current_device()

    if dist.get_rank() == 0:
        model = model.to(device=device_id, non_blocking=True)
    else:
        model = model.to_empty(device=device_id)

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=cpu_offload,
        broadcast_from_rank0=True,
    )
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        # Ensure model is on CPU but buffers are on GPU for FSDP2 CPU offload
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(device_id)
