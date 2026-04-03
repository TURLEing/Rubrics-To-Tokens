import gc
import time
from typing import Tuple, Iterable
from collections import OrderedDict

import torch
import torch.distributed as dist
import vllm

from roll.utils.collective import collective
from roll.platforms import current_platform
from roll.utils.send_recv_utils import named_tensors_from_bucket, monkey_patch_torch_reductions
from roll.utils.logging import get_logger
from roll.third_party.vllm.vllm_utils import TensorLoRARequest, patch_vllm_lora_manager
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer

logger = get_logger()


class WorkerBase:
    def custom_init_worker(self, *args, **kwargs):
        self.weight_loaded : bool = True
        self.kv_cache_loaded : bool = True
        self.buffers = None
        self.buffer_cache = None

    def reload_model(self):
        if not self.weight_loaded:
            self.wake_up(["weights"])
            self.weight_loaded = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # before updating the parameters, we need to reinitialize the previously released model
        self.reload_model()
        if vllm.__version__ < "0.8.5":
            from roll.third_party.vllm.vllm_utils import patch_vllm_moe_model_weight_loader
            patch_vllm_moe_model_weight_loader(self.model_runner.model)
        self.model_runner.model.load_weights(weights=weights)

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.wake_up(["kv_cache"])
            self.kv_cache_loaded = True
        if vllm.__version__ < "0.8.5" and  self.buffers is not None:
            # https://github.com/vllm-project/vllm/issues/16564
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self.buffers:
                    buffer.data.copy_(self.buffers[name].data)
            self.buffers = None

    def offload_states(self, level):
        assert (self.weight_loaded and self.kv_cache_loaded) or (not self.weight_loaded and not self.kv_cache_loaded)
        if not self.weight_loaded:
            return
        if vllm.__version__ < "0.8.5" and level == 2:
            # https://github.com/vllm-project/vllm/issues/16564
            model = self.model_runner.model
            self.buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
        self.sleep(level)
        self.weight_loaded = False
        self.kv_cache_loaded = False
        if hasattr(self, 'recv_manager'):
            self.recv_manager.clear()
        gc.collect()
        current_platform.empty_cache()

    def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        group_rank = self.rank + rank_offset
        collective.init_collective_group(world_size, rank=group_rank, backend=backend, group_name=group_name,
                                         master_addr=master_address, master_port=master_port)
        logger.info(f"setup_collective_group: {group_name} rank: {group_rank} world_size: {world_size}")

    def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        weights_and_handles = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            target_dtype = (
                dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            )
            weight = torch.empty(shape, dtype=target_dtype, device=self.device)
            handle = collective.broadcast(tensor=weight, src_rank=0, group_name=group_name, async_op=True)
            weights_and_handles.append((name, weight, handle))
        def weights_iter():
            for name, weight, handle in weights_and_handles:
                handle.wait()
                yield name, weight

        if is_lora:
            for name, weight in weights_iter():
                self.lora_params[name] = weight
            return
        self.load_weights(weights=weights_iter())

    def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        monkey_patch_torch_reductions()
        bucket_with_meta = MultiprocessingSerializer.deserialize(serialized_named_tensors[self.rank])
        named_params = named_tensors_from_bucket(**bucket_with_meta)
        if is_lora:
            for name, weight in named_params:
                self.lora_params[name] = weight
            return
        self.load_weights([(name, weight) for name, weight in named_params])


class WorkerV1(WorkerBase):
    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
        self.lora_params = OrderedDict()
        patch_vllm_lora_manager()

    # Use custom prefix because worker_extension_cls can not has
    # conflicting method name with vllm worker.
    def custom_add_lora(self, peft_config) -> bool:
        lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = TensorLoRARequest(
            lora_name=f"{lora_int_id}",
            lora_int_id=lora_int_id,
            lora_path="dummy_lora_path",
            peft_config=peft_config,
            lora_tensors=self.lora_params,
        )
        del self.lora_params
        self.lora_params = OrderedDict()
        super().reload_model()
        return self.model_runner.add_lora(lora_request)
