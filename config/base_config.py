import functools
from dataclasses import dataclass

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing_wrapper,
)
from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@dataclass
class base_config:

    # seed
    seed: int = 2022
    verbose: bool = True  # how much info to show...
    # how many mini batches to time with
    total_steps_to_run: int = 5

    # training
    batch_size_training: int = 8#64
    num_epochs: int = 1

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD #ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    run_profiler: bool = False

    # backward prefetch
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # disable forward_prefetch since it currently doesn't work with activation
    # checkpointing for several cases
    forward_prefetch = False

    # log
    log_every: int = 1

    # dataloaders
    num_workers_dataloader: int = 2

    # policies
    use_mixed_precision: bool = False
    # this is only for fp32 scenario...
    use_tf32: bool = False


    # activation checkpointing
    fsdp_activation_checkpointing: bool = False

    # validation
    run_validation: bool = False
    val_batch_size = 8#4#8#32#8#128

    # logging
    track_memory = False
    memory_report: bool = False
    nccl_debug_handler: bool = True
    distributed_debug: bool = True


def get_policy_base(blocks):
    recursive_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=blocks,
    )
    return recursive_policy
    # The ParamExecOrderPolicy that is in development
    # from torch.distributed.fsdp.wrap import (
    #     always_wrap_policy,
    #     ParamExecOrderPolicy,
    #     HandleInitMode,
    # )
    # return ParamExecOrderPolicy(
    #     handle_init_mode=HandleInitMode.MODULE_LEVEL,
    #     bucket_size=int(17000000 * 2 + 1),
    #     module_level_group_policy=recursive_policy,
    # )


def fsdp_checkpointing_base(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, blocks)
    apply_activation_checkpointing_wrapper(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )