"""Standalone Hugging Face Transformers package for CoRD."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_cord import CordConfig
from .modeling_cord import CordForCausalLM, CordModel, CordPreTrainedModel
from .outputs import CordCausalLMOutputWithPast, CordConceptPacket, CordModelOutput, CordSearchOutput
from .state_graph import CordDecodeContext, CordOperator, CordSearchConfig, CordVerifierResult
from .training import (
    CordLossTargets,
    build_cord_optimizer_param_groups,
    compute_cord_loss,
    update_router_biases,
)

__version__ = "0.1.0"


def register_cord_auto_classes() -> None:
    """Register CoRD classes in the process-local Transformers auto maps."""

    try:
        AutoConfig.register(CordConfig.model_type, CordConfig)
    except ValueError as error:
        if AutoConfig.for_model(CordConfig.model_type).__class__ is not CordConfig:
            raise error
    try:
        AutoModel.register(CordConfig, CordModel)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(CordConfig, CordForCausalLM)
    except ValueError:
        pass


register_cord_auto_classes()

__all__ = [
    "CordConfig",
    "CordPreTrainedModel",
    "CordModel",
    "CordForCausalLM",
    "CordConceptPacket",
    "CordModelOutput",
    "CordCausalLMOutputWithPast",
    "CordSearchOutput",
    "CordOperator",
    "CordSearchConfig",
    "CordDecodeContext",
    "CordVerifierResult",
    "CordLossTargets",
    "compute_cord_loss",
    "build_cord_optimizer_param_groups",
    "update_router_biases",
    "register_cord_auto_classes",
]
