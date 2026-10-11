# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Prepare frozen, normalized AutoEP expert parameters for bounded direct transfer."""

from typing import Mapping

import torch

from .affine_transfer import PlanBudget, plan_transfer
from .affine_transfer_executor import TransferRequest
from .autoep_affine import autoep_metadata_to_affine_map
from .constants import AUTOEP_PARAM_EP_RANK, AUTOEP_PARAM_LOGICAL_SHAPE, DS_AUTOEP_UC_META

__all__ = ['make_autoep_transfer_request']


def _parameter_buffers(parameters, metadata, map_, key):
    buffers = {}
    for location, parameter in parameters.items():
        attached = getattr(parameter, DS_AUTOEP_UC_META, None)
        if attached is None:
            raise ValueError(f'{key}: parameter has no AutoEP UC metadata')
        if attached.get(AUTOEP_PARAM_EP_RANK) != location:
            raise ValueError(f'{key}: parameter metadata disagrees with its EP-local location')
        shape = tuple(metadata[AUTOEP_PARAM_LOGICAL_SHAPE])
        if tuple(attached[AUTOEP_PARAM_LOGICAL_SHAPE]) != shape:
            raise ValueError(f'{key}: parameter metadata has a different logical shape')
        attached_map = autoep_metadata_to_affine_map(attached, shape, key)
        if attached_map.to_dict() != map_.to_dict():
            raise ValueError(f'{key}: local parameter placement differs from the transfer metadata')
        if tuple(parameter.shape) != map_.shard_shapes[location] or not parameter.is_contiguous():
            raise ValueError(f'{key}: normalize ZeRO/EDP fragments to packed EP tensors before transfer')
        buffers[location] = parameter.detach()
    return buffers


def make_autoep_transfer_request(key: str,
                                 source_metadata: Mapping,
                                 target_metadata: Mapping,
                                 source_parameters: Mapping[int, torch.nn.Parameter],
                                 target_parameters: Mapping[int, torch.nn.Parameter],
                                 plan_budget: PlanBudget = PlanBudget(),
                                 replica_selector=None) -> TransferRequest:
    """Use persisted placement and actual Parameter storage, with explicit EP-local keys.

    Metadata describes the complete source and target placement on every participant, including
    ranks without local parameters. Only frozen, normalized expert weights are covered; this
    function does not gather ZeRO fragments or infer geometry from parameter names.

    Local preparation errors are carried into ``execute_transfers`` so peers agree on rejection
    before any parameter is written. Build the whole batch, then execute it with one dtype.
    """
    try:
        source_shape = tuple(source_metadata[AUTOEP_PARAM_LOGICAL_SHAPE])
        target_shape = tuple(target_metadata[AUTOEP_PARAM_LOGICAL_SHAPE])
        if source_shape != target_shape:
            raise ValueError(f'{key}: source and target logical shapes differ')
        source_map = autoep_metadata_to_affine_map(source_metadata, source_shape, key)
        target_map = autoep_metadata_to_affine_map(target_metadata, target_shape, key)
        plan = plan_transfer(target_map, source_map, plan_budget, replica_selector=replica_selector)
        sources = _parameter_buffers(source_parameters, source_metadata, source_map, key)
        targets = _parameter_buffers(target_parameters, target_metadata, target_map, key)
        return TransferRequest(key, plan, sources, targets)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return TransferRequest(key, None, {}, {}, preflight_error=f'{key}: {error}')
