"""FMAD-OOD: Foundation Model Anomaly Detection under Distribution Shift."""

import torch

try:
    import torch.utils._pytree as pytree

    if not hasattr(pytree, "register_pytree_node") and hasattr(
        pytree, "_register_pytree_node"
    ):

        def _compat_register_pytree_node(typ, flatten_fn, unflatten_fn, **kwargs):
            supported_kwargs = {
                k: kwargs[k]
                for k in ("to_dumpable_context", "from_dumpable_context")
                if k in kwargs
            }
            return pytree._register_pytree_node(
                typ, flatten_fn, unflatten_fn, **supported_kwargs
            )

        pytree.register_pytree_node = _compat_register_pytree_node
except Exception:
    pass
