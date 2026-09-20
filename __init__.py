"""ComfyUI-Qwen 2.1 Options: reference strength controls for Qwen Image 2.1."""
import inspect

from . import nodes

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from comfy_api.internal import _ComfyNodeInternal
except Exception:  # very old ComfyUI: no V3 support
    _ComfyNodeInternal = None

# Auto-pick every node class defined in nodes.py:
# - V3 nodes (io.ComfyNode subclasses with define_schema)
# - legacy V1 nodes (classes exposing INPUT_TYPES + FUNCTION, named *Node)
for _name, _obj in vars(nodes).items():
    if not inspect.isclass(_obj):
        continue
    if (_ComfyNodeInternal is not None
            and issubclass(_obj, _ComfyNodeInternal)
            and hasattr(_obj, 'define_schema')):
        _schema = _obj.define_schema()
        NODE_CLASS_MAPPINGS[_schema.node_id] = _obj
        # NOTE: for V3 nodes the display name comes from the schema itself
        # (GET_NODE_INFO_V1), NODE_DISPLAY_NAME_MAPPINGS is only used by legacy V1 nodes
        NODE_DISPLAY_NAME_MAPPINGS[_schema.node_id] = (
            getattr(_obj, 'TITLE', None) or _schema.display_name
            or _schema.node_id)
    elif (_name.endswith('Node')
            and hasattr(_obj, 'INPUT_TYPES') and hasattr(_obj, 'FUNCTION')):
        NODE_CLASS_MAPPINGS[_name] = _obj
        NODE_DISPLAY_NAME_MAPPINGS[_name] = getattr(_obj, 'TITLE', _name)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
