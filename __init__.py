from .nodes import *
try:
    from typing import override
except Exception:
    try:
        from typing_extensions import override
    except Exception:
        def override(func):
            return func

class Sam3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadSam3Model,
            Sam3ImageSegmentation,
            Sam3VideoSegmentation,
            Sam3VideoModelExtraConfig,
            Sam3Visualization,
            Sam3GetObjectIds,
            Sam3GetObjectMask,
            StringToBBox,
            FramesEditor,
        ]

async def comfy_entrypoint() -> Sam3Extension:
    return Sam3Extension()

# Web directory for custom UI (interactive SAM3 detector)
WEB_DIRECTORY = "./web"

# Export for ComfyUI
__all__ = ['WEB_DIRECTORY']

