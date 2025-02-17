# __init__.py
import os
from .IF_Zonos import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Specify the web directory for any web components
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]