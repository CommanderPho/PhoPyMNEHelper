"""Resource access helpers for phopymnehelper package."""
import importlib.resources as resources
from pathlib import Path


def get_resource_path(relative_path: str) -> Path:
    """
    Get the absolute path to a resource file within the package.
    
    Args:
        relative_path: Relative path from the package root (e.g., "resources/ElectrodeLayouts/simplified/head_bem_1922V_fill_fixed.stl")
    
    Returns:
        Path object pointing to the resource file
    
    Example:
        >>> stl_path = get_resource_path("resources/ElectrodeLayouts/simplified/head_bem_1922V_fill_fixed.stl")
        >>> print(stl_path)
    """
    return Path(resources.files("phopymnehelper").joinpath(relative_path)).resolve()


def get_simplified_head_mesh_path() -> Path:
    """Get the path to the simplified head BEM mesh STL file."""
    return get_resource_path("resources/ElectrodeLayouts/simplified/head_bem_1922V_fill_fixed.stl")


def get_simplified_fullhead_mesh_path() -> Path:
    """Get the path to the simplified full head mesh STL file."""
    return get_resource_path("resources/ElectrodeLayouts/simplified/pho_2025-06-23_FullHead_0007_fixed.stl")

