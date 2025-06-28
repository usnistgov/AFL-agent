import importlib
import pkgutil
from typing import Callable, Dict

Loader = Callable[[str], 'xr.Dataset']


def load_plugins() -> Dict[str, Loader]:
    """Discover available dataset loader plugins."""
    plugins: Dict[str, Loader] = {}
    package = importlib.import_module(__name__)
    for modinfo in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{__name__}.{modinfo.name}")
        loader = getattr(module, "load", None)
        extensions = getattr(module, "extensions", [])
        if callable(loader):
            for ext in extensions:
                plugins[ext] = loader
    return plugins
