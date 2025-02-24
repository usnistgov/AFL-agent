from .Pipeline import *
from .AcquisitionFunction import *
from .Extrapolator import *
from .PairMetric import *
from .PhaseLabeler import *
from .Preprocessor import *
from .Generator import *
from .plotting import *
from .Boundary import *

import os
import subprocess
import warnings
from pathlib import Path

def _get_version():
    try:
        # First try to import existing version
        from ._version import __version__
        return __version__
    except ImportError:
        try:
            # Try to generate version file using hatch-vcs if we're in a git repo
            repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], 
                                             stderr=subprocess.DEVNULL).decode('utf-8').strip()
            if os.path.exists(os.path.join(repo_root, '.git')):
                try:
                    # Get version from git using similar logic to hatch-vcs
                    git_tag = subprocess.check_output(['git', 'describe', '--tags', '--always'],
                                                    stderr=subprocess.DEVNULL).decode('utf-8').strip()
                    version = git_tag.lstrip('v')
                    # Write version to file
                    version_file = Path(__file__).parent / '_version.py'
                    version_file.write_text(f'__version__ = "{version}"\n')
                    return version
                except subprocess.CalledProcessError:
                    pass
        except (subprocess.CalledProcessError, OSError):
            pass
        
        # Fallback for when we can't determine the version
        warnings.warn("Could not determine version from git or _version.py. Using default version.", ImportWarning)
        return "0.0.0+dev"

__version__ = _get_version()

