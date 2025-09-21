# Bridge package so that `import ctcdecode` works when project root is on sys.path.
# Re-export everything from the inner actual package directory.
from .ctcdecode import *  # noqa
