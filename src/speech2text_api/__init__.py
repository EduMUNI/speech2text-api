import logging
from importlib.metadata import distribution

_distribution = distribution("speech2text_api")

__version__ = _distribution.version

logger = logging.getLogger(__name__)
