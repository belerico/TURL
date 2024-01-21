__version__ = "2.1.1"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging

    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from .data import is_sklearn_available

# Files and general utilities
from .file_utils import is_tf_available, is_torch_available

if is_sklearn_available():
    pass

# Tokenizers

# Configurations

# Modeling
if is_torch_available():
    # Optimization
    pass


# TensorFlow
if is_tf_available():
    pass

# TF 2.0 <=> PyTorch conversion utilities

if not is_tf_available() and not is_torch_available():
    logger.warning(
        "Neither PyTorch nor TensorFlow >= 2.0 have been found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
