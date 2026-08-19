"""Per-foundation-model backbone adapters.

Importing this package registers all available adapters with the global registry.
Each adapter conforms to BackboneInterface and produces the 5-level contract
pyramid: strides {1, 2, 4, 8, 16} with channels {32, 64, 128, 256, 512}, finest
first. See docs/HEAD_DESIGN.md.
"""
from . import voco               # noqa: F401
from . import vista3d            # noqa: F401
from . import dino3d             # noqa: F401
from . import stunet             # noqa: F401
from . import biomedparse        # noqa: F401
from . import ctclip             # noqa: F401
from . import ctfm               # noqa: F401
from . import suprem_swinunetr   # noqa: F401
from . import suprem_segresnet   # noqa: F401
from . import suprem_unet        # noqa: F401
from . import sam_med3d          # noqa: F401
from . import merlin             # noqa: F401
