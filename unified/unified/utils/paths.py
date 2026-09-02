"""Machine-specific filesystem roots (upstream checkouts + pretrained weights).

Two things in this project are properties of the *machine*, not of an
experiment, and both move whenever the project is copied to a new server:

* the sibling upstream checkouts several adapters import architecture code from
  (``VISTA/``, ``3DINO/``, ``CT-CLIP/``, ``SAM-Med3D/``, ``SuPreM/``,
  ``BiomedParse/``, ``Merlin/``), and
* the directory holding the pretrained checkpoints the model configs name.

Both are resolved from this file's own location — the checkouts sit next to the
``unified/`` repo — so a fresh clone works with no edits. Override either with
an environment variable when the layout differs:

    FM_ROOT       directory holding the upstream checkouts  (default: repo parent)
    WEIGHTS_ROOT  pretrained checkpoints                    (default: $FM_ROOT/weights)
"""
from __future__ import annotations
import os
from pathlib import Path

# .../<FM_ROOT>/unified/unified/utils/paths.py  ->  parents[3] == <FM_ROOT>
_DEFAULT_FM_ROOT = Path(__file__).resolve().parents[3]

FM_ROOT = Path(os.environ.get("FM_ROOT", _DEFAULT_FM_ROOT))
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", FM_ROOT / "weights"))


def upstream(*parts: str) -> Path:
    """Path inside a sibling upstream checkout, e.g. ``upstream("VISTA", "vista3d")``.

    Does not check existence — each adapter raises its own error naming the file
    it wanted, which is more useful than a generic "repo missing" here.
    """
    return FM_ROOT.joinpath(*parts)


def weights(*parts: str) -> Path:
    """Path inside the pretrained-weights tree."""
    return WEIGHTS_ROOT.joinpath(*parts)
