"""Metadata-only name fixes for the configured CyberLite BF16 source.

The pinned EXL3 Config.get_tensor_name_fixes hook accepts suffix substitutions.
Use complete source names as suffixes so the source bytes and offsets are never
rewritten. Install the returned mapping on the isolated runtime's config class
before constructing Config; this module does not import or patch that runtime.
"""

from collections.abc import Iterable


SOURCE_PREFIX = "model.language_model.language_model.language_model."
RUNTIME_PREFIX = "model.language_model."
VISION_SOURCE_PREFIX = "model.language_model.visual."
VISION_RUNTIME_PREFIX = "model.visual."


def tensor_name_fixes(source_keys: Iterable[str]) -> dict[str, str]:
    """Return exact-key EXL3 loader fixes, rejecting ambiguous destinations.

    Vision names lose their extra language_model prefix. MTP, lm_head, and
    already canonical names remain unchanged. This does not select which
    components to convert; the caller must explicitly select text-only scope.
    A mixed source with two names for the same destination fails instead of silently
    selecting whichever shard the EXL3 directory scan happens to visit last.
    """
    fixes: dict[str, str] = {}
    destinations: dict[str, str] = {}
    for source in source_keys:
        if not isinstance(source, str) or not source:
            raise ValueError("Tensor names must be nonempty strings")
        destination = (
            RUNTIME_PREFIX + source[len(SOURCE_PREFIX):]
            if source.startswith(SOURCE_PREFIX)
            else source
        )
        if source.startswith(VISION_SOURCE_PREFIX):
            destination = VISION_RUNTIME_PREFIX + source[len(VISION_SOURCE_PREFIX):]
        if destination in destinations:
            raise ValueError(
                f"Duplicate tensor destination {destination!r}: "
                f"{destinations[destination]!r} and {source!r}"
            )
        destinations[destination] = source
        if destination != source:
            fixes[source] = destination
    return fixes
