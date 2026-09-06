"""Missing image/annotation pairs observed in the complete Adobe v1.0 release.

All ten image shards were downloaded, hashed and fully traversed on 2026-09-05.
The extraction manifest accounts for 41,874 eligible images; seven additional
line JSONs have no corresponding extracted image. Keep this list narrow: any
other missing image is an error, and a listed image is included if available.
"""

ADOBE_V1_MISSING_LINE_IMAGES = frozenset({
    "162724", "168329", "176337", "182570", "187288", "192989", "196163",
})
