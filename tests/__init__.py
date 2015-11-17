import numpy as np

from KESMAnalysis.models import SampleRegion


def tupleToSafeString(tup, dtype):
    numItems = len(tup)

    if dtype == np.uint8:
        fmt = "%i"
    if dtype == np.float64:
        fmt = "%0.5f"
    if dtype == np.bool:
        fmt = "%i"

    fmtString = "_".join([fmt] * numItems)
    return fmtString % tup


def getTestRegions():
    return [
        SampleRegion(**{
            "sampleName": "ii-2001-33-2-rat-brain",
            "zBounds": (28, 28.05),
            "offsetBounds": (2, 3),
            "xyMargins": (9000, 8000, 0, 0),
        }),
        SampleRegion(**{
            "sampleName": "ii-2001-33-2-rat-brain",
            "zBounds": (28, 28.05),
            "offsetBounds": (2, 3),
            "xyMargins": (8000, 8000, 0, 0),
        }),
        SampleRegion(**{
            "sampleName": "ii-2001-33-2-rat-brain",
            "zBounds": (28, 28.05),
            "offsetBounds": (2, 2),
            "xyMargins": (9500, 9000, 200, 0),
        }),
        SampleRegion(**{
            "sampleName": "ii-2001-33-2-rat-brain",
            "zBounds": (28, 29.40),
            "offsetBounds": (2, 3),
            "xyMargins": (8000, 8000, 0, 0),
        }),
    ]
