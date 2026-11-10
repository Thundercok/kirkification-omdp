"""
Canonical MediaPipe Face Mesh Landmark Index Definitions.
Reference: Google MediaPipe Face Mesh (468 3D landmarks).
"""

from typing import List, Set

# Outer Cranial / Face Oval Contour Landmarks (Head boundary)
FACE_OVAL_INDICES: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Inner Facial Features (Eyes, Eyebrows, Nose, Lips)
LEFT_EYE_INDICES: List[int] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_INDICES: List[int] = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]

LEFT_EYEBROW_INDICES: List[int] = [
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46
]

RIGHT_EYEBROW_INDICES: List[int] = [
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276
]

NOSE_INDICES: List[int] = [
    1, 2, 98, 327, 195, 5, 4, 197, 168, 6, 19, 94, 97, 326, 278, 48, 219, 439, 236, 456
]

LIPS_INDICES: List[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415
]

# Consolidated set of inner facial features
INNER_FACE_INDICES: List[int] = sorted(list(set(
    LEFT_EYE_INDICES + RIGHT_EYE_INDICES +
    LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES +
    NOSE_INDICES + LIPS_INDICES
)))
