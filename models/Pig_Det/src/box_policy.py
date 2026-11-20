"""Team contract: expand detected bounding boxes by 12% in width AND height.
Expansion is around center, i.e. 6% of original dimension on each side.
Clip to the original frame; do not mutate tracking boxes or training labels.
"""
from math import isfinite

BOX_EXPANSION = 0.12

def expand_xyxy(box, image_width, image_height, fraction=BOX_EXPANSION):
    x1, y1, x2, y2 = map(float, box)
    if not all(map(isfinite, (x1, y1, x2, y2, image_width, image_height, fraction))):
        raise ValueError('nonfinite box/dimensions')
    if not (image_width > 0 and image_height > 0 and 0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height and fraction >= 0):
        raise ValueError('invalid box/dimensions/expansion')
    dx = (x2 - x1) * fraction / 2
    dy = (y2 - y1) * fraction / 2
    return [max(0., x1-dx), max(0., y1-dy), min(float(image_width), x2+dx), min(float(image_height), y2+dy)]
