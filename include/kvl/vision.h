#ifndef KVL_VISION_H
#define KVL_VISION_H

#include "kvl/trunk_store.h"

/* Official Kimi-VL MoonViT forward for one image.
 *
 * `patches` is row-major [grid_h*grid_w, 3*14*14] FP32 after official image
 * normalization/patchification. grid_h/grid_w must be positive, even and <= 512.
 * `out` must hold (grid_h/2)*(grid_w/2)*2048 floats. The vision store is an aligned
 * vision.bin/vision.idx produced by tools/pack_vision.py.
 *
 * The implementation streams one vision block's weights at a time and therefore does not
 * require the ~0.83 GiB MoonViT/projector weights to remain resident. */
int kvl_vision_forward(KvlTrunkStore *vision_store,
                       const float *patches,
                       int grid_h,
                       int grid_w,
                       float *out,
                       int *out_tokens);

#endif
