module MXNet.NN.ModelZoo.Utils.Box where

import RIO
import Data.Array.Repa (Array, U, DIM1, Z(..), (:.)(..))
import qualified Data.Array.Repa as Repa

import MXNet.NN.ModelZoo.Utils.Repa


type RBox = Array U DIM1 Float

bboxArea :: RBox -> Float
bboxArea box = (box ^#! 2 - box ^#! 0 + 1) * (box ^#! 3 - box ^#! 1 + 1)

bboxIntersect :: RBox -> RBox -> Maybe RBox
bboxIntersect box1 box2 | not valid = Nothing
                        | otherwise = Just $ Repa.fromListUnboxed (Z:.4) [x1, y1, x2, y2]
  where
    valid = x2 - x1 > 0 && y2 - y1 > 0
    x1 = max (box1 ^#! 0) (box2 ^#! 0)
    x2 = min (box1 ^#! 2) (box2 ^#! 2)
    y1 = max (box1 ^#! 1) (box2 ^#! 1)
    y2 = min (box1 ^#! 3) (box2 ^#! 3)

bboxIOU :: RBox -> RBox -> Float
bboxIOU box1 box2 = case bboxIntersect box1 box2 of
                      Nothing -> 0
                      Just boxI -> let areaI = bboxArea boxI
                                       areaU = bboxArea box1 + bboxArea box2 - areaI
                                   in areaI / areaU

whctr :: RBox -> RBox
whctr box1 = Repa.fromListUnboxed (Z:.4) [w, h, x, y]
  where
    [x0, y0, x1, y1] = Repa.toList box1
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    x = x0 + 0.5 * (w - 1)
    y = y0 + 0.5 * (h - 1)

bboxTransform :: RBox -> RBox -> RBox -> RBox
bboxTransform stds box1 box2 =
    let [w1, h1, cx1, cy1] = Repa.toList $ whctr box1
        [w2, h2, cx2, cy2] = Repa.toList $ whctr box2
        dx = (cx2 - cx1) / (w1 + 1e-14)
        dy = (cy2 - cy1) / (h1 + 1e-14)
        dw = log (w2 / w1)
        dh = log (h2 / h1)
    in Repa.computeS $ Repa.fromListUnboxed (Z:.4) [dx, dy, dw, dh] Repa./^ stds

ctrwh :: RBox -> RBox
ctrwh box1 = Repa.fromListUnboxed (Z:.4) [x0, y0, x1, y1]
  where
    [w, h, cx, cy] = Repa.toList box1
    x0 = cx - 0.5 * (w - 1)
    y0 = cy - 0.5 * (h - 1)
    x1 = w + x0 - 1
    y1 = h + y0 - 1

bboxTransInv :: RBox -> RBox -> RBox -> RBox
bboxTransInv stds box delta =
    let [dx, dy, dw, dh] = Repa.toList $ delta Repa.*^ stds
        [w1, h1, cx1, cy1] = Repa.toList $ whctr box
        w2 = exp dw * w1
        h2 = exp dh * w2
        cx2 = dx * w1 + cx1
        cy2 = dy * h1 + cy1
    in ctrwh $ Repa.fromListUnboxed (Z:.4) [w2, h2, cx2, cy2]


bboxClip :: Float -> Float -> RBox -> RBox
bboxClip height width box = Repa.fromListUnboxed (Z:.4) [x0', y0', x1', y1']
  where
    [x0, y0, x1, y1] = Repa.toList box
    w' = width - 1
    h' = height - 1
    x0' = max 0 (min x0 w')
    y0' = max 0 (min y0 h')
    x1' = max 0 (min x1 w')
    y1' = max 0 (min y1 h')
