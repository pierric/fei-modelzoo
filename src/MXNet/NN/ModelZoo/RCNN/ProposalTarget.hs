module MXNet.NN.ModelZoo.RCNN.ProposalTarget where

import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as UVM
import qualified Data.Array.Repa as Repa
import Data.Array.Repa (Array, U, D)
import Data.Array.Repa.Index
import Data.Random (shuffle, runRVar, StdRandom(..))
import Data.Random.Vector (randomElement)
import Control.Lens ((^.), (^?!), ix, makeLenses)
import Control.Monad (replicateM, forM_, join)
import Control.Exception.Base(assert)
import GHC.Stack (HasCallStack)

import MXNet.Base
import MXNet.Base.Operators.NDArray (_set_value_upd)
import MXNet.NN.ModelZoo.Utils.Box
import MXNet.NN.ModelZoo.Utils.Repa


data ProposalTargetProp = ProposalTargetProp {
    _pt_num_classes :: Int,
    _pt_batch_images :: Int,
    _pt_batch_rois :: Int,
    _pt_fg_fraction :: Float,
    _pt_fg_overlap :: Float,
    _pt_box_stds :: [Float]
}
makeLenses ''ProposalTargetProp

instance CustomOperationProp ProposalTargetProp where
    prop_list_arguments _        = ["rois", "gt_boxes"]
    prop_list_outputs _          = ["rois_output", "label", "bbox_target", "bbox_weight"]
    prop_list_auxiliary_states _ = []
    prop_infer_shape prop [rpn_rois_shape, gt_boxes_shape] =
        let prop_batch_size   = prop ^. pt_batch_rois
            prop_num_classes  = prop ^. pt_num_classes
            output_rois_shape = [prop_batch_size, 5]
            label_shape       = [prop_batch_size]
            bbox_target_shape = [prop_batch_size, prop_num_classes * 4]
            bbox_weight_shape = [prop_batch_size, prop_num_classes * 4]
        in ([rpn_rois_shape, gt_boxes_shape],
            [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape],
            [])
    prop_declare_backward_dependency prop grad_out data_in data_out = []

    data Operation ProposalTargetProp = ProposalTarget ProposalTargetProp
    prop_create_operator prop _ _ = return (ProposalTarget prop)

instance CustomOperation (Operation ProposalTargetProp) where
    forward (ProposalTarget prop) [ReqWrite, ReqWrite, ReqWrite, ReqWrite] inputs outputs aux is_train = do
        -- :param: rois, shape of (N*nms_top_n, 5), [image_index_in_batch, bbox0, bbox1, bbox2, bbox3]
        -- :param: gt_boxes, shape of (N, M, 5), M varies per image. [bbox0, bbox1, bbox2, bbox3, class]
        let [rois, gt_boxes] = inputs
            [rois_output, label_output, bbox_target_output, bbox_weight_output] = outputs
            batch_size = prop ^. pt_batch_images

        -- convert NDArray to Vector of Repa array.
        r_rois   <- toRepa @DIM2 (NDArray rois)     >>= return . vunstack
        r_gt     <- toRepa @DIM3 (NDArray gt_boxes) >>= return . vunstack

        assert (batch_size == length r_gt) (return ())

        -- sample rois for each example of the batch
        -- and concatenate
        (rois, labels, bbox_targets, bbox_weights) <- V.unzip4 <$>
            V.mapM (sample_batch r_rois r_gt) (V.enumFromN (0 :: Int) batch_size)

        let rois'   = vstack $ V.map (Repa.computeUnboxedS . Repa.reshape (Z :. 1 :. 5 :: DIM2)) $ join rois
            labels' = join labels
            bbox_targets' = vstack bbox_targets
            bbox_weights' = vstack bbox_weights

            rois_output_nd        = NDArray rois_output        :: NDArray Float
            bbox_target_output_nd = NDArray bbox_target_output :: NDArray Float
            bbox_weight_output_nd = NDArray bbox_weight_output :: NDArray Float
            label_output_nd       = NDArray label_output       :: NDArray Float

        copyFromRepa rois_output_nd rois'
        copyFromRepa bbox_target_output_nd bbox_targets'
        copyFromRepa bbox_weight_output_nd bbox_weights'
        copyFromVector label_output_nd $ V.convert labels'

      where
        sample_batch r_rois r_gt index = do
            let rois_this_image   = V.filter (\roi -> floor (roi ^#! 0) == index) r_rois
                all_gt_this_image = vunstack $ r_gt ^%! index
                gt_this_image     = V.filter (\gt  -> gt  ^#! 4 > 0) all_gt_this_image

            -- WHY?
            -- append gt boxes to rois
            let index_r = Repa.fromListUnboxed (Z :. 1) [fromIntegral index]
                prepend_index = Repa.computeUnboxedS . (index_r Repa.++)
                gt_boxes_as_rois = V.map (\gt -> prepend_index $ Repa.extract (Z :. 0) (Z :. 4) gt)
                                         gt_this_image
                rois_this_image' = rois_this_image V.++ gt_boxes_as_rois

            sample_rois rois_this_image' gt_this_image prop

    backward _ [ReqWrite, ReqWrite] _ _ [in_grad_0, in_grad_1] _ _ = do
        _set_value_upd [in_grad_0] (#src := 0 .& Nil)
        _set_value_upd [in_grad_1] (#src := 0 .& Nil)


sample_rois :: V.Vector (Array U DIM1 Float) -> V.Vector (Array U DIM1 Float)
            -> ProposalTargetProp
            -> IO (V.Vector (Array U DIM1 Float),
                   V.Vector Float,
                   Array U DIM2 Float,
                   Array U DIM2 Float)
sample_rois rois gt prop = do
    -- :param rois: [num_rois, 5] (batch_index, x1, y1, x2, y2)
    -- :param gt: [num_rois, 5] (x1, y1, x2, y2, cls)
    --
    -- :returns: sampled (rois, labels, regression, weight)
    let aoi_boxes = V.map (Repa.computeUnboxedS . Repa.extract (Z:.1) (Z:.4)) rois
        gt_boxes  = V.map (Repa.computeUnboxedS . Repa.extract (Z:.0) (Z:.4)) gt
        overlaps  = Repa.computeUnboxedS $ overlapMatrix aoi_boxes gt_boxes

    let maxIndices = argMax overlaps
        gt_chosen  = V.map (gt ^%!) maxIndices

    -- a uniform sampling w/o replacement from the fg boxes if there are too many
    fg_indexes <- let fg_indexes = V.filter (\(i, j) -> Repa.index overlaps (Z :. i :. j) >= fg_overlap) (V.indexed maxIndices)
                  in if length fg_indexes > fg_rois_per_image then
                        V.fromList . take fg_rois_per_image <$> runRVar' (shuffle $ V.toList fg_indexes)
                     else
                        return fg_indexes

    -- slightly different from the orignal implemetation:
    -- a uniform sampling w/ replacement if not enough bg boxes
    let bg_rois_this_image = rois_per_image - length fg_indexes
    bg_indexes <- let bg_indexes = V.filter (\(i, j) -> Repa.index overlaps (Z :. i :. j) <  fg_overlap) (V.indexed maxIndices)
                      num_bg_indexes = length bg_indexes
                  in case compare num_bg_indexes bg_rois_this_image of
                        GT -> V.fromList . take bg_rois_this_image <$> runRVar' (shuffle $ V.toList bg_indexes)
                        LT -> V.fromList <$> runRVar' (replicateM bg_rois_this_image (randomElement bg_indexes))
                        EQ -> return bg_indexes

    let keep_indexes = V.map fst $ fg_indexes V.++ bg_indexes

        rois_keep    = V.map (rois ^%!) keep_indexes
        roi_box_keep = V.map (Repa.computeUnboxedS . Repa.extract (Z:.1) (Z:.4)) rois_keep

        gt_keep      = V.map (gt_chosen  ^%!) keep_indexes
        gt_box_keep  = V.map (Repa.computeUnboxedS . Repa.extract (Z:.0) (Z:.4)) gt_keep
        labels_keep  = V.take (length fg_indexes) (V.map (^#! 4) gt_keep) V.++ V.replicate bg_rois_this_image 0

        targets = V.zipWith (bboxTransform box_stds) roi_box_keep gt_box_keep

    -- regression is indexed by class.
    bbox_target <- UVM.replicate (rois_per_image * 4 * num_classes) (0 :: Float)
    bbox_weight <- UVM.replicate (rois_per_image * 4 * num_classes) (0 :: Float)

    -- only assign regression and weights for the foreground boxes.
    forM_ [0..length fg_indexes-1] $ \i -> do
        let lbl = floor (labels_keep ^%! i)
            tgt = targets ^%! i
        assert (lbl >= 0 && lbl < num_classes) (return ())
        let tgt_dst = UVM.slice (i * 4 * num_classes + 4 * lbl) 4 bbox_target
        forM_ [0..3] $ \i ->
            UVM.write tgt_dst i (tgt ^#! i)
        let wgh_dst = UVM.slice (i * 4 * num_classes + 4 * lbl) 4 bbox_weight
        UVM.set wgh_dst 1

    let shape = Z :. rois_per_image :. 4 * num_classes
    bbox_target <- Repa.fromUnboxed shape <$> UV.freeze bbox_target
    bbox_weight <- Repa.fromUnboxed shape <$> UV.freeze bbox_weight
    return (rois_keep, labels_keep, bbox_target, bbox_weight)

  where
    runRVar' = flip runRVar StdRandom
    num_classes = prop ^. pt_num_classes
    rois_per_image = (prop ^. pt_batch_rois) `div` (prop ^. pt_batch_images)
    fg_rois_per_image = round (prop ^. pt_fg_fraction * fromIntegral rois_per_image)
    fg_overlap = prop ^. pt_fg_overlap
    box_stds = Repa.fromListUnboxed (Z:.4) (prop ^. pt_box_stds)

overlapMatrix :: V.Vector (Array U DIM1 Float) -> V.Vector (Array U DIM1 Float) -> Array D DIM2 Float
overlapMatrix rois gt = Repa.fromFunction (Z :. width :. height) calcOvp
  where
    width  = length rois
    height = length gt

    area1 = V.map bboxArea rois
    area2 = V.map bboxArea gt

    calcOvp (Z :. ind_rois :. ind_gt) =
        case bboxIntersect (rois ^%! ind_rois) (gt ^%! ind_gt) of
           Nothing -> 0
           Just boxI -> let areaI = bboxArea boxI
                            areaU = area1 ^%! ind_rois + area2 ^%! ind_gt - areaI
                        in areaI / areaU


(^%!) :: HasCallStack => V.Vector a -> Int -> a
-- a ^%! i = a V.! i
a ^%! i = V.unsafeIndex a i


-- test_sample_rois = let
--         v1 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 0.8, 0.8, 2.2, 2.2]
--         v2 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 2.2, 2.2, 4.5, 4.5]
--         v3 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 4.2, 1, 6.5, 2.8]
--         v4 = Repa.fromListUnboxed (Z:.5::DIM1) [0, 6, 3, 7, 4]
--         rois = V.fromList [v1, v2, v3, v4]
--         g1 = Repa.fromListUnboxed (Z:.5::DIM1) [1,1,2,2,1]
--         g2 = Repa.fromListUnboxed (Z:.5::DIM1) [2,3,3,4,1]
--         g3 = Repa.fromListUnboxed (Z:.5::DIM1) [4,1,6,3,2]
--         gt_boxes = V.fromList [g1, g2, g3]
--       in sample_rois rois gt_boxes 3 6 2 0.5 [0.1, 0.1, 0.1, 0.1]



