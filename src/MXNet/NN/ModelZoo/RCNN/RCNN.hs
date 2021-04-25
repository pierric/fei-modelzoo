{-# LANGUAGE ExplicitForAll      #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.ModelZoo.RCNN.RCNN where

import           RIO
import           RIO.List                    (unzip, unzip3, unzip4, zip4)

import           Fei.Einops
import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor as T
import           MXNet.NN.Layer


rcnnSampler :: forall a . NumericDType a
            => Int -> Int -> Int -> Double -> Double -> Int
            -> Symbol a -> Symbol a -> Symbol a
            -> Layer (Symbol a, Symbol a, Symbol a)
rcnnSampler batch_size num_proposal num_sample fg_overlap fg_fraction max_num_gt
            rois scores gt_boxes = do
    -- B: batch_size, N: num_proposal (post-topk), S: num_sample (rcnn_batch_rois)
    -- rois:     (B,N,4), min_x, min_y, max_x, max_y
    -- scores:   (B,N,1), value range [0,1], -1 for being ignored
    -- gt_boxes: (B,M,4), min_x, min_y, max_x, max_y
    -- return:
    --   rois:   (B,S,4)
    --   samples:(B,S), value -1 (negative), 0 (ignore), 1 (positive)
    --   matches:(B,S), value [0, M)
    (rois, samples, matches) <- unzip3 <$> mapM sampler [0..batch_size-1]

    rois    <- stack 0 rois
    samples <- stack 0 samples
    matches <- stack 0 matches

    return (rois, samples, matches)

  where
      sampler batch_index = unique (tshow batch_index) $ do
          roi    <- getBatch rois batch_index
          score  <- getBatch scores batch_index
          gt_box <- getBatch gt_boxes batch_index

          -- why sum up the coordinates as score?
          -- because of padding gt are coded as all -1
          gt_score <- addScalar 1 =<< sum_ gt_box (Just [(-1)]) True
          gt_score <- prim T._sign (#data := gt_score .& Nil)

          -- all_rois   (N+M, 4)
          -- all_scores (N+M,)
          all_rois   <- concat_ 0 [roi, gt_box]
          all_scores <- concat_ 0 [score, gt_score] >>= squeeze (Just [-1])

          ious <- prim T.__contrib_box_iou (#lhs := all_rois
                                         .& #rhs := gt_box
                                         .& #format := #corner .& Nil)
          -- iou of the best gt box of each roi
          ious_max <- prim T._max (#data := ious .& #axis := Just [-1] .& Nil)
          -- index of the best gt box of each roi
          ious_argmax <- argmax ious (Just (-1)) False >>= castToNum @(DTypeName a)

          class_0 <- zerosLike ious_max
          class_2 <- onesLike  ious_max >>= mulScalar 2
          class_3 <- onesLike  ious_max >>= mulScalar 3

          ignore_indices <- ltScalar 0 all_scores
          pos_indices    <- gtScalar fg_overlap ious_max

          -- mask (mark the class of each roi)
          -- score == -1 ==> ignore (class 0)
          -- iou <= fg_overlap ==> neg sample (class 2)
          -- iou >  fg_overlap ==> pos sample (class 3)
          mask <- where_ ignore_indices class_0 class_2
          mask <- where_ pos_indices    class_3 mask

          -- shuffle mask and ious_argmax
          rand <- prim T.__random_uniform (#low := 0 .& #high := 1
                                        .& #shape := [num_proposal + max_num_gt] .& Nil)
          rand <- prim T._slice_like (#data := rand .& #shape_like := ious_max .& Nil)
          index<- prim T._argsort    (#data := rand .& Nil)
          mask <- takeI index mask
          ious_argmax <- takeI index ious_argmax

          -- sort in order of pos, neg, ignore
          let max_pos = floor $ fromIntegral num_sample * fg_fraction
          topk <- prim T._topk (#data := mask .& #k := max_pos .& #is_ascend := False .& Nil)
          topk_indices <- takeI topk index
          topk_samples <- takeI topk mask
          topk_matches <- takeI topk ious_argmax

          -- sample the positive class
          pos_class <- onesLike topk_samples
          neg_class <- onesLike topk_samples >>= mulScalar (-1)
          -- class 3 ==> label 1
          -- class 2 ==> label -1
          -- class 0 ==> label 0
          cond <- eqScalar 3 topk_samples
          topk_samples <- where_ cond pos_class topk_samples
          cond <- eqScalar 2 topk_samples
          topk_samples <- where_ cond neg_class topk_samples

          -- sample the negative class
          index       <- sliceAxis index 0 max_pos Nothing
          mask        <- sliceAxis mask  0 max_pos Nothing
          ious_argmax <- sliceAxis ious_argmax 0 max_pos Nothing
          -- class 2 ==> class 4
          class_4 <- onesLike mask >>= mulScalar 4
          cond    <- eqScalar 2 mask
          mask    <- where_ cond class_4 mask

          let num_neg = num_sample - max_pos
          bottomk <- prim T._topk (#data := mask .& #k := num_neg .& #is_ascend := False .& Nil)
          bottomk_indices <- takeI bottomk index
          bottomk_samples <- takeI bottomk mask
          bottomk_matches <- takeI bottomk ious_argmax

          -- class 4 ==> label -1
          -- class 3 ==> label 1
          -- class 0 ==> label 0
          cond <- eqScalar 3 bottomk_samples
          pos_class <- onesLike bottomk_samples
          bottomk_samples <- where_ cond pos_class bottomk_samples
          cond <- eqScalar 4 bottomk_samples
          neg_class <- onesLike bottomk_samples >>= mulScalar (-1)
          bottomk_samples <- where_ cond neg_class bottomk_samples

          -- concat
          indices <- concat_ 0 [topk_indices, bottomk_indices]
          samples <- concat_ 0 [topk_samples, bottomk_samples]
          matches <- concat_ 0 [topk_matches, bottomk_matches]

          sampled_rois <- takeI indices all_rois
          [x1, y1, x2, y2] <- splitBySections 4 (-1) True sampled_rois
          rois_areas <- join $ liftM2 mul_ (sub_ x2 x1) (sub_ y2 y1)
          ind <- prim T._argsort (#data := rois_areas .& Nil)
          r <- takeI ind sampled_rois
          s <- takeI ind samples
          m <- takeI ind matches
          return (r, s, m)

      getBatch s i = squeeze (Just [0]) =<<
                     sliceAxis s 0 i (Just (i + 1))


bboxTargetGenerator :: forall a . NumericDType a
                    => Int -> Int -> Int
                    -> Symbol a
                    -> Symbol a
                    -> Symbol a
                    -> Symbol a
                    -> Symbol a
                    -> Symbol a
                    -> Symbol a
                    -> Layer (Symbol a, Symbol a, Symbol a, Symbol a)
bboxTargetGenerator batch_size num_fg_classes max_pos samples matches anchors gt_label gt_boxes means stds = do
    -- B: batch_size, N: num_rois, M: num_gt, N_pos: max_pos, C: num_fg_classes
    --
    -- samples: (B, N), value -1 (negative), 0 (ignore), 1 (positive)
    -- matches: (B, N), value range [0, M), the best-matched gt of each roi
    -- anchors: (B, N, 4), anchor boxes, min_x, min_y, max_x, max_y
    -- gt_label: (B, M), value range [0, num_fg_classes), excluding background class
    -- gt_boxes: (B, N, 4), gt boxes, min_x, min_y, max_x, max_y
    --
    -- returns:
    --   cls_targets: (B, N_pos), value [0, num_classes], -1 to be ignored
    --   box_targets: (B, N_pos, C, 4)
    --   box_masks:   (B, N_pos, C, 4)
    --   mask_sel:    (B, N_pos)
    --
    (fg_cls_targets, cls_targets) <- multiClassEncode gt_label samples matches

    [box_targets, box_masks] <- named "BoxEncoder" $
        primMulti T.__contrib_box_encode (#samples := samples
                                       .& #matches := matches
                                       .& #anchors := anchors
                                       .& #refs    := gt_boxes
                                       .& #means   := means
                                       .& #stds    := stds .& Nil)

    target_class_fg_onehot <- sequential "FG_1hot" $ do
        -- fg_cls_targets <- expandDims 2 fg_cls_targets
        fg_cls_targets <- rearrange fg_cls_targets "b (n c) -> b n c" [#c .== 1]
        class_ids_fg <- arange Proxy 0 (Just $ fromIntegral num_fg_classes) Nothing
        class_ids_fg <- rearrange class_ids_fg "(b n c) -> b n c" [#b .== 1, #n .== 1]
        -- (B, N, C), one hot indicator for the best gt class id for each roi of each batch
        eq_ fg_cls_targets class_ids_fg >>= castToNum @(DTypeName a)

    masks_sel <- sequential "MaskSel" $ do
        masks_sel <- sliceAxis box_masks (-1) 0 (Just 1)
        masks_sel <- prim T._argsort (#data := masks_sel .& #axis := Just 1 .& #is_ascend := False .& Nil)
        masks_sel <- rearrange masks_sel "b n k -> b (n k)" [#k .== 1]
        -- mask indices of those positive ones (take at most max_pos items)
        sliceAxis masks_sel 1 0 (Just max_pos)

    (box_targets, box_masks, clsid_ohs) <- sequential "PerBatch" $
        fmap unzip3 $ forM [0..batch_size-1] $ \i -> do
            ind      <- sliceAxis masks_sel 0 i (Just (i+1)) >>= squeeze (Just [0])
            target   <- sliceAxis box_targets 0 i (Just (i+1)) >>= squeeze (Just [0])
            mask     <- sliceAxis box_masks 0 i (Just (i+1)) >>= squeeze (Just [0])
            clsid_oh <- sliceAxis target_class_fg_onehot 0 i (Just (i+1)) >>= squeeze (Just [0])

            target   <- takeI ind target   >>= expandDims 0
            mask     <- takeI ind mask     >>= expandDims 0
            clsid_oh <- takeI ind clsid_oh >>= expandDims 0

            return (target, mask, clsid_oh)

    sequential "Merge" $ do
        box_targets <- concat_ 0 (box_targets :: [Symbol _]) >>= expandDims 2
        box_masks   <- concat_ 0 (box_masks   :: [Symbol _]) >>= expandDims 2
        -- broadcast the one-hot indicator
        clsid_ohs   <- concat_ 0 (clsid_ohs   :: [Symbol _]) >>= expandDims 3 >>= broadcastAxis [3] [4]

        box_targets <- broadcastAxis [2] [num_fg_classes] box_targets
        box_masks   <- mul_ box_masks clsid_ohs
        -- return the index of positive masks because we will calculate box loss only on those items
        return (cls_targets, box_targets, box_masks, masks_sel)


maskTargetGenerator :: forall a . NumericDType a
                    => Int -> Int -> Int
                    -> Symbol a
                    -> Symbol a
                    -> Symbol a
                    -> Symbol a
                    -> Layer (Symbol a, Symbol a)
maskTargetGenerator batch_size num_fg_classes mask_size gt_masks rois matches cls_targets = do
    -- rois: (B, N, 4), input proposals
    -- gt_masks: (B, M, H, W), input masks of full image size
    -- matches: (B, N), value [0, M), index to gt_label and gt_box.
    -- cls_targets: (B, N), value [0, num_class), excluding background class.
    --
    -- returns:
    --   mask_targets: (B, N, C, MS, MS), sampled masks.
    --   box_weight:   (B, N, C, MS, MS), only foreground class has nonzero weight.

    -- gt_masks (B, M, H, W) -> (B, M, 1, H, W) -> B * (M, 1, H, W)
    gt_masks <- rearrange gt_masks "b (m k) h w -> b m k h w" [#k .== 1]
    gt_masks <- splitBySections batch_size 0 True gt_masks

    -- rois (B, N, 4) -> B * (N, 4)
    rois <- splitBySections batch_size 0 True rois

    -- remove all -1 (setting to 0), (B, N) -> B * (N,)
    matches <- prim T._relu (#data := matches .& Nil)
    matches <- splitBySections batch_size 0 True matches

    -- (B, N) -> B * (N,)
    cls_targets <- splitBySections batch_size 0 True cls_targets

    class_ids_fg <- arange Proxy 0 (Just $ fromIntegral num_fg_classes) Nothing
    -- (C,) -> (1, C)
    class_ids_fg <- reshape [1, -1] class_ids_fg

    masks <- unique "make" $ mapM (make_target class_ids_fg) $ zip4 rois gt_masks matches cls_targets
    let (mask_targets, mask_weights) = unzip masks

    mask_targets <- stack 0 mask_targets
    mask_weights <- stack 0 mask_weights
    return (mask_targets, mask_weights)

    where
        make_target cids (roi, gt, match, cls_targets) = do
            -- gt: (M, 1, H, W)
            -- padded_rois: (N, 5), along the dim-2, gt_index (1) and rois_box (4)
            match <- reshape [-1, 1] match
            padded_rois <- concat_ (-1) [match, roi]
            -- (N, 1, mask_size, mask_size)
            pooled_mask <- prim T.__contrib_ROIAlign (#data := gt
                                                   .& #rois := padded_rois
                                                   .& #pooled_size := [mask_size, mask_size]
                                                   .& #spatial_scale := 1
                                                   .& #sample_ratio := 2 .& Nil)
            -- (N,) -> (N,1)
            cls_targets <- expandDims 1 cls_targets
            -- (N,1) (1,C) -> (N,C)
            cid_onehot <- eq_ cls_targets cids >>= castToNum @(DTypeName a)

            cid_onehot <- rearrange cid_onehot "n (c w h) -> n c w h" [#w .== 1, #h .== 1]
            -- (N, C, mask_size, mask_size)
            mask_weights <- prim T._broadcast_like
                                (#lhs := cid_onehot
                              .& #rhs := pooled_mask
                              .& #lhs_axes := Just [2, 3]
                              .& #rhs_axes := Just [2, 3] .& Nil)
            -- (N, 1, mask_size, mask_size) -> (N, C, mask_size, mask_size)
            mask_targets <- broadcastAxis [1] [num_fg_classes] pooled_mask
            return (mask_targets, mask_weights)


multiClassEncode gt_label samples matches = sequential "MultiClassEncoder" $ do
    -- gt_label: (B, M), value range [0, num_fg_classes), excluding background class
    -- samples:  (B, N), value -1 (negative), 0 (ignore), 1 (positive)
    -- matches:  (B, N), value range [0, M), the best-matched gt of each roi
    labels <- rearrange gt_label "b (k m) -> b k m" [#k .== 1]
    labels <- prim T._broadcast_like (#lhs := labels .& #rhs := matches .& #lhs_axes := Just [1] .& #rhs_axes := Just [1] .& Nil)
    -- labels: (B,N,M) forall batch, roi, gt. class id
    -- fg_cls_targets: (B,N) forall batch, roi, class id of the best gt
    fg_cls_targets <- pick (Just 2) matches labels
    -- shift by 1, reserve 0 for the background class
    cls_targets <- addScalar 1 fg_cls_targets

    ign <- onesLike cls_targets >>= mulScalar (-1)
    bck <- zerosLike cls_targets
    pos <- gtScalar 0.5 samples
    neg <- ltScalar (-0.5) samples
    -- [1, num_fg_classes] for fg, 0 for background, -1 for being ignored
    cls_targets <- where_ pos cls_targets ign
    cls_targets <- where_ neg bck cls_targets

    return (fg_cls_targets, cls_targets)


multiClassDecodeWithClsId :: (DType a, InEnum (DTypeName a) NumericDTypes)
                          => Int -> Int -> Double -> Symbol a
                          -> Layer (Symbol a, Symbol a)
multiClassDecodeWithClsId num_classes axis threshold prediction = do
    -- num_classes: number of classes, including the background class
    -- axis: the axis where the class prediction is
    -- threshold: prediction under the threshold will be masked
    -- prediction: (B, N, num_classes), predicated probablities
    -- return:
    --      cls_ids: (B, N, num_classes-1)
    --      pred_fg: (B, N, num_classes-1)
    let num_fg_classes = num_classes - 1
    pred_fg <- sliceAxis prediction axis 1 Nothing

    -- make a (B, N, num_fg_classes) of values [0..num_fg_classes-1]
    zero <- zerosLike =<< sliceAxis prediction axis 0 (Just 1)
    cls_ids <- reshape [1, 1, num_fg_classes]
                =<< arange Proxy 0 (Just $ fromIntegral num_fg_classes) Nothing
    cls_ids <- add_ zero cls_ids

    mask <- gtScalar threshold pred_fg
    ign1 <- zerosLike pred_fg
    ign2 <- mulScalar (-1) =<< onesLike cls_ids
    pred_fg <- where_ mask pred_fg ign1
    cls_ids <- where_ mask cls_ids ign2

    return (cls_ids, pred_fg)
