module MXNet.NN.ModelZoo.RCNN.RCNN where

import           RIO
import           RIO.List                    (unzip3)

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor as T
import           MXNet.NN.Layer


rcnnSampler :: Int -> Int -> Int -> Float -> Float -> Int
            -> SymbolHandle -> SymbolHandle -> SymbolHandle
            -> Layer (SymbolHandle, SymbolHandle, SymbolHandle)
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
      sampler batch_index = do
          roi    <- getBatch rois batch_index
          score  <- getBatch scores batch_index
          gt_box <- getBatch gt_boxes batch_index

          -- TODO why sum up the coordinates as score?
          gt_score <- sum_ gt_box (Just [(-1)]) True
          gt_score <- addScalar 1 gt_score
          gt_score <- prim T._sign (#data := gt_score .& Nil)

          all_rois   <- concat_ 0 [roi, gt_box]
          all_scores <- concat_ 0 [score, gt_score]

          ious <- prim T.__contrib_box_iou (#lhs := all_rois
                                         .& #rhs := gt_box
                                         .& #format := #corner .& Nil)
          -- iou of the best gt box of each roi
          ious_max <- prim T._max (#data := ious .& #axis := Just [-1] .& Nil)
          -- index of the best gt box of each roi
          ious_argmax <- argmax ious (Just (-1)) False

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
          order <- prim T._argsort (#data := mask .& #is_ascend := False .& Nil)
          let max_pos = floor $ fromIntegral num_sample * fg_fraction
          -- topk
          topk  <- slice_axis order 0 0 (Just max_pos)
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
          index       <- slice_axis index 0 max_pos Nothing
          mask        <- slice_axis mask  0 max_pos Nothing
          ious_argmax <- slice_axis ious_argmax 0 max_pos Nothing
          -- class 2 ==> class 4
          class_4 <- onesLike mask >>= mulScalar 4
          cond    <- eqScalar 2 mask
          mask    <- where_ cond class_4 mask
          order   <- prim T._argsort (#data := mask .& #is_ascend := False .& Nil)

          let num_neg = num_sample - max_pos
          bottomk <- slice_axis order 0 0 (Just num_neg)
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
                     slice_axis s 0 i (Just (i + 1))


rcnnTargetGenerator :: Int -> Int -> Int
                    -> SymbolHandle
                    -> SymbolHandle
                    -> SymbolHandle
                    -> SymbolHandle
                    -> SymbolHandle
                    -> SymbolHandle
                    -> SymbolHandle
                    -> Layer (SymbolHandle, SymbolHandle, SymbolHandle, SymbolHandle)
rcnnTargetGenerator batch_size num_fg_classes max_pos samples matches anchors gt_label gt_boxes mean std = do
    -- B: batch_size, N: num_rois, M: num_gt, N_pos: max_pos, C: num_fg_classes
    --
    -- samples: (B, N), value -1 (negative), 0 (ignore), 1 (positive)
    -- matches: (B, N), value range [0, M), the best-matched gt of each roi
    -- anchors: (B, N, 4), anchor boxes, min_x, min_y, max_x, max_y
    -- gt_label: (B, M), value range [0, num_fg_classes), excluding background class
    -- gt_boxes: (B, N, 4), gt boxes, min_x, min_y, max_x, max_y
    -- mean: (4,)
    -- std:  (4,)
    --
    -- returns:
    --   cls_targets: (B, N_pos), value [0, num_classes], -1 to be ignored
    --   box_targets: (B, N_pos, C, 4)
    --   box_masks:   (B, N_pos, C, 4)
    --   mask_sel:    (B, N_pos)
    --
    labels <- reshape [0, 1, -1] gt_label
    labels <- prim T._broadcast_like (#lhs := labels .& #rhs := matches .& #lhs_axes := Just [1] .& #rhs_axes := Just [1] .& Nil)
    -- labels: (B,N,M) forall batch, roi, gt. class id
    -- fg_cls_targets: (B,N) forall batch, roi, class id of the best gt
    fg_cls_targets <- pick (#data := labels .& #index := matches .& #axis := Just 2 .& Nil)
    -- shift by 1, reserve 0 for the background class
    cls_targets <- addScalar 1 fg_cls_targets

    ign <- onesLike cls_targets >>= mulScalar (-1)
    bck <- zerosLike cls_targets
    pos <- gtScalar 0.5 samples
    neg <- ltScalar (-0.5) samples
    -- [1, num_fg_classes] for fg, 0 for background, -1 for being ignored
    cls_targets <- where_ pos cls_targets ign
    cls_targets <- where_ neg bck cls_targets

    ret <- prim T.__contrib_box_encode (#samples := samples
                                     .& #matches := matches
                                     .& #anchors := anchors
                                     .& #refs    := gt_boxes
                                     .& #means   := mean
                                     .& #stds    := std .& Nil)
    [box_targets, box_masks] <- mapM (ret `at`) ([0, 1] :: [Int])

    fg_cls_targets <- expandDims 2 fg_cls_targets
    class_ids_fg <- prim T.__arange (#start := 0 .& #stop := Just (fromIntegral num_fg_classes) .& Nil)
    class_ids_fg <- reshape [1,1,-1] class_ids_fg
    -- (B, N, C), one hot indicator for the best gt class id for each roi of each batch
    target_class_fg_onehot <- eqBroadcast fg_cls_targets class_ids_fg

    masks_sel <- slice_axis box_masks (-1) 0 (Just 1)
    masks_sel <- prim T._argsort (#data := masks_sel .& #axis := Just 1 .& #is_ascend := False .& Nil)
    masks_sel <- reshape [batch_size, -1] masks_sel
    -- mask indices of those positive ones (take at most max_pos items)
    masks_sel <- slice_axis masks_sel 1 0 (Just max_pos)

    (box_targets, box_masks, clsid_ohs) <- fmap unzip3 $ forM [0..batch_size-1] $ \i -> do
        ind      <- slice_axis masks_sel 0 i (Just (i+1)) >>= squeeze (Just [0])
        target   <- slice_axis box_targets 0 i (Just (i+1)) >>= squeeze (Just [0])
        mask     <- slice_axis box_masks 0 i (Just (i+1)) >>= squeeze (Just [0])
        clsid_oh <- slice_axis target_class_fg_onehot 0 i (Just (i+1)) >>= squeeze (Just [0])

        target   <- takeI ind target   >>= expandDims 0
        mask     <- takeI ind mask     >>= expandDims 0
        clsid_oh <- takeI ind clsid_oh >>= expandDims 0

        return (target, mask, clsid_oh)

    box_targets <- concat_ 0 (box_targets :: [SymbolHandle]) >>= expandDims 2
    box_masks   <- concat_ 0 (box_masks   :: [SymbolHandle]) >>= expandDims 2
    -- broadcast the one-hot indicator
    clsid_ohs   <- concat_ 0 (clsid_ohs:: [SymbolHandle]) >>= expandDims 3 >>= broadcastAxis [3] [4]

    box_targets <- broadcastAxis [2] [num_fg_classes] box_targets
    box_masks   <- mulBroadcast box_masks clsid_ohs
    -- return the index of positive masks because we will calculate box loss only on those items
    return (cls_targets, box_targets, box_masks, masks_sel)

