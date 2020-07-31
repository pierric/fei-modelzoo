module MXNet.NN.ModelZoo.RCNN.RCNN where

import           RIO
import           RIO.List                    (unzip3)

import           MXNet.Base
import qualified MXNet.Base.Operators.Symbol as S
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

    rois    <- stack (#data := rois    .& #axis := 0 .& Nil)
    samples <- stack (#data := samples .& #axis := 0 .& Nil)
    matches <- stack (#data := matches .& #axis := 0 .& Nil)

    return (rois, samples, matches)

  where
      sampler batch_index = do
          roi    <- getBatch rois batch_index
          score  <- getBatch scores batch_index
          gt_box <- getBatch gt_boxes batch_index

          -- TODO why sum up the coordinates as score?
          gt_score <- prim S.sum (#data := gt_box
                               .& #axis := Just [(-1)]
                               .& #keepdims := True .& Nil)
          gt_score <- addScalar 1 gt_score
          gt_score <- prim S.sign (#data := gt_score .& Nil)

          all_rois   <- concat_ 0 [roi, gt_box]
          all_scores <- concat_ 0 [score, gt_score]

          ious <- prim S._contrib_box_iou (#lhs := all_rois
                                        .& #rhs := gt_box
                                        .& #format := #corner .& Nil)
          -- iou of the best gt box of each roi
          ious_max <- prim S.max (#data := ious .& #axis := Just [-1] .& Nil)
          -- index of the best gt box of each roi
          ious_argmax <- prim S.argmax (#data := ious .& #axis := Just (-1) .& Nil)

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
          rand <- prim S._random_uniform (#low := 0 .& #high := 1
                                       .& #shape := [num_proposal + max_num_gt] .& Nil)
          rand <- prim S.slice_like (#data := rand .& #shape_like := ious_max .& Nil)
          index<- prim S.argsort    (#data := rand .& Nil)
          mask <- takeI index mask
          ious_argmax <- takeI index ious_argmax

          -- sort in order of pos, neg, ignore
          order <- prim S.argsort (#data := mask .& #is_ascend := False .& Nil)
          let max_pos = floor $ fromIntegral num_sample * fg_fraction
          -- topk
          topk  <- prim S.slice_axis (#data := order
                                   .& #axis  := 0
                                   .& #begin := 0
                                   .& #end := Just max_pos .& Nil)
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
          index       <- prim S.slice_axis (#data := index
                                         .& #axis  := 0
                                         .& #begin := max_pos
                                         .& #end := Nothing .& Nil)
          mask        <- prim S.slice_axis (#data := mask
                                         .& #axis  := 0
                                         .& #begin := max_pos
                                         .& #end := Nothing .& Nil)
          ious_argmax <- prim S.slice_axis (#data := ious_argmax
                                         .& #axis  := 0
                                         .& #begin := max_pos
                                         .& #end := Nothing .& Nil)
          -- class 2 ==> class 4
          class_4 <- onesLike mask >>= mulScalar 4
          cond    <- eqScalar 2 mask
          mask    <- where_ cond class_4 mask
          order   <- prim S.argsort (#data := mask .& #is_ascend := False .& Nil)

          let num_neg = num_sample - max_pos
          bottomk <- prim S.slice_axis (#data := order
                                     .& #axis  := 0
                                     .& #begin := 0
                                     .& #end := Just num_neg .& Nil)
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
          ind <- prim S.argsort (#data := rois_areas .& Nil)
          r <- takeI ind sampled_rois
          s <- takeI ind samples
          m <- takeI ind matches
          return (r, s, m)

      getBatch s i = squeeze (Just [0]) =<<
                     prim S.slice_axis (#data  := s
                               .& #axis  := 0
                               .& #begin := i
                               .& #end   := Just (i + 1) .& Nil)


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
    labels <- prim S.broadcast_like (#lhs := labels .& #rhs := matches .& #lhs_axes := Just [1] .& #rhs_axes := Just [1] .& Nil)
    -- labels: (B,N,M) forall batch, roi, gt. class id
    -- fg_cls_targets: (B,N) forall batch, roi, class id of the best gt
    fg_cls_targets <- prim S.pick (#data := labels .& #index := matches .& #axis := Just 2 .& Nil)
    -- shift by 1, reserve 0 for the background class
    cls_targets <- addScalar 1 fg_cls_targets

    ign <- onesLike cls_targets >>= mulScalar (-1)
    bck <- zerosLike cls_targets
    pos <- gtScalar 0.5 samples
    neg <- ltScalar (-0.5) samples
    -- [1, num_fg_classes] for fg, 0 for background, -1 for being ignored
    cls_targets <- where_ pos cls_targets ign
    cls_targets <- where_ neg bck cls_targets

    ret <- prim S._contrib_box_encode (#samples := samples
                                    .& #matches := matches
                                    .& #anchors := anchors
                                    .& #refs    := gt_boxes
                                    .& #means   := mean
                                    .& #stds    := std .& Nil)
    [box_targets, box_masks] <- mapM (ret `at`) ([0, 1] :: [Int])

    fg_cls_targets <- expandDims 2 fg_cls_targets
    class_ids_fg <- prim S._arange (#start := 0 .& #stop := Just (fromIntegral num_fg_classes) .& Nil)
    class_ids_fg <- reshape [1,1,-1] class_ids_fg
    -- (B, N, C), one hot indicator for the best gt class id for each roi of each batch
    target_class_fg_onehot <- eqBroadcast fg_cls_targets class_ids_fg

    masks_sel <- prim S.slice_axis (#data := box_masks .& #axis := -1 .& #begin := 0 .& #end := Just 1 .& Nil)
    masks_sel <- prim S.argsort (#data := masks_sel .& #axis := Just 1 .& #is_ascend := False .& Nil)
    masks_sel <- reshape [batch_size, -1] masks_sel
    -- mask indices of those positive ones (take at most max_pos items)
    masks_sel <- prim S.slice_axis (#data := masks_sel .& #axis := 1 .& #begin := 0 .& #end := Just max_pos .& Nil)

    (box_targets, box_masks, clsid_ohs) <- fmap unzip3 $ forM [0..batch_size-1] $ \i -> do
        ind    <- prim S.slice_axis (#data := masks_sel .& #axis := 0 .& #begin := i .& #end := Just (i+1) .& Nil)
                    >>= squeeze (Just [0])

        target <- prim S.slice_axis (#data := box_targets .& #axis := 0 .& #begin := i .& #end := Just (i+1) .& Nil)
                    >>= squeeze (Just [0])

        mask   <- prim S.slice_axis (#data := box_masks   .& #axis := 0 .& #begin := i .& #end := Just (i+1) .& Nil)
                    >>= squeeze (Just [0])

        clsid_oh <- prim S.slice_axis (#data := target_class_fg_onehot .& #axis := 0 .& #begin := i .& #end := Just (i+1) .& Nil)
                    >>= squeeze (Just [0])

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

