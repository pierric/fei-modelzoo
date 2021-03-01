module MXNet.NN.ModelZoo.RCNN.FasterRCNN where

import           RIO
import           RIO.List                    (unzip3, zip3, zip4)
import           RIO.List.Partial            (head, last)
import qualified RIO.NonEmpty                as NE (toList)

import           MXNet.Base
import           MXNet.Base.Operators.Tensor (_Custom, _MakeLoss, __arange,
                                              __contrib_AdaptiveAvgPooling2D,
                                              __contrib_ROIAlign,
                                              __contrib_box_decode,
                                              __contrib_box_nms, __zeros,
                                              _add_n, _clip, _repeat, _sigmoid,
                                              _smooth_l1, _transpose)
import           MXNet.NN.Layer
import           MXNet.NN.ModelZoo.RCNN.FPN
import           MXNet.NN.ModelZoo.RCNN.RCNN
import qualified MXNet.NN.ModelZoo.Resnet    as Resnet
import qualified MXNet.NN.ModelZoo.VGG       as VGG

data Backbone = VGG16 | RESNET50 | RESNET101 | RESNET50FPN deriving
    ( Eq
    , Read
    , Show
    )

data RcnnConfiguration
  = RcnnConfigurationTrain
      { backbone             :: Backbone
      , batch_size           :: Int
      , feature_strides      :: [Int]
      , pretrained_weights   :: String
      , bbox_reg_std         :: (Float, Float, Float, Float)
      , rpn_anchor_scales    :: [Int]
      , rpn_anchor_ratios    :: [Float]
      , rpn_anchor_base_size :: Int
      , rpn_pre_topk         :: Int
      , rpn_post_topk        :: Int
      , rpn_nms_thresh       :: Float
      , rpn_min_size         :: Int
      , rpn_batch_rois       :: Int
      , rpn_fg_fraction      :: Double
      , rpn_fg_overlap       :: Double
      , rpn_bg_overlap       :: Double
      , rpn_allowd_border    :: Int
      , rcnn_num_classes     :: Int
      , rcnn_pooled_size     :: Int
      , rcnn_batch_rois      :: Int
      , rcnn_fg_fraction     :: Double
      , rcnn_fg_overlap      :: Double
      , rcnn_max_num_gt      :: Int
      }
  | RcnnConfigurationInference
      { backbone             :: Backbone
      , batch_size           :: Int
      , feature_strides      :: [Int]
      , checkpoint           :: String
      , bbox_reg_std         :: (Float, Float, Float, Float)
      , rpn_anchor_scales    :: [Int]
      , rpn_anchor_ratios    :: [Float]
      , rpn_anchor_base_size :: Int
      , rpn_pre_topk         :: Int
      , rpn_post_topk        :: Int
      , rpn_nms_thresh       :: Float
      , rpn_min_size         :: Int
      , rcnn_num_classes     :: Int
      , rcnn_pooled_size     :: Int
      , rcnn_batch_rois      :: Int
      , rcnn_force_nms       :: Bool
      , rcnn_nms_thresh      :: Float
      , rcnn_topk            :: Int
      }
  deriving (Show)

data FasterRCNN
  = FasterRCNN
      { _rpn_loss         :: (SymbolHandle, SymbolHandle, SymbolHandle)
      , _box_loss         :: (SymbolHandle, SymbolHandle)
      , _cls_targets      :: SymbolHandle
      , _roi_boxes        :: SymbolHandle
      , _gt_matches       :: SymbolHandle
      , _positive_indices :: SymbolHandle
      , _top_feature      :: SymbolHandle
      }
  | FasterRCNNInferenceOnly
      { _top_feature :: SymbolHandle
      , _cls_ids     :: SymbolHandle
      , _scores      :: SymbolHandle
      , _boxes       :: SymbolHandle
      }

stageList :: Backbone -> [Int]
stageList RESNET50FPN = [2..5]
stageList _           = [3]

resnet50Args = (#num_stages := 4
             .& #filter_list := [64, 256, 512, 1024, 2048]
             .& #units := [3,4,6,3]
             .& #bottle_neck := True
             .& #workspace := 256
             .& Nil)

resnet101Args = (#num_stages := 4
             .& #filter_list := [64, 256, 512, 1024, 2048]
             .& #units := [3,4,23,3]
             .& #bottle_neck := True
             .& #workspace := 256
             .& Nil)

features1 :: Backbone -> SymbolHandle -> Layer (NonEmpty SymbolHandle)
features1 VGG16     dat = fmap (:| []) $ VGG.getFeature dat [2, 2, 3, 3, 3] [64, 128, 256, 512, 512] False False
features1 RESNET50  dat = fmap (:| []) $ Resnet.getFeature dat resnet50Args
features1 RESNET101 dat = fmap (:| []) $ Resnet.getFeature dat resnet101Args
features1 RESNET50FPN dat = do
    sym <- Resnet.getFeature dat resnet50Args
    sym <- Resnet.getTopFeature sym resnet50Args
    fpnFeatureExpander sym
        [ ("features.5.2.plus_output", 256)
        , ("features.6.3.plus_output", 256)
        , ("features.7.5.plus_output", 256)
        , ("features.8.2.plus_output", 256) ]

features2 :: Backbone -> SymbolHandle -> Layer SymbolHandle
features2 VGG16       dat = VGG.getTopFeature dat
features2 RESNET50    dat = Resnet.getTopFeature dat resnet50Args
features2 RESNET101   dat = Resnet.getTopFeature dat resnet101Args
features2 RESNET50FPN dat = return dat

rpn :: RcnnConfiguration
    -> NonEmpty SymbolHandle -> SymbolHandle
    -> Layer (SymbolHandle, SymbolHandle, SymbolHandle, SymbolHandle)
rpn conf convFeats imInfo = unique "rpn" $ do
    conv3x3_feat <- named "rpn_conv_3x3" $
                    convolutionShared (#kernel := [3,3]
                                    .& #pad := [1,1]
                                    .& #num_filter := 512 .& Nil)
    conv1x1_cls  <- named "rpn_cls_score" $
                    convolutionShared (#kernel := [1,1]
                                    .& #pad := [0,0]
                                    .& #num_filter := num_variation .& Nil)
    conv1x1_reg  <- named "rpn_bbox_pred" $
                    convolutionShared (#kernel := [1,1]
                                    .& #pad := [0,0]
                                    .& #num_filter := 4 * num_variation .& Nil)
    layers <- zipWithM (rpn_head conv3x3_feat conv1x1_cls conv1x1_reg)
                       (NE.toList convFeats)
                       (feature_strides conf)

    let (rpn_pres, rpn_raw_scores, rpn_raw_boxregs) = unzip3 layers
    -- concat the list of RPN ROI boxes, in which boxes are decoded, and suppressed items are -1s
    -- result shape: (batch_size, Σ(feat_H_i*feat_W_i*num_variation), 5)
    rpn_pres        <- concat_ 1 rpn_pres
    -- concat the list of RPN raw scores of all predictions,
    -- result shape: (batch_size, Σ(feat_H_i*feat_W_i*num_variation), 1)
    rpn_raw_scores  <- concat_ 1 rpn_raw_scores
    -- concat the list of RPN raw box regression of all predictions,
    -- result shape: (batch_size, Σ(feat_H_i*feat_W_i*num_variation), 4)
    rpn_raw_boxregs <- concat_ 1 rpn_raw_boxregs

    -- non-maximum suppress the rois, and split the score and box part
    (rpn_roi_scores, rpn_roi_boxes) <- nms rpn_pres

    return (rpn_roi_scores, rpn_roi_boxes, rpn_raw_scores, rpn_raw_boxregs)

    where
        num_variation = length (rpn_anchor_scales conf) * length (rpn_anchor_ratios conf)
        rpn_head conv3x3_feat conv1x1_cls conv1x1_reg feat stride = do
            x <- conv3x3_feat feat
            x <- activation (#data := x
                          .& #act_type := #relu .& Nil)

            anchors <- prim _Custom (#op_type := "anchor_generator"
                                  .& #data    := [x]
                                  .& #stride     :≅ stride
                                  .& #scales     :≅ rpn_anchor_scales conf
                                  .& #ratios     :≅ rpn_anchor_ratios conf
                                  .& #base_size  :≅ rpn_anchor_base_size conf
                                  .& #alloc_size :≅ ((128, 128) :: (Int, Int))
                                  .& Nil)

            rpn_raw_score <- conv1x1_cls x
            -- (batch_size, num_variation, H, W) ==> (batch_size, H, W, num_variation)
            rpn_raw_score <- prim _transpose (#data := rpn_raw_score  .& #axes := [0, 2, 3, 1] .& Nil)
            -- (batch_size, H, W, num_variation) ==> (batch_size, H*W*num_variation, 1)
            rpn_raw_score <- reshape [0, -1, 1] rpn_raw_score

            rpn_cls_score <- blockGrad =<< prim _sigmoid (#data := rpn_raw_score .& Nil)
            rpn_cls_score <- reshape [0, -1, 1] rpn_cls_score

            rpn_raw_boxreg <- conv1x1_reg x
            -- (batch_size, num_variation * 4, H, W) ==> (batch_size, H, W, num_variation * 4)
            rpn_raw_boxreg <- prim _transpose (#data := rpn_raw_boxreg .& #axes := [0, 2, 3, 1] .& Nil)
            -- (batch_size, H, W, num_variation * 4) ==> (batch_size, H*W*num_variation, 4)
            rpn_raw_boxreg <- reshape [0, -1, 4] rpn_raw_boxreg

            rpn_pre <- region_proposer (fromIntegral $ rpn_min_size conf)
                                       anchors
                                       rpn_raw_boxreg
                                       rpn_cls_score
                                       (1, 1, 1, 1)

            return (rpn_pre, rpn_raw_score, rpn_raw_boxreg)

        region_proposer min_size anchors boxregs scores stds = do
            let (std0, std1, std2, std3) = stds
            rois <- prim __contrib_box_decode (#data := boxregs
                                            .& #anchors := anchors
                                            .& #format := #corner
                                            .& #std0 := std0
                                            .& #std1 := std1
                                            .& #std2 := std2
                                            .& #std3 := std3
                                            .& Nil)
            (xmin, ymin, xmax, ymax) <- bbox_clip_to_image rois imInfo
            width   <- sub_ xmax xmin >>= addScalar 1
            height  <- sub_ ymax ymin >>= addScalar 1
            invalid <- ltScalar min_size width  >>= \c1 ->
                       ltScalar min_size height >>= \c2 ->
                            or_ c1 c2
            mask    <- onesLike invalid >>= mulScalar (-1)
            scores  <- where_ invalid mask scores
            invalid <- broadcastAxis [2] [4] invalid
            mask    <- onesLike invalid >>= mulScalar (-1)
            rois    <- concat_ (-1) [xmin, ymin, xmax, ymax]
            rois    <- where_ invalid mask rois
            blockGrad =<< named "proposals" (concat_ (-1) [scores, rois])

        nms rpn_pre = do
            -- rpn_pre shape: (batch_size, num_anchors, 5)
            -- in dim 3: [score, xmin, ymin, xmax, ymax]
            tmp <- prim __contrib_box_nms (#data := rpn_pre
                                        .& #overlap_thresh := rpn_nms_thresh conf
                                        .& #topk := rpn_pre_topk conf
                                        .& #coord_start := 1
                                        .& #score_index := 0
                                        .& #id_index    := (-1)
                                        .& #force_suppress := True .& Nil)
            tmp <- sliceAxis tmp 1 0 (Just $ rpn_post_topk conf)
            rpn_roi_scores <- blockGrad =<<
                              sliceAxis tmp (-1) 0 (Just 1)
            rpn_roi_boxes  <- blockGrad =<<
                              sliceAxis tmp (-1) 1 Nothing
            return (rpn_roi_scores, rpn_roi_boxes)

        bbox_clip_to_image rois info = do
            -- rois: (B, N, 4)
            -- info: (B, 3)
            -- return: (B,N), (B,N), (B,N), (B,N)
            [xmin, ymin, xmax, ymax] <- splitBySections 4 (-1) False rois
            [height, width, _]       <- splitBySections 3 (-1) False info
            height <- expandDims (-1) height
            width  <- expandDims (-1) width
            w_ub <- subScalar 1 width
            h_ub <- subScalar 1 height
            z <- zerosLike xmin
            w <- onesLike xmin >>= mulBroadcast w_ub
            h <- onesLike xmin >>= mulBroadcast h_ub
            cond <- ltScalar 0 xmin
            xmin <- where_ cond z xmin
            cond <- ltScalar 0 ymin
            ymin <- where_ cond z ymin
            cond <- gtBroadcast xmax w_ub
            xmax <- where_ cond w xmax
            cond <- gtBroadcast ymax h_ub
            ymax <- where_ cond h ymax
            return (xmin, ymin, xmax, ymax)

alignROIs :: NonEmpty SymbolHandle -> SymbolHandle -> [Int] -> Int -> [Int] -> Layer _
alignROIs features rois stage_indices roi_pooled_size strides = do
    -- rois: (N, 5), batch_index, min_x, min_y, max_x, max_y
    let min_stage = head stage_indices
        max_stage = last stage_indices
    [_, xmin, ymin, xmax, ymax] <- splitBySections 5 (-1) False rois
    w <- xmax `sub_` xmin >>= addScalar 1
    h <- ymax `sub_` ymin >>= addScalar 1
    -- heuristic to compute the stage where each rois box fits
    -- bigger box in higher stage
    -- smaller box in lower stage
    roi_level_raw <- w `mul_` h >>=
                     sqrt_ >>=
                     divScalar 224 >>=
                     addScalar 1e-6 >>=
                     log2_ >>=
                     addScalar 4 >>=
                     floor_
    roi_level <- prim _clip (#data := roi_level_raw
                          .& #a_min := fromIntegral min_stage
                          .& #a_max := fromIntegral max_stage .& Nil)
                 >>= squeeze Nothing
    let align (lvl, feat, stride) = do
            cond <- eqScalar (fromIntegral lvl) roi_level
            omit <- onesLike rois >>= mulScalar (-1)
            masked <- where_ cond rois omit
            prim __contrib_ROIAlign (#data := feat
                                 .& #rois := masked
                                 .& #pooled_size := [roi_pooled_size, roi_pooled_size]
                                 .& #spatial_scale := 1 / fromIntegral stride
                                 .& #sample_ratio := 2 .& Nil)
    features <- mapM align $ zip3 [max_stage,max_stage-1..min_stage] (NE.toList features) strides
    prim _add_n (#args := features .& Nil)


graphT :: RcnnConfiguration -> Layer (FasterRCNN, SymbolHandle)
graphT conf@RcnnConfigurationTrain{..} =  do
    -- dat: (B, image_height, image_width)
    dat <- variable "data"
    -- imInfo: (B, 3,)
    imInfo <- variable "im_info"
    -- gt_boxes: (B, M, 5), the last dim: min_x, min_y, max_x, max_y, class_id (background class: 0)
    gt_boxes <- variable "gt_boxes"
    rpn_cls_targets <- variable "rpn_cls_targets"
    rpn_box_targets <- variable "rpn_box_targets"
    rpn_box_masks   <- variable "rpn_box_masks"

    gt_labels <- unique' $ sliceAxis gt_boxes (-1) 4 Nothing
    gt_boxes  <- unique' $ sliceAxis gt_boxes (-1) 0 (Just 4)

    let (std0, std1, std2, std3) = bbox_reg_std
    bbox_reg_mean <- named "bbox_reg_mean" $ prim __zeros (#shape := [4] .& Nil)
    bbox_reg_std  <- named "bbox_reg_std"  $ constant [4] [std0, std1, std2, std3]

    sequential "features" $ do
        feats <- features1 backbone dat

        (rois_scores, roi_boxes, rpn_raw_scores, rpn_raw_boxregs) <- rpn conf feats imInfo

        -- total number of ROIs in a batch
        -- batch_size: number of images
        -- rcnn_batch_rois: number of rois per image
        (feat_aligned, roi_boxes, samples, matches) <- unique "rcnn" $ do
            (rois_boxes, samples, matches) <- rcnnSampler batch_size
                                                    rpn_post_topk
                                                    rcnn_batch_rois
                                                    rcnn_fg_overlap
                                                    rcnn_fg_fraction
                                                    rcnn_max_num_gt
                                                    roi_boxes
                                                    rois_scores
                                                    gt_boxes

            roi_batchid <- prim __arange (#start := 0 .& #stop := Just (fromIntegral batch_size) .& Nil)
            roi_batchid <- prim _repeat  (#data := roi_batchid .& #repeats := rcnn_batch_rois .& Nil)
            roi_batchid <- reshape [-1, 1] roi_batchid
            -- rois: (B * rcnn_batch_rois, 4)
            rois <- reshape [-1, 4] rois_boxes
            rois <- concat_ 1 [roi_batchid, rois] >>= blockGrad
            feat <- alignROIs feats rois (stageList backbone) rcnn_pooled_size feature_strides
            return (feat, rois_boxes, samples, matches)

        -- feat_aligned: (batch_size * rcnn_batch_rois, num_channels, feature_height, feature_width)
        -- TODO num_channels is set to rcnn_batch_rois, it is coincidance or on purpose?
        -- Apply the remaining feature extraction layers
        top_feat <- features2 backbone feat_aligned

        unique "rcnn" $ do
            (cls_targets, bbox_targets, bbox_masks, positive_indices) <-
                bboxTargetGenerator batch_size
                                    (rcnn_num_classes-1)
                                    (floor $ rcnn_fg_fraction * fromIntegral rcnn_batch_rois)
                                    samples
                                    matches
                                    roi_boxes
                                    gt_labels
                                    gt_boxes
                                    bbox_reg_mean
                                    bbox_reg_std

            -- sigmoid + binary-cross-entropy
            -- rpn_raw_scores: (B, num_rois, 1)
            -- rpn_cls_targets: (B, num_rois, 1)
            rpn_cls_prob <- prim _sigmoid (#data := rpn_raw_scores .& Nil)
            sample_mask  <- geqScalar 0 rpn_cls_targets
            rpn_cls_loss <- sigmoidBCE rpn_raw_scores rpn_cls_targets (Just sample_mask) AggSum
            -- a  <- log2_ rpn_cls_prob
            -- ra <- log2_ =<< rsubScalar 1 rpn_cls_prob
            -- b  <- identity rpn_cls_targets
            -- rb <- rsubScalar 1 rpn_cls_targets
            -- rpn_cls_loss <- (join $ liftM2 add_ (mul_ a b) (mul_ ra rb)) >>= rsubScalar 0

            -- average number of targets per batch example
            cls_mask <- geqScalar 0 rpn_cls_targets
            num_pos_avg  <- sum_ cls_mask Nothing False >>= divScalar (fromIntegral batch_size) >>= addScalar 1e-14

            -- rpn_cls_loss: (B,)
            rpn_cls_loss <- sum_ rpn_cls_loss Nothing False >>= flip div_ num_pos_avg
            rpn_cls_loss <- prim _MakeLoss (#data := rpn_cls_loss .& #grad_scale := 1.0 .& Nil)

            rpn_bbox_reg  <- sub_ rpn_raw_boxregs rpn_box_targets
            rpn_bbox_reg  <- prim _smooth_l1 (#data := rpn_bbox_reg .& #scalar := 3.0 .& Nil)
            rpn_bbox_loss <- mul_ rpn_bbox_reg rpn_box_masks
            rpn_bbox_loss <- sum_ rpn_bbox_loss Nothing False >>= flip div_ num_pos_avg
            rpn_bbox_loss <- prim _MakeLoss
                                (#data := rpn_bbox_loss .& #grad_scale := 1.0 .& Nil)

            box_feat <- prim __contrib_AdaptiveAvgPooling2D (#data := top_feat .& #output_size := [7, 7] .& Nil)
            -- box_feat <- pooling     (#data      := top_feat
            --                       .& #kernel    := [3,3]
            --                       .& #stride    := [2,2]
            --                       .& #pad       := [1,1]
            --                       .& #pool_type := #avg .& Nil)
            -- box_feat <- named "rcnn_cls_score_fc" $
            --             fullyConnected (#data := box_feat .& #num_hidden := 1024 .& Nil)
            box_feat <- activation     (#data := box_feat .& #act_type  := #relu .& Nil)

            -- rcnn class prediction
            -- cls_score: (batch_size * rcnn_batch_rois, rcnn_num_classes)
            cls_score <- named "rcnn_cls_score" $
                         fullyConnected (#data := box_feat .& #num_hidden := rcnn_num_classes .& Nil)
            cls_score <- reshape [batch_size, rcnn_batch_rois, rcnn_num_classes] cls_score

            -- `preserve_shape = True` makes softmax on the last dim.
            -- `normalization = valid` divides the loss by the number of valid items.
            --      we actually want to divide by average number of valid items in the batch,
            --      so scale up by the size batch_size
            cls_prob  <- named "rcnn_cls_prob" $
                         softmaxoutput (#data := cls_score
                                     .& #label := cls_targets
                                     .& #preserve_shape := True
                                     .& #use_ignore := True
                                     .& #ignore_label := -1
                                     .& #normalization := #valid
                                     .& #grad_scale := fromIntegral batch_size .& Nil)

            ---------------------------
            -- bbox_loss part
            --
            bbox_feature <- named "rcnn_bbox_feature" $ fullyConnected (#data := top_feat .& #num_hidden := 1024 .& Nil)
            bbox_feature <- activation  (#data := bbox_feature .& #act_type := #relu .& Nil)
            -- bbox_feature: (B * rcnn_batch_rois, num_hidden) ==> (B, rcnn_batch_rois, num_hidden)
            bbox_feature <- expandDims 0 bbox_feature >>= reshape [batch_size, rcnn_batch_rois, 1024]
            -- select only feature that has foreground gt for each batch example
            -- positive_indices: (B, rcnn_fg_fraction * num_sample)
            bbox_feature <- forM ([0..batch_size-1] :: [_]) $ \i -> do
                ind <- sliceAxis positive_indices 0 i (Just (i+1)) >>= squeeze Nothing
                bat <- sliceAxis bbox_feature 0 i (Just (i+1)) >>= squeeze Nothing
                takeI ind bat
            bbox_feature <- concat_ 0 bbox_feature

            -- for each foreground ROI, predict boxes (reg) for each foreground class
            avg_valid_pred <- gtScalar (-1) cls_targets
                                >>= \s -> sum_ s Nothing False
                                >>= divScalar (fromIntegral batch_size)
                                >>= addScalar 1e-14

            bbox_pred <- named "rcnn_bbox_pred" $
                         fullyConnected (#data := bbox_feature .& #num_hidden := 4 * (rcnn_num_classes - 1) .& Nil)
            -- bbox_pred: (B * rcnn_fg_fraction * num_sample, num_fg_classes * 4)
            --        ==> (B, rcnn_fg_fraction * num_sample, num_fg_classes, 4)
            bbox_pred <- reshape [batch_size, -1, rcnn_num_classes - 1, 4] bbox_pred
            bbox_reg  <- sub_ bbox_pred bbox_targets
            bbox_reg  <- prim _smooth_l1 (#data := bbox_reg .& #scalar := 1.0 .& Nil)
            bbox_loss <- mul_ bbox_reg bbox_masks
            bbox_loss <- sum_ bbox_loss Nothing False >>= flip div_ avg_valid_pred
            bbox_loss <- prim _MakeLoss (#data := bbox_loss .& #grad_scale := 1.0 .& Nil)

            cls_targets   <- reshape [batch_size, -1] cls_targets >>= blockGrad
            box_targets   <- blockGrad bbox_targets
            rpn_cls_prob  <- blockGrad rpn_cls_prob

            result_sym <- group $ [rpn_cls_prob, rpn_cls_loss, rpn_bbox_loss, cls_prob, bbox_loss, cls_targets]

            return $ (FasterRCNN {
                _rpn_loss = (rpn_cls_prob, rpn_cls_loss, rpn_bbox_loss),
                _box_loss = (cls_prob, bbox_loss),
                _cls_targets = cls_targets,
                _roi_boxes = roi_boxes,
                _gt_matches = matches,
                _positive_indices = positive_indices,
                _top_feature = top_feat
            }, result_sym)


graphI :: RcnnConfiguration -> Layer (FasterRCNN, SymbolHandle)
graphI conf@RcnnConfigurationInference{..} =  do
    -- dat: (B, image_height, image_width)
    dat <- variable "data"
    -- imInfo: (B, 3,)
    imInfo <- variable "im_info"

    sequential "features" $ do
        feats <- features1 backbone dat

        -- roi_boxes: (B, rpn_post_topk, 4), rpn predicted boxes (x0,y0,x1,y1)
        (roi_scores, roi_boxes, _, _) <- rpn conf feats imInfo

        roi_batchid <- prim __arange (#start := 0 .& #stop := Just (fromIntegral batch_size) .& Nil)
        roi_batchid <- prim _repeat  (#data := roi_batchid .& #repeats := rpn_post_topk .& Nil)
        roi_batchid <- reshape [-1, 1] roi_batchid
        -- rois: (B * rpn_post_topk, 4)
        rois <- reshape [-1, 4] roi_boxes
        rois <- concat_ 1 [roi_batchid, rois]
        feat_aligned <- alignROIs feats rois (stageList backbone) rcnn_pooled_size feature_strides

        top_feat <- features2 backbone feat_aligned

        sequential "rcnn" $ do
            box_feat <- prim __contrib_AdaptiveAvgPooling2D (#data := top_feat .& #output_size := [7, 7] .& Nil)
            box_feat <- activation (#data := box_feat .& #act_type := #relu .& Nil)

            -- rcnn class prediction
            -- cls_score: (batch_size * rpn_post_topk, rcnn_num_classes)
            cls_score <- named "rcnn_cls_score" $
                         fullyConnected (#data := box_feat .& #num_hidden := rcnn_num_classes .& Nil)
            cls_score <- reshape [batch_size, rpn_post_topk, rcnn_num_classes] cls_score

            cls_prob  <- softmax (#data := cls_score .& #axis := (-1) .& Nil)

            bbox_feature <- named "rcnn_bbox_feature" $ fullyConnected (#data := top_feat .& #num_hidden := 1024 .& Nil)
            -- bbox_feature: (B * rpn_post_topk, num_hidden)
            bbox_feature <- activation  (#data := bbox_feature .& #act_type := #relu .& Nil)

            bbox_pred    <- named "rcnn_bbox_pred" $
                            fullyConnected (#data := bbox_feature
                                         .& #num_hidden := 4 * (rcnn_num_classes - 1) .& Nil)
            -- bbox_pred: (B * rpn_post_topk, num_fg_classes * 4)
            --        ==> (B, rpn_post_topk, num_fg_classes, 4)
            bbox_pred    <- reshape [batch_size, -1, rcnn_num_classes - 1, 4] bbox_pred

            -------------------
            -- decode the classes and boxes
            --

            -- `cls_prob` predicts `rcnn_num_classes` classes, background class at 0.
            (clsids, scores) <- multiClassDecodeWithClsId rcnn_num_classes (-1) 0.01 cls_prob

            -- tranpose the bbox_pred, clsids, scores, because bbox_nms does suppresion on
            -- the last two dimensions
            bbox_pred <- transpose bbox_pred [0, 2, 1, 3]
            clsids    <- transpose clsids    [0, 2, 1]
            scores    <- transpose scores    [0, 2, 1]

            -- `roi_boxes` B x (1, rpn_post_topk, 4)
            roi_boxes   <- splitBySections batch_size 0 False roi_boxes
            -- `bbox_pred` B x (num_fg_classes, rpn_post_topk, 4)
            bbox_pred   <- splitBySections batch_size 0 True bbox_pred
            -- `bbox_clsids` B x (num_fg_classes, rpn_post_topk)
            bbox_clsids <- splitBySections batch_size 0 True clsids
            -- `bbox_scores` B x (num_fg_classes, rpn_post_topk)
            bbox_scores <- splitBySections batch_size 0 True scores


            let (std0, std1, std2, std3) = bbox_reg_std

            results <- forM (zip4 roi_boxes bbox_pred bbox_clsids bbox_scores) $ \ (roi, pred, clsid, score) -> do
                bbox_decoded <- prim __contrib_box_decode (#data := pred
                                                        .& #anchors := roi
                                                        .& #format := #corner
                                                        .& #std0 := std0
                                                        .& #std1 := std1
                                                        .& #std2 := std2
                                                        .& #std3 := std3
                                                        .& Nil)

                -- concatenate all clsid, score, and box
                -- res: (num_fg_classes, rpn_post_topk, 6)
                clsid <- expandDims (-1) clsid
                score <- expandDims (-1) score
                res <- concat_ (-1) [clsid, score, bbox_decoded]

                -- if force_nms, we do nms all boxes from all classes
                res <- if rcnn_force_nms
                       then reshape [1, -1, 0] res
                       else return res

                res <- prim __contrib_box_nms (#data := res
                                            .& #overlap_thresh := rcnn_nms_thresh
                                            .& #valid_thresh := 0.001
                                            .& #topk := rcnn_topk
                                            .& #coord_start := 2
                                            .& #score_index := 1
                                            .& #id_index    := 0
                                            .& #force_suppress := rcnn_force_nms .& Nil)
                res <- sliceAxis res 1 0 (Just rcnn_topk)
                -- final result: (num_fg_classes * rcnn_topk, 6)
                reshape [-3, 0] res

            results <- stack 0 results
            result_cls_ids <- sliceAxis results (-1) 0 (Just 1)
            result_scores  <- sliceAxis results (-1) 1 (Just 2)
            result_boxes   <- sliceAxis results (-1) 2 Nothing

            res_sym <- group [result_cls_ids, result_scores, result_boxes]
            let res_data = FasterRCNNInferenceOnly
                            { _top_feature = top_feat
                            , _cls_ids     = result_cls_ids
                            , _scores      = result_scores
                            , _boxes       = result_boxes
                            }
            return $ (res_data, res_sym)

