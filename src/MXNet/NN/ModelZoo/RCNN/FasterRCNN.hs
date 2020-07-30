module MXNet.NN.ModelZoo.RCNN.FasterRCNN where

import           Control.Lens                 (bimap, ix, (^?!))
import qualified Data.Array.Repa              as Repa
import           Data.Array.Repa.Index
import           Data.Array.Repa.Shape
import           Formatting
import           RIO
import qualified RIO.HashMap                  as M
import           RIO.List                     (unzip3, zip3)
import           RIO.List.Partial             (head, last)
import qualified RIO.NonEmpty                 as NE (toList)
import qualified RIO.Text                     as T
import qualified RIO.Vector.Boxed             as V
import qualified RIO.Vector.Unboxed           as UV

import           MXNet.Base
import qualified MXNet.Base.NDArray           as A
import           MXNet.Base.Operators.NDArray (argmax, argmax_channel)
import           MXNet.Base.Operators.Symbol  (add_n, clip, repeat, sigmoid,
                                               slice_axis, smooth_l1, transpose,
                                               _Custom, _MakeLoss, _arange,
                                               _contrib_AdaptiveAvgPooling2D,
                                               _contrib_ROIAlign,
                                               _contrib_box_decode,
                                               _contrib_box_nms, _ones, _zeros)
import           MXNet.NN.EvalMetric
import           MXNet.NN.Layer
import           MXNet.NN.ModelZoo.RCNN.FPN
import           MXNet.NN.ModelZoo.RCNN.RCNN
import qualified MXNet.NN.ModelZoo.Resnet     as Resnet
import qualified MXNet.NN.ModelZoo.VGG        as VGG
import qualified MXNet.NN.NDArray             as A
import           MXNet.NN.Utils.Repa

data Backbone = VGG16
    | RESNET50
    | RESNET101
    | RESNET50FPN
    deriving (Show, Read, Eq)

data RcnnConfiguration = RcnnConfiguration
    { rpn_anchor_scales    :: [Int]
    , rpn_anchor_ratios    :: [Float]
    , rpn_anchor_base_size :: Int
    -- , rpn_feature_stride  :: Int
    , rpn_batch_rois       :: Int
    , rpn_pre_topk         :: Int
    , rpn_post_topk        :: Int
    , rpn_nms_thresh       :: Float
    , rpn_min_size         :: Int
    , rpn_fg_fraction      :: Float
    , rpn_fg_overlap       :: Float
    , rpn_bg_overlap       :: Float
    , rpn_allowd_border    :: Int
    , rcnn_num_classes     :: Int
    , rcnn_pooled_size     :: [Int]
    , rcnn_batch_rois      :: Int
    , rcnn_fg_fraction     :: Float
    , rcnn_fg_overlap      :: Float
    -- , rcnn_bbox_stds      :: [Float]
    , rcnn_max_num_gt      :: Int
    , feature_strides      :: [Int]
    , batch_size           :: Int
    , pretrained_weights   :: String
    , backbone             :: Backbone
    }
    deriving Show

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
rpn RcnnConfiguration{..} convFeats imInfo = unique "rpn" $ do
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
    layers <- zipWithM (rpn_head conv3x3_feat conv1x1_cls conv1x1_reg) (NE.toList convFeats) feature_strides

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
        num_variation = length rpn_anchor_scales * length rpn_anchor_ratios
        rpn_head conv3x3_feat conv1x1_cls conv1x1_reg feat stride = do
            x <- conv3x3_feat feat
            x <- activation (#data := x
                          .& #act_type := #relu .& Nil)

            anchors <- prim _Custom (#op_type := "anchor_generator"
                                  .& #data    := [x]
                                  .& #stride     :≅ stride
                                  .& #scales     :≅ rpn_anchor_scales
                                  .& #ratios     :≅ rpn_anchor_ratios
                                  .& #base_size  :≅ rpn_anchor_base_size
                                  .& #alloc_size :≅ ((128, 128) :: (Int, Int))
                                  .& Nil)

            rpn_raw_score <- conv1x1_cls x
            -- (batch_size, num_variation, H, W) ==> (batch_size, H, W, num_variation)
            rpn_raw_score <- prim transpose (#data := rpn_raw_score  .& #axes := [0, 2, 3, 1] .& Nil)
            -- (batch_size, H, W, num_variation) ==> (batch_size, H*W*num_variation, 1)
            rpn_raw_score <- reshape [0, -1, 1] rpn_raw_score

            rpn_cls_score <- prim sigmoid (#data := rpn_raw_score .& Nil)
            rpn_cls_score <- reshape [0, -1, 1] rpn_cls_score

            rpn_raw_boxreg <- conv1x1_reg x
            -- (batch_size, num_variation * 4, H, W) ==> (batch_size, H, W, num_variation * 4)
            rpn_raw_boxreg <- prim transpose (#data := rpn_raw_boxreg .& #axes := [0, 2, 3, 1] .& Nil)
            -- (batch_size, H, W, num_variation * 4) ==> (batch_size, H*W*num_variation, 4)
            rpn_raw_boxreg <- reshape [0, -1, 4] rpn_raw_boxreg

            rpn_pre <- region_proposer (fromIntegral rpn_min_size) anchors rpn_raw_boxreg rpn_cls_score

            return (rpn_pre, rpn_raw_score, rpn_raw_boxreg)

        region_proposer min_size anchors boxregs scores = do
            rois <- prim _contrib_box_decode (#data := boxregs .& #anchors := anchors .& Nil)
            (xmin, ymin, xmax, ymax) <- bbox_clip_to_image rois imInfo
            width   <- sub_ xmax xmin >>= addScalar 1
            height  <- sub_ ymax ymin >>= addScalar 1
            invalid <- join $ liftM2 add_ (ltScalar min_size width) (ltScalar min_size height)
            mask    <- onesLike invalid >>= mulScalar (-1)
            scores  <- where_ invalid mask scores
            invalid <- broadcastAxis [2] [4] invalid
            mask    <- onesLike invalid >>= mulScalar (-1)
            rois    <- where_ invalid mask rois
            blockGrad =<< named "proposals" (concat_ (-1) [scores, rois])

        nms rpn_pre = do
            -- rpn_pre shape: (batch_size, num_anchors, 5)
            -- with dim 3: [score, xmin, ymin, xmax, ymax]
            tmp <- prim _contrib_box_nms (#data := rpn_pre
                               .& #overlap_thresh := rpn_nms_thresh
                               .& #topk := rpn_pre_topk
                               .& #coord_start := 1
                               .& #score_index := 0
                               .& #id_index    := (-1)
                               .& #force_suppress := True .& Nil)
            tmp <- prim slice_axis (#data := tmp
                                 .& #axis := 1
                                 .& #begin := 0
                                 .& #end := Just rpn_post_topk .& Nil)
            rpn_roi_scores <- blockGrad =<<
                              prim slice_axis (#data := tmp
                                            .& #axis := (-1)
                                            .& #begin := 0
                                            .& #end := Just 1 .& Nil)
            rpn_roi_boxes  <- blockGrad =<<
                              prim slice_axis (#data := tmp
                                            .& #axis := (-1)
                                            .& #begin := 1
                                            .& #end := Nothing .& Nil)
            return (rpn_roi_scores, rpn_roi_boxes)

        bbox_clip_to_image rois info = do
            -- rois: (B, N, 4)
            -- info: (B, 3)
            -- return: (B,N), (B,N), (B,N), (B,N)
            [xmin, ymin, xmax, ymax] <- splitBySections 4 (-1) False rois
            [_, height, width]       <- splitBySections 3 (-1) False info
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

alignROIs :: NonEmpty SymbolHandle -> SymbolHandle -> [Int] -> [Int] -> [Int] -> Layer _
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
    roi_level <- prim clip (#data := roi_level_raw
                         .& #a_min := fromIntegral min_stage
                         .& #a_max := fromIntegral max_stage .& Nil)
                 >>= squeeze Nothing
    let align (lvl, feat, stride) = do
            cond <- eqScalar (fromIntegral lvl) roi_level
            omit <- onesLike rois >>= mulScalar (-1)
            masked <- where_ cond rois omit
            prim _contrib_ROIAlign (#data := feat
                                 .& #rois := masked
                                 .& #pooled_size := roi_pooled_size
                                 .& #spatial_scale := 1 / fromIntegral stride
                                 .& #sample_ratio := 2 .& Nil)
    features <- mapM align $ zip3 [max_stage,max_stage-1..min_stage] (NE.toList features) strides
    prim add_n (#args := features .& Nil)


symbolTrain :: RcnnConfiguration -> Layer SymbolHandle
symbolTrain conf@RcnnConfiguration{..} =  do
    -- dat: (B, image_height, image_width)
    dat <- variable "data"
    -- imInfo: (B, 3,)
    imInfo <- variable "im_info"
    -- gt_boxes: (B, M, 5), the last dim: min_x, min_y, max_x, max_y, class_id (background class: 0)
    gt_boxes <- variable "gt_boxes"
    -- box regression mean/std: (4,)
    box_reg_mean <- variable "box_reg_mean"
    box_reg_std  <- variable "box_reg_std"
    rpn_cls_targets <- variable "rpn_cls_targets"
    rpn_box_targets <- variable "rpn_box_targets"
    rpn_box_masks   <- variable "rpn_box_masks"

    -- box_reg_mean <- unique' $ prim _zeros (#shape := [4] .& #dtype := #float32 .& Nil)
    -- box_reg_std  <- unique' $ prim _ones  (#shape := [4] .& #dtype := #float32 .& Nil)

    gt_labels <- unique' $ prim slice_axis (#data  := gt_boxes
                               .& #axis  := (-1)
                               .& #begin := 4
                               .& #end   := Nothing .& Nil)

    gt_boxes  <- unique' $ prim slice_axis (#data  := gt_boxes
                               .& #axis  := (-1)
                               .& #begin := 0
                               .& #end   := Just 4 .& Nil)

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

            roi_batchid <- prim _arange (#start := 0 .& #stop := Just (fromIntegral batch_size) .& Nil)
            roi_batchid <- prim repeat  (#data := roi_batchid .& #repeats := rcnn_batch_rois .& Nil)
            roi_batchid <- reshape [-1, 1] roi_batchid
            -- rois: (B * rcnn_batch_rois, 4)
            rois <- reshape [-1, 4] rois_boxes
            rois <- concat_ 1 [roi_batchid, rois] >>= blockGrad
            feat <- alignROIs feats rois (stageList backbone) rcnn_pooled_size feature_strides
            return (feat, rois_boxes, samples, matches)

        -- feat_aligned: (batch_size * rcnn_batch_rois, num_channels, feature_height, feature_width)
        -- Apply the remaining feature extraction layers
        top_feat <- features2 backbone feat_aligned

        unique "rcnn" $ do
            (cls_targets, bbox_targets, bbox_masks, positive_indices) <-
                rcnnTargetGenerator batch_size
                                    (rcnn_num_classes-1)
                                    (floor $ rcnn_fg_fraction * fromIntegral rcnn_batch_rois)
                                    samples
                                    matches
                                    roi_boxes
                                    gt_labels
                                    gt_boxes
                                    box_reg_mean
                                    box_reg_std

            -- sigmoid + binary-cross-entropy
            -- rpn_raw_scores: (B, num_rois, 1)
            -- rpn_cls_targets: (B, num_rois, 1)
            rpn_cls_prob <- prim sigmoid (#data := rpn_raw_scores .& Nil)
            a  <- log2_ rpn_cls_prob
            ra <- rsubScalar 1 a >>= log2_
            b  <- identity rpn_cls_targets
            rb <- rsubScalar 1 rpn_cls_targets
            rpn_cls_loss <- (join $ liftM2 add_ (mul_ a b) (mul_ ra rb)) >>= rsubScalar 0

            -- average number of targets per batch example
            cls_mask <- geqScalar 0 rpn_cls_targets
            num_pos_avg  <- sum_ cls_mask Nothing >>= divScalar (fromIntegral batch_size)

            rpn_cls_loss <- mul_ rpn_cls_loss cls_mask >>= flip divBroadcast num_pos_avg
            rpn_cls_loss <- prim _MakeLoss (#data := rpn_cls_loss .& #grad_scale := 1.0 .& Nil)

            rpn_bbox_reg  <- sub_ rpn_raw_boxregs rpn_box_targets
            rpn_bbox_reg  <- prim smooth_l1 (#data := rpn_bbox_reg .& #scalar := 3.0 .& Nil)
            rpn_bbox_loss <- mul_ rpn_bbox_reg rpn_box_masks >>= flip divBroadcast num_pos_avg
            rpn_bbox_loss <- prim _MakeLoss
                                (#data := rpn_bbox_loss .& #grad_scale := 1.0 .& Nil)

            box_feat <- prim _contrib_AdaptiveAvgPooling2D (#data := top_feat .& #output_size := [1, 1] .& Nil)

            -- rcnn class prediction
            -- cls_score: (batch_size * rcnn_batch_rois, rcnn_num_classes)
            cls_score <- named "rcnn_cls_score" $
                         fullyConnected (#data := box_feat .& #num_hidden := rcnn_num_classes .& Nil)
            cls_score <- reshape [batch_size, rcnn_batch_rois, rcnn_num_classes] cls_score

            cls_prob  <- named "rcnn_cls_prob" $
                         softmaxoutput (#data := cls_score
                                     .& #label := cls_targets
                                     .& #normalization := #batch
                                     .& #preserve_shape := True
                                     .& #use_ignore := True
                                     .& #ignore_label := -1
                                     .& #grad_scale := 1 / fromIntegral rcnn_batch_rois .& Nil)

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
                ind <- prim slice_axis (#data := positive_indices
                                     .& #axis := 0
                                     .& #begin := i .& #end := Just (i+1) .& Nil)
                       >>= squeeze Nothing
                bat <- prim slice_axis (#data := bbox_feature
                                     .& #axis := 0
                                     .& #begin := i .& #end := Just (i+1) .& Nil)
                       >>= squeeze Nothing
                takeI ind bat
            bbox_feature <- concat_ 0 bbox_feature

            -- for each foreground ROI, predict boxes (reg) for each foreground class
            bbox_pred <- named "rcnn_bbox_pred" $
                         fullyConnected (#data := bbox_feature .& #num_hidden := 4 * (rcnn_num_classes - 1) .& Nil)
            -- bbox_pred: (B * rcnn_fg_fraction * num_sample, num_fg_classes * 4)
            --        ==> (B, rcnn_fg_fraction * num_sample, num_fg_classes, 4)
            bbox_pred <- reshape [batch_size, -1, rcnn_num_classes - 1, 4] bbox_pred
            bbox_reg  <- sub_ bbox_pred bbox_targets
            bbox_reg  <- prim smooth_l1 (#data := bbox_reg .& #scalar := 1.0 .& Nil)
            bbox_loss <- mul_ bbox_reg bbox_masks
            bbox_loss <- prim _MakeLoss (#data := bbox_loss .& #grad_scale := 1.0 / fromIntegral rcnn_batch_rois .& Nil)

            cls_targets   <- reshape [batch_size, -1] cls_targets >>= blockGrad
            rpn_cls_prob  <- blockGrad rpn_cls_prob

            group $ [rpn_cls_prob, rpn_cls_loss, rpn_bbox_loss, cls_prob, bbox_loss, cls_targets]


symbolInfer :: RcnnConfiguration -> Layer SymbolHandle
symbolInfer conf@RcnnConfiguration{..} = error "no infer"
--    -- dat:
--    dat <- variable "data"
--    -- imInfo:
--    imInfo <- variable "im_info"
--
--    sequential "features" $ do
--        convFeat <- features1 backbone dat
--        (convFeat, rois, _, _) <- rpn conf convFeat imInfo
--
--        ---------------------------
--        -- cls_prob part
--        --
--        roiPool <- named "roi_pool" $ prim _ROIPooling
--                     (#data := convFeat
--                   .& #rois := rois
--                   .& #pooled_size := rcnn_pooled_size
--                   .& #spatial_scale := 1.0 / fromIntegral rcnn_feature_stride .& Nil)
--
--        topFeat <- features2 backbone roiPool
--
--        unique "rcnn" $ do
--            clsScore <- named "cls_score" $
--                        fullyConnected (#data := topFeat
--                                     .& #num_hidden := rcnn_num_classes .& Nil)
--            clsProb  <- named "cls_prob" $
--                        softmax (#data := clsScore .& Nil)
--
--            ---------------------------
--            -- bbox_loss part
--            --
--            bboxPred <- named "bbox_pred" $
--                        fullyConnected (#data := topFeat
--                                     .& #num_hidden := 4 * rcnn_num_classes .& Nil)
--
--            group [rois, clsProb, bboxPred]

--------------------------------
data RPNAccMetric a = RPNAccMetric Text

instance EvalMetricMethod RPNAccMetric where
    data MetricData RPNAccMetric a = RPNAccMetricData Text Text (IORef Int) (IORef Int)
    newMetric phase (RPNAccMetric label) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNAccMetricData phase label a b

    formatMetric (RPNAccMetricData _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RPNAcc: " % fixed 2 % ">") (100 * fromIntegral s / fromIntegral n :: Float)

    evalMetric (RPNAccMetricData phase lname cntRef sumRef) bindings outputs = liftIO $  do
        -- traceShowM "rpnacc"
        -- rpn class pred: (B, rpn_post_topk, 1)
        -- rpn class label: (B, rpn_post_topk, 1)
        pred  <- toRepa @DIM3 (outputs  ^?! ix 0)
        label <- toRepa @DIM3 (bindings ^?! ix lname)

        let pred_thr = Repa.map (\v -> if v > 0.5 then 1 else 0) pred
            matches  = Repa.zipWith (\v w -> if v == w then 1 else 0) pred_thr label
            valid_mask = Repa.map (\v -> if v >= 0 then 1 else 0) label

        num_valid   <- Repa.sumAllP valid_mask
        num_matches <- Repa.sumAllP matches

        modifyIORef' sumRef (+ num_matches)
        modifyIORef' cntRef (+ num_valid)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = fromIntegral s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc


data RCNNAccMetric a = RCNNAccMetric

instance EvalMetricMethod RCNNAccMetric where
    data MetricData RCNNAccMetric a = RCNNAccMetricData {
        _rcnn_acc_phase :: Text,
        _rcnn_acc_fg  :: IORef (Int, Int)
    }
    newMetric phase RCNNAccMetric = do
        a <- liftIO $ newIORef (0, 0)
        return $ RCNNAccMetricData phase a

    formatMetric (RCNNAccMetricData _ accum_fg) = liftIO $ do
        (fg_s, fg_n)   <- liftIO $ readIORef accum_fg
        return $ sformat ("<RCNNAcc: " % fixed 2 % ">")
            (100 * fromIntegral fg_s  / fromIntegral fg_n  :: Float)

    evalMetric rcnn_acc _ outputs = liftIO $  do
        -- traceShowM "rcnnacc"
        -- cls_prob: (B, num_pos_rois, rcnn_num_classes)
        -- label:    (B, num_pos_rois)
        let cls_prob = outputs ^?! ix 3

        label <- toRepa @DIM2 (outputs ^?! ix 5)
        cls_prob   <- A.makeNDArrayLike cls_prob contextCPU >>= A.copy cls_prob
        pred_class <- sing argmax (#data := unNDArray cls_prob .& #axis := Just 2 .& Nil)
        pred_class <- toRepa @DIM2 (NDArray pred_class :: NDArray a)

        num_matches <- Repa.sumAllP $ Repa.zipWith (\v w -> if v == w then 1 else 0) pred_class label
        num_fg  <- Repa.sumAllP $ Repa.map (\v -> if v >  0 then 1 else 0) label

        let ref_acc_fg  = _rcnn_acc_fg  rcnn_acc
        modifyIORef' ref_acc_fg  (bimap (+ num_matches) (+ num_fg))

        (fg_s,  fg_n)  <- readIORef ref_acc_fg
        let fg_acc  = fromIntegral fg_s  / fromIntegral fg_n
            phase   = _rcnn_acc_phase rcnn_acc
        return $ M.fromList [(phase `T.append` "_acc", fg_acc)]

data RPNLogLossMetric a = RPNLogLossMetric Text

instance EvalMetricMethod RPNLogLossMetric where
    data MetricData RPNLogLossMetric a = RPNLogLossMetricData Text Text (IORef Int) (IORef Double)
    newMetric phase (RPNLogLossMetric lname) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNLogLossMetricData phase lname a b

    formatMetric (RPNLogLossMetricData _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RPNLogLoss: " % fixed 4 % ">") (realToFrac s / fromIntegral n :: Float)

    evalMetric (RPNLogLossMetricData phase lname cntRef sumRef) bindings outputs = liftIO $  do
        -- traceShowM "rpnlogloss"
        let pred  = outputs  ^?! ix 0
            label = bindings ^?! ix lname

        -- both pred and label: (B, rpn_post_topk, 1)
        label <- A.reshape label [-1]
        label <- toRepa @DIM1 label
        pred  <- A.reshape pred [-1]
        pred  <- toRepa @DIM1 pred

        let Z :. size = Repa.extent label
            ep = constant (Z :. size) 1e-14
            one = constant (Z :. size) 1

        let a = Repa.map log (pred Repa.+^ ep) Repa.*^ label
            b = Repa.map log (one Repa.-^ pred Repa.+^ ep) Repa.*^ (one Repa.-^ label)
            ce = Repa.map (0-) $ a Repa.+^ b
        cls_loss_val <- realToFrac <$> Repa.sumAllP ce
        modifyIORef' sumRef (+ cls_loss_val)
        modifyIORef' cntRef (+ size)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc

data RCNNLogLossMetric a = RCNNLogLossMetric

instance EvalMetricMethod RCNNLogLossMetric where
    data MetricData RCNNLogLossMetric a = RCNNLogLossMetricData Text (IORef Int) (IORef Double)
    newMetric phase RCNNLogLossMetric = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RCNNLogLossMetricData phase a b

    formatMetric (RCNNLogLossMetricData _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RCNNLogLoss: " % fixed 4 % ">") (realToFrac s / fromIntegral n :: Float)

    evalMetric (RCNNLogLossMetricData phase cntRef sumRef) _ outputs = liftIO $  do
        -- traceShowM "rcnnlogloss"
        -- rcnn class prediction: (B, rcnn_batch_rois, num_fg_classes+1)
        cls_prob <- toRepa @DIM3 (outputs ^?! ix 3)
        -- rcnn generated class target: (B, rcnn_batch_rois), value [0, num_fg_classes] or -1
        label    <- toRepa @DIM2 (outputs ^?! ix 5)

        let lbl_shp@(Z :. _ :. num_rois) = Repa.extent label
            ce = Repa.fromFunction lbl_shp (\ pos@(Z :. bi :. ai) ->
                    let target = floor $ label Repa.! pos
                        prob   = cls_prob Repa.! (Z :. bi :. ai :. target)
                        eps    = 1e-14
                     in if target == -1 then 0 else - log (eps + prob))

        ce <- Repa.sumAllP ce
        num_valid <- Repa.sumAllP $ Repa.map (\v -> if v == -1 then 0 else 1) label
        modifyIORef' sumRef (+ realToFrac ce)
        modifyIORef' cntRef (+ num_valid)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc

data RPNL1LossMetric a = RPNL1LossMetric

instance EvalMetricMethod RPNL1LossMetric where
    data MetricData RPNL1LossMetric a = RPNL1LossMetricData Text (IORef Int) (IORef Double)
    newMetric phase RPNL1LossMetric = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNL1LossMetricData phase a b

    formatMetric (RPNL1LossMetricData _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RPNL1Loss: " % fixed 3 % ">") (realToFrac s / fromIntegral n :: Float)

    evalMetric (RPNL1LossMetricData phase cntRef sumRef) bindings outputs = liftIO $  do
        -- traceShowM "rpnl1loss"
        bbox_loss <- toRepa @DIM3 (outputs ^?! ix 2)
        all_loss  <- Repa.sumAllP $ Repa.map abs bbox_loss
        let Z:.batch_size:._:._ = Repa.extent bbox_loss

        modifyIORef' sumRef (+ realToFrac all_loss)
        modifyIORef' cntRef (+ batch_size)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc

data RCNNL1LossMetric a = RCNNL1LossMetric

instance EvalMetricMethod RCNNL1LossMetric where
    data MetricData RCNNL1LossMetric a = RCNNL1LossMetricData Text (IORef Int) (IORef Double)
    newMetric phase RCNNL1LossMetric = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RCNNL1LossMetricData phase a b

    formatMetric (RCNNL1LossMetricData _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RCNNL1Loss: " % fixed 5 % ">") (realToFrac s / fromIntegral n :: Float)

    evalMetric (RCNNL1LossMetricData phase cntRef sumRef) _ outputs = liftIO $ do
        -- traceShowM "rcnnlogloss"
        -- rcnn box loss: (B, num_pos_rois, num_fg_classes, 4)

        -- bbox_masks <- toRepa @DIM4 (outputs ^?! ix 6)
        -- let ind = Repa.fromFunction (Z:.64 :: DIM1) (\(Z:.i) ->
        --             let ind = Repa.slice bbox_masks (Z:.(0::Int):.i:.Repa.All:.(0::Int))
        --             in case UV.findIndex (==1) $ Repa.toUnboxed $ Repa.computeUnboxedS ind of
        --                  Just n  -> n
        --                  Nothing -> -1)
        -- traceShowM (Repa.computeUnboxedS ind)
        -- cnt <- Repa.sumAllP $ Repa.map (\v -> if v > 0 then 1 else 0::Int) bbox_masks
        -- traceShowM cnt

        -- bbox_preds <- toRepa @DIM4 (outputs ^?! ix 7)
        -- traceShowM $ Repa.computeUnboxedS $ Repa.slice bbox_preds (Z:.(0::Int):.(0::Int):.Repa.All:.Repa.All)

        bbox_loss <- toRepa @DIM4 (outputs ^?! ix 4)
        all_loss  <- Repa.sumAllP $ Repa.map abs bbox_loss

        let Z:.batch_size:.num_pos_rois:._:._ = Repa.extent bbox_loss

        modifyIORef' sumRef (+ realToFrac all_loss)
        modifyIORef' cntRef (+ batch_size * num_pos_rois)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc

constant :: (Shape sh, UV.Unbox a) => sh -> a -> Repa.Array Repa.U sh a
constant shp val = Repa.fromListUnboxed shp (replicate (size shp) val)
