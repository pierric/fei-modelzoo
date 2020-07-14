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
import           MXNet.Base.Operators.Symbol  (add_n, clip,
                                               repeat, sigmoid, slice_axis,
                                               smooth_l1, transpose, _Custom,
                                               _MakeLoss, _arange,
                                               _contrib_ROIAlign,
                                               _contrib_box_decode,
                                               _contrib_box_nms)
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
    { rpn_anchor_scales   :: [Int]
    , rpn_anchor_ratios   :: [Float]
    -- , rpn_feature_stride  :: Int
    , rpn_batch_rois      :: Int
    , rpn_pre_topk        :: Int
    , rpn_post_topk       :: Int
    , rpn_nms_thresh      :: Float
    , rpn_min_size        :: Int
    , rpn_fg_fraction     :: Float
    , rpn_fg_overlap      :: Float
    , rpn_bg_overlap      :: Float
    , rpn_allowd_border   :: Int
    , rcnn_num_classes    :: Int
    , rcnn_feature_stride :: [Int]
    , rcnn_pooled_size    :: [Int]
    , rcnn_batch_rois     :: Int
    , rcnn_batch_size     :: Int
    , rcnn_fg_fraction    :: Float
    , rcnn_fg_overlap     :: Float
    -- , rcnn_bbox_stds      :: [Float]
    , rcnn_max_num_gt     :: Int
    , pretrained_weights  :: String
    , backbone            :: Backbone
    }
    deriving Show

stageList :: Backbone -> [Int]
stageList RESNET50FPN = [2..5]
stageList _           = []

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
    case convFeats of
      [convFeat] -> do
          let conv3x3_feat x = named "rpn_conv_3x3" $
                               convolution (#kernel := [3,3]
                                         .& #pad := [1,1]
                                         .& #num_filter := 512 .& Nil)
              conv1x1_cls  x = named "rpn_cls_score" $
                               convolution (#data := x
                                         .& #kernel := [1,1]
                                         .& #pad := [0,0]
                                         .& #num_filter := num_variation .& Nil)
              conv1x1_reg  x = named "rpn_bbox_pred" $
                               convolution (#data := x
                                         .& #kernel := [1,1]
                                         .& #pad := [0,0]
                                         .& #num_filter := 4 * num_variation .& Nil)
          (rpn_pre, rpn_raw_score, rpn_raw_boxreg) <- rpn_head conv3x3_feat conv1x1_cls conv1x1_reg convFeat

          (rpn_roi_score, rpn_roi_box) <- nms rpn_pre
          return (rpn_roi_score, rpn_roi_box, rpn_raw_score, rpn_raw_boxreg)

      _ -> do
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
          layers <- forM convFeats $ rpn_head conv3x3_feat conv1x1_reg conv1x1_reg

          let (rpn_pres, rpn_raw_scores, rpn_raw_boxregs) = unzip3 $ NE.toList layers
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
        rpn_head conv3x3_feat conv1x1_cls conv1x1_reg feat = do
            x <- conv3x3_feat feat
            x <- activation (#data := x
                          .& #act_type := #relu .& Nil)

            anchors <- prim _Custom (#op_type := "anchor_generator"
                                  .& #data := [x]
                                  .& #stride :≅ (1 :: Int)
                                  .& #scale  :≅ rpn_anchor_scales
                                  .& #ratios :≅ rpn_anchor_ratios
                                  .& #base_size  :≅ ([32, 32] :: [Int])
                                  .& #alloc_size :≅ ([128, 128] :: [Int])
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
            rois <- prim _Custom (#data := [rois, imInfo] .& #op_type := "bbox_clip_to_image" .& Nil)
            [xmin, ymin, xmax, ymax] <- splitBySections 4 (-1) False rois
            width   <- sub_ xmax xmin >>= addScalar 1
            height  <- sub_ ymax ymin >>= addScalar 1
            invalid <- join $ liftM2 add_ (ltScalar min_size width) (ltScalar min_size height)
            mask    <- onesLike invalid >>= mulScalar (-1)
            scores  <- where_ invalid mask scores
            invalid <- broadcastAxis [2] [4] invalid
            rois    <- where_ invalid mask rois
            blockGrad =<< concat_ (-1) [scores, rois]

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


alignROIs :: NonEmpty SymbolHandle -> SymbolHandle -> [Int] -> [Int] -> [Int] -> Layer SymbolHandle
alignROIs features rois stage_indices roi_pooled_size strides = do
    let min_stage = head stage_indices
        max_stage = last stage_indices
    [xmin, ymin, xmax, ymax] <- splitBySections 4 (-1) False rois
    w <- xmax `sub_` xmin >>= addScalar 1
    h <- ymax `sub_` ymin >>= addScalar 1
    roi_level <- w `mul_` h >>=
                 sqrt_ >>=
                 divScalar 224 >>=
                 addScalar 1e-6 >>=
                 log2_ >>=
                 addScalar 4 >>=
                 floor_
    roi_level <- prim clip (#data := roi_level .& #a_min := fromIntegral min_stage .& #a_max := fromIntegral max_stage .& Nil)
                 >>= squeeze Nothing
    let align (lvl, feat, stride) = do
            cond <- eqScalar (fromIntegral lvl) roi_level
            omit <- onesLike rois >>= mulScalar (-1)
            masked <- where_  cond rois omit
            prim _contrib_ROIAlign (#data := feat
                                 .& #rois := masked
                                 .& #pooled_size := roi_pooled_size
                                 .& #spatial_scale := 1 / fromIntegral stride
                                 .& #sample_ratio := 2 .& Nil)
    features <- mapM align $ zip3 [min_stage..max_stage] (NE.toList features) strides
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
    box_reg_mean <- variable "bbox_reg_mean"
    box_reg_std  <- variable "bbox_reg_std"

    gt_labels <- prim slice_axis (#data  := gt_boxes
                               .& #axis  := (-1)
                               .& #begin := 4
                               .& #end   := Nothing .& Nil)

    gt_boxes  <- prim slice_axis (#data  := gt_boxes
                               .& #axis  := (-1)
                               .& #begin := 0
                               .& #end   := Just 4 .& Nil)

    sequential "features" $ do
        feats <- features1 backbone dat

        (rois_scores, roi_boxes, rpn_raw_scores, rpn_raw_boxregs) <- rpn conf feats imInfo

        -- total number of ROIs in a batch
        -- rcnn_batch_size: number of images
        -- rcnn_batch_rois: number of rois per image
        let num_sample = rcnn_batch_size * rcnn_batch_rois

        (feat_aligned, samples, matches) <- unique "rcnn" $ do
            (rois, samples, matches) <- rcnnSampler rcnn_batch_size
                                                    rpn_post_topk
                                                    num_sample
                                                    rcnn_fg_overlap
                                                    rcnn_fg_fraction
                                                    rcnn_max_num_gt
                                                    roi_boxes
                                                    rois_scores
                                                    gt_boxes

            roi_batchid <- prim _arange (#start := 0 .& #stop := Just (fromIntegral rcnn_batch_size) .& Nil)
            roi_batchid <- prim repeat  (#data := roi_batchid .& #repeats := num_sample .& Nil)
            roi_batchid <- reshape [-1, 1] roi_batchid

            rois <- reshape [-1, 1] rois
            rois <- concat_ 1 [roi_batchid, rois] >>= blockGrad

            feat <- alignROIs feats rois (stageList backbone) rcnn_pooled_size rcnn_feature_stride
            return (feat, samples, matches)

        topFeat <- features2 backbone feat_aligned

        unique "rcnn" $ do
            (labels, bbox_targets, bbox_weights) <- rcnnTargetGenerator rcnn_batch_size
                                                                        (rcnn_num_classes-1)
                                                                        (floor $ rcnn_fg_fraction * fromIntegral num_sample)
                                                                        samples
                                                                        matches
                                                                        roi_boxes
                                                                        gt_labels
                                                                        gt_boxes
                                                                        box_reg_mean
                                                                        box_reg_std
            rpn_cls_prob <- softmaxoutput (#data := rpn_raw_scores
                                        .& #label := labels
                                        .& #multi_output := True
                                        .& #normalization := #valid
                                        .& #use_ignore := True
                                        .& #ignore_label := -1 .& Nil)

            rpn_bbox_reg  <- sub_ rpn_raw_boxregs bbox_targets
            rpn_bbox_reg  <- prim smooth_l1 (#data := rpn_bbox_reg .& #scalar := 3.0 .& Nil)
            rpn_bbox_loss <- mul_ bbox_weights rpn_bbox_reg
            rpn_bbox_loss <- prim _MakeLoss
                                (#data := rpn_bbox_loss .& #grad_scale := 1.0 / fromIntegral rpn_batch_rois .& Nil)

            cls_score <- named "cls_score" $
                         fullyConnected (#data := topFeat .& #num_hidden := rcnn_num_classes .& Nil)
            cls_prob  <- named "cls_prob" $
                         softmaxoutput (#data := cls_score .& #label := labels .& #normalization := #batch .& Nil)

            ---------------------------
            -- bbox_loss part
            --
            bbox_pred <- named "bbox_pred" $
                         fullyConnected (#data := topFeat .& #num_hidden := 4 * rcnn_num_classes .& Nil)
            bbox_reg  <- sub_ bbox_pred bbox_targets
            bbox_reg  <- prim smooth_l1 (#data := bbox_reg .& #scalar := 1.0 .& Nil)
            bbox_loss <- mul_ bbox_reg bbox_weights
            bbox_loss <- named "bbox_loss" $
                         prim _MakeLoss (#data := bbox_loss .& #grad_scale := 1.0 / fromIntegral rcnn_batch_rois .& Nil)

            label     <- reshape [rcnn_batch_size, -1] labels >>= blockGrad
            cls_prob  <- reshape [rcnn_batch_size, -1, rcnn_num_classes] cls_prob
            bbox_loss <- reshape [rcnn_batch_size, -1, 4 * rcnn_num_classes] bbox_loss

            -- include topFeatures and clsScores for debug
            topFeatuSG <- named "topfeatu_sg" $ blockGrad topFeat
            clsScoreSG <- named "clsscore_sg" $ blockGrad cls_score
            -- roisSG     <- _BlockGrad "rois_sg"     (#data := rois.& Nil)
            -- bboxTSG    <- _BlockGrad "bboxT_sg"    (#data := bbox_targets .& Nil)
            -- bboxWSG    <- _BlockGrad "bboxW_sg"    (#data := bbox_weights .& Nil)

            group [rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, label, topFeatuSG, clsScoreSG]


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
data RPNAccMetric a = RPNAccMetric Int Text

instance EvalMetricMethod RPNAccMetric where
    data MetricData RPNAccMetric a = RPNAccMetricData Text Int Text (IORef Int) (IORef Int)
    newMetric phase (RPNAccMetric oindex label) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNAccMetricData phase oindex label a b

    formatMetric (RPNAccMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RPNAcc: " % fixed 2 % ">") (100 * fromIntegral s / fromIntegral n :: Float)

    evalMetric (RPNAccMetricData phase oindex lname cntRef sumRef) bindings outputs = liftIO $  do
        let label = bindings ^?! ix lname
            pred  = outputs  ^?! ix oindex

        pred <- A.makeNDArrayLike pred contextCPU >>= A.copy pred
        label <- V.convert <$> toVector label

        [pred_label] <- argmax_channel (#data := unNDArray pred .& Nil)
        pred_label <- V.convert <$> toVector (NDArray pred_label)

        let pairs = V.filter ((/= -1) . fst) $ V.zip label pred_label
            equal = V.filter (uncurry (==)) pairs

        modifyIORef' sumRef (+ length equal)
        modifyIORef' cntRef (+ length pairs)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = fromIntegral s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc


data RCNNAccMetric a = RCNNAccMetric Int Int

instance EvalMetricMethod RCNNAccMetric where
    data MetricData RCNNAccMetric a = RCNNAccMetricData {
        _rcnn_acc_phase :: Text,
        _rcnn_acc_cindex :: Int,
        _rcnn_acc_lindex :: Int,
        _rcnn_acc_all :: IORef (Int, Int),
        _rcnn_acc_fg  :: IORef (Int, Int)
    }
    newMetric phase (RCNNAccMetric cindex lindex) = do
        a <- liftIO $ newIORef (0, 0)
        b <- liftIO $ newIORef (0, 0)
        return $ RCNNAccMetricData phase cindex lindex a b

    formatMetric (RCNNAccMetricData _ _ _ accum_all accum_fg) = liftIO $ do
        (all_s, all_n) <- liftIO $ readIORef accum_all
        (fg_s, fg_n)   <- liftIO $ readIORef accum_fg
        return $ sformat ("<RCNNAcc: " % fixed 2 % " " % fixed 2 % ">")
            (100 * fromIntegral all_s / fromIntegral all_n :: Float)
            (100 * fromIntegral fg_s  / fromIntegral fg_n  :: Float)

    evalMetric rcnn_acc _ outputs = liftIO $  do
        -- cls_prob: (batch_size, #num_anchors*feat_w*feat_h, #num_classes)
        -- label:    (batch_size, #num_anchors*feat_w*feat_h)
        let cls_prob = outputs ^?! ix (_rcnn_acc_cindex rcnn_acc)
            label    = outputs ^?! ix (_rcnn_acc_lindex rcnn_acc)

        cls_prob <- A.makeNDArrayLike cls_prob contextCPU >>= A.copy cls_prob
        [pred_class] <- argmax (#data := unNDArray cls_prob .& #axis := Just 2 .& Nil)

        -- -- debug only
        -- s1 <- ndshape (NDArray pred_class :: NDArray Float)
        -- v1 <- toVector (NDArray pred_class :: NDArray Float)
        -- print (s1, SV.map floor v1 :: SV.Vector Int)

        -- s1 <- ndshape label
        -- v1 <- toVector label
        -- print (s1, SV.map floor v1 :: SV.Vector Int)

        pred_class <- toRepa @DIM2 (NDArray pred_class)
        label <- toRepa @DIM2 label

        let pairs_all = UV.zip (Repa.toUnboxed label) (Repa.toUnboxed pred_class)
            equal_all = UV.filter (uncurry (==)) pairs_all

            pairs_fg  = UV.filter ((>0) . fst) pairs_all
            equal_fg  = UV.filter (uncurry (==)) pairs_fg

        -- print (UV.map (bimap floor floor) pairs_fg :: UV.Vector (Int, Int))

        let ref_acc_all = _rcnn_acc_all rcnn_acc
            ref_acc_fg  = _rcnn_acc_fg  rcnn_acc
        modifyIORef' ref_acc_all (bimap (+ UV.length equal_all) (+ UV.length pairs_all))
        modifyIORef' ref_acc_fg  (bimap (+ UV.length equal_fg)  (+ UV.length pairs_fg))

        (all_s, all_n) <- readIORef ref_acc_all
        (fg_s,  fg_n)  <- readIORef ref_acc_fg
        let all_acc = fromIntegral all_s / fromIntegral all_n
            fg_acc  = fromIntegral fg_s  / fromIntegral fg_n
            phase   = _rcnn_acc_phase rcnn_acc
        return $ M.fromList [
            (phase `T.append` "_with_bg_acc", all_acc),
            (phase `T.append` "_fg_only_acc", fg_acc)]

data RPNLogLossMetric a = RPNLogLossMetric Int Text

instance EvalMetricMethod RPNLogLossMetric where
    data MetricData RPNLogLossMetric a = RPNLogLossMetricData Text Int Text (IORef Int) (IORef Double)
    newMetric phase (RPNLogLossMetric cindex lname) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNLogLossMetricData phase cindex lname a b

    formatMetric (RPNLogLossMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RPNLogLoss: " % fixed 4 % ">") (realToFrac s / fromIntegral n :: Float)

    evalMetric (RPNLogLossMetricData phase cindex lname cntRef sumRef) bindings outputs = liftIO $  do
        let cls_prob = outputs  ^?! ix cindex
            label    = bindings ^?! ix lname

        -- (batch_size, #num_anchors*feat_w*feat_h) to (batch_size*#num_anchors*feat_w*feat_h,)
        label <- A.reshape label [-1]
        label <- toRepa @DIM1 label
        let Z :. size = Repa.extent label

        -- (batch_size, #channel, #num_anchors*feat_w, feat_h) to (batch_size, #channel, #num_anchors*feat_w*feat_h)
        -- to (batch_size, #num_anchors*feat_w*feat_h, #channel) to (batch_size*#num_anchors*feat_w*feat_h, #channel)
        cls_prob <- A.makeNDArrayLike cls_prob contextCPU >>= A.copy cls_prob
        pred  <- A.reshape cls_prob [0, 0, -1] >>= flip A.transpose [0, 2, 1] >>= flip A.reshape [size, -1]
        pred  <- toRepa @DIM2 pred

        -- mark out labels where value -1
        let mask = Repa.computeUnboxedS $ Repa.map (/= -1) label

        pred  <- Repa.selectP
                    (mask ^#!)
                    (\i -> let cls = floor (label ^#! i)
                           in pred ^?! ixr (Z :. i :. cls))
                    size

        let pred_with_ep = Repa.map ((0 -) . log)  (pred Repa.+^ constant (Z :. size) 1e-14)
        cls_loss_val <- realToFrac <$> Repa.sumAllP pred_with_ep
        modifyIORef' sumRef (+ cls_loss_val)
        modifyIORef' cntRef (+ size)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc

data RCNNLogLossMetric a = RCNNLogLossMetric Int Int

instance EvalMetricMethod RCNNLogLossMetric where
    data MetricData RCNNLogLossMetric a = RCNNLogLossMetricData Text Int Int (IORef Int) (IORef Double)
    newMetric phase (RCNNLogLossMetric cindex lindex) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RCNNLogLossMetricData phase cindex lindex a b

    formatMetric (RCNNLogLossMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RCNNLogLoss: " % fixed 4 % ">") (realToFrac s / fromIntegral n :: Float)

    evalMetric (RCNNLogLossMetricData phase cindex lindex cntRef sumRef) _ outputs = liftIO $  do
        cls_prob <- toRepa @DIM3 (outputs ^?! ix cindex)
        label    <- toRepa @DIM2 (outputs ^?! ix lindex)

        let lbl_shp@(Z :. _ :. num_lbl) = Repa.extent label
            cls = Repa.fromFunction lbl_shp (\ pos@(Z :. bi :. ai) ->
                    cls_prob Repa.! (Z :. bi :. ai :. (floor $ label Repa.! pos)))

        cls_loss_val <- Repa.sumAllP $ Repa.map (\v -> - log(1e-14 + v)) cls
        -- traceShowM cls_loss_val
        modifyIORef' sumRef (+ realToFrac cls_loss_val)
        modifyIORef' cntRef (+ num_lbl)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc

data RPNL1LossMetric a = RPNL1LossMetric Int Text

instance EvalMetricMethod RPNL1LossMetric where
    data MetricData RPNL1LossMetric a = RPNL1LossMetricData Text Int Text (IORef Int) (IORef Double)
    newMetric phase (RPNL1LossMetric bindex blabel) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RPNL1LossMetricData phase bindex blabel a b

    formatMetric (RPNL1LossMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RPNL1Loss: " % fixed 3 % ">") (realToFrac s / fromIntegral n :: Float)

    evalMetric (RPNL1LossMetricData phase bindex blabel cntRef sumRef) bindings outputs = liftIO $  do
        bbox_loss   <- toRepa @DIM4 (outputs ^?! ix bindex)
        all_loss    <- Repa.sumAllP bbox_loss

        bbox_weight <- toRepa @DIM4 (bindings ^?! ix blabel)
        all_pos_weight <- Repa.sumAllP $ Repa.map (\w -> if w > 0 then 1 else 0) bbox_weight

        modifyIORef' sumRef (+ realToFrac all_loss)
        modifyIORef' cntRef (+ (all_pos_weight `div` 4))

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc

data RCNNL1LossMetric a = RCNNL1LossMetric Int Int

instance EvalMetricMethod RCNNL1LossMetric where
    data MetricData RCNNL1LossMetric a = RCNNL1LossMetricData Text Int Int (IORef Int) (IORef Double)
    newMetric phase (RCNNL1LossMetric bindex lindex) = do
        a <- liftIO $ newIORef 0
        b <- liftIO $ newIORef 0
        return $ RCNNL1LossMetricData phase bindex lindex a b

    formatMetric (RCNNL1LossMetricData _ _ _ cntRef sumRef) = liftIO $ do
        s <- liftIO $ readIORef sumRef
        n <- liftIO $ readIORef cntRef
        return $ sformat ("<RCNNL1Loss: " % fixed 3 % ">") (realToFrac s / fromIntegral n :: Float)

    evalMetric (RCNNL1LossMetricData phase bindex lindex cntRef sumRef) _ outputs = liftIO $ do
        bbox_loss <- toRepa @DIM3 (outputs ^?! ix bindex)
        all_loss  <- Repa.sumAllP bbox_loss

        label     <- toRepa @DIM2 (outputs ^?! ix lindex)
        all_pos   <- Repa.sumAllP $ Repa.map (\w -> if w > 0 then 1 else 0) label

        modifyIORef' sumRef (+ realToFrac all_loss)
        modifyIORef' cntRef (+ all_pos)

        s <- readIORef sumRef
        n <- readIORef cntRef
        let acc = s / fromIntegral n
        return $ M.singleton (phase `T.append` "_acc") acc

constant :: (Shape sh, UV.Unbox a) => sh -> a -> Repa.Array Repa.U sh a
constant shp val = Repa.fromListUnboxed shp (replicate (size shp) val)
