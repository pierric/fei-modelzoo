module MXNet.NN.ModelZoo.RCNN.MaskRCNN where

import           RIO

import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor       as T
import           MXNet.NN.Layer
import qualified MXNet.NN.ModelZoo.RCNN.FasterRCNN as FasterRCNN
import           MXNet.NN.ModelZoo.RCNN.RCNN


data MaskRCNN = MaskRCNN
    { _faster_rcnn_result :: FasterRCNN.FasterRCNN
    , _masks_loss         :: SymbolHandle
    }
    | MaskRCNNInferenceOnly
    { _faster_rcnn_result :: FasterRCNN.FasterRCNN
    , _masks              :: SymbolHandle
    }

maskHead :: SymbolHandle -> Int -> Int -> Int -> Int -> Layer SymbolHandle
maskHead top_feat num_fcn_conv num_fg_classes batch_size num_mask_channels = do
    -- top_feat: The network input tensor of shape (B * N, fC, fH, fW).
    --
    -- returns:
    --   Mask prediction of shape (B, N, C, MS, MS)
    feat <- sequential "conv" $ foldM one_conv top_feat ([1..num_fcn_conv] :: [_])
    feat <- named "conv-transposed" $
            prim T._Deconvolution (#data := feat
                                .& #num_filter := num_mask_channels
                                .& #kernel := [2, 2]
                                .& #stride := [2, 2]
                                .& #pad := [0, 0] .& Nil)
    feat <- activation  (#data := feat .& #act_type := #relu .& Nil)
    mask <- named "conv-last" $
            convolution (#data := feat
                      .& #kernel := [1, 1]
                      .& #num_filter := num_fg_classes
                      .& #stride := [1, 1]
                      .& #pad := [0, 0] .& Nil)
    reshape [-4, batch_size, -1, 0, 0, 0] mask

    where
        one_conv x _ = do
            x <- convolution (#data := x
                           .& #num_filter := num_mask_channels
                           .& #kernel := [3, 3]
                           .& #stride := [1, 1]
                           .& #pad := [1, 1] .& Nil)
            activation (#data := x .& #act_type := #relu .& Nil)


graphT :: FasterRCNN.RcnnConfiguration -> Layer (MaskRCNN, SymbolHandle)
graphT conf@(FasterRCNN.RcnnConfigurationTrain{..}) = do
    gt_masks <- variable "gt_masks"
    (fr@FasterRCNN.FasterRCNN{..}, fr_outputs) <- FasterRCNN.graphT conf
    unique "mask" $ do
        let take_pos t = do
                ts <- forM ([0..batch_size-1] :: [_]) $ \i -> do
                  ind <- slice_axis _positive_indices 0 i (Just (i+1)) >>= squeeze Nothing
                  bat <- slice_axis t 0 i (Just (i+1)) >>= squeeze Nothing
                  takeI ind bat
                concat_ 0 ts

        -- like box_feature, we select only layers of the feature/... that have
        -- foreground gt, for each example in the batch.
        -- positive_indices: (B, rcnn_fg_fraction * rcnn_batch_rois)
        -- _top_feature:     (B * rcnn_batch_rois, num_channels, rcnn_pooled_size, rcnn_pooled_size)
        --                                           => (B * rcnn_fg_fraction * rcnn_batch_rois, .., .., ..)
        -- _roi_boxes:       (B, rcnn_batch_rois, 4) => (B, rcnn_fg_fraction * rcnn_batch_rois, 4)
        -- _gt_matches:      (B, rcnn_batch_rois)    => (B, rcnn_fg_fraction * rcnn_batch_rois)
        -- _cls_targets:     (B, rcnn_batch_rois)    => (B, rcnn_fg_fraction * rcnn_batch_rois)
        feature     <- take_pos =<< reshape [batch_size, -1, 0, 0, 0] =<< expandDims 0 _top_feature
        roi_boxes   <- reshape [batch_size, -1, 4] =<< take_pos _roi_boxes
        gt_matches  <- reshape [batch_size, -1]    =<< take_pos _gt_matches
        cls_targets <- reshape [batch_size, -1]    =<< take_pos _cls_targets

        let num_fcn_conv = case backbone of
                             FasterRCNN.RESNET50FPN -> 4
                             _                      -> 0
            num_fg_classes = rcnn_num_classes-1
            -- mask_size should be twice the final feature size
            -- becuase there is only one Conv2DTranspose layer
            mask_size = rcnn_pooled_size * 2
        masks <- unique "mask_head" $ maskHead feature num_fcn_conv num_fg_classes batch_size 256

        (mask_targets, mask_weights) <- unique "target_gen" $
                                        maskTargetGenerator batch_size
                                                            num_fg_classes
                                                            mask_size
                                                            gt_masks
                                                            roi_boxes
                                                            gt_matches
                                                            cls_targets
        masks_loss <- unique "loss" $ do
            masks_loss   <- sigmoidBCE masks mask_targets (Just mask_weights) AggSum
            num_pos_avg  <- sum_ mask_weights Nothing False >>= divScalar (fromIntegral batch_size) >>= addScalar 1e-14
            masks_loss   <- divBroadcast masks_loss num_pos_avg
            prim T._MakeLoss (#data := masks_loss .& #grad_scale := 1.0 .& Nil)

        result_sym <- group $ [fr_outputs, masks_loss]
        return $ (MaskRCNN {
            _faster_rcnn_result = fr,
            _masks_loss = masks_loss
        }, result_sym)

graphI :: FasterRCNN.RcnnConfiguration -> Layer (MaskRCNN, SymbolHandle)
graphI conf@(FasterRCNN.RcnnConfigurationInference{..}) = do
    (fr@FasterRCNN.FasterRCNNInferenceOnly{..}, fr_outputs) <- FasterRCNN.graphI conf
    feature <- reshape [batch_size, -1, 0, 0, 0] =<< expandDims 0 _top_feature
    let num_fcn_conv = case backbone of
                         FasterRCNN.RESNET50FPN -> 4
                         _                      -> 0
        num_fg_classes = rcnn_num_classes-1
    masks <- unique "mask_head" $ maskHead feature num_fcn_conv num_fg_classes batch_size 256
    masks <- prim T._sigmoid (#data := masks .& Nil)
    res_sym <- group [fr_outputs, masks]
    let res_data = MaskRCNNInferenceOnly fr masks
    return (res_data, res_sym)
