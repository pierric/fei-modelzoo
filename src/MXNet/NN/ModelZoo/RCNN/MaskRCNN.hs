module MXNet.NN.ModelZoo.RCNN.MaskRCNN where

import           RIO

import           Fei.Einops
import           MXNet.Base
import qualified MXNet.Base.Operators.Tensor       as T
import           MXNet.NN.Layer
import qualified MXNet.NN.ModelZoo.RCNN.FasterRCNN as FasterRCNN
import           MXNet.NN.ModelZoo.RCNN.RCNN


data MaskRCNN a
  = MaskRCNN
      { _faster_rcnn_result :: FasterRCNN.FasterRCNN a
      , _masks_loss         :: Symbol a
      }
  | MaskRCNNInferenceOnly
      { _faster_rcnn_result :: FasterRCNN.FasterRCNN a
      , _masks              :: Symbol a
      }

maskHead :: NumericDType a => Symbol a -> Int -> Int -> Int -> Int -> Layer (Symbol a)
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
    rearrange mask "(b n) c h w -> b n c h w" [#b .== batch_size]

    where
        one_conv x _ = do
            x <- convolution (#data := x
                           .& #num_filter := num_mask_channels
                           .& #kernel := [3, 3]
                           .& #stride := [1, 1]
                           .& #pad := [1, 1] .& Nil)
            unique' $ activation (#data := x .& #act_type := #relu .& Nil)


graphT :: NumericDType a
       => FasterRCNN.RcnnConfiguration -> Layer (MaskRCNN a, Symbol a)
graphT conf@(FasterRCNN.RcnnConfigurationTrain{..}) = do
    gt_masks <- variable "gt_masks"
    (fr@FasterRCNN.FasterRCNN{..}, fr_outputs) <- FasterRCNN.graphT conf
    unique "mask" $ do
        let take_pos t = do
                ts <- forM ([0..batch_size-1] :: [_]) $ \i -> do
                  ind <- sliceAxis _positive_indices 0 i (Just (i+1)) >>= squeeze Nothing
                  bat <- sliceAxis t 0 i (Just (i+1)) >>= squeeze Nothing
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
        feature <- rearrange _top_feature "(b n) c h w -> b n c h w" [#b .== batch_size]
        feature <- take_pos feature
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
        masks <- named "mask" $ identity masks
        mask_targets <- named "mask-t" $ identity mask_targets
        mask_weights <- named "mask-w" $ identity mask_weights
        unique "loss" $ do
            masks_loss  <- sigmoidBCE masks mask_targets (Just mask_weights) AggSum
            masks_loss  <- sum_ masks_loss Nothing False
            num_pos_avg <- sum_ mask_weights Nothing False
                            >>= divScalar (fromIntegral batch_size)
                            >>= addScalar 1e-14
                            >>= blockGrad
            -- makeLoss cannot take scalar value (empty shape) as loss, so we
            -- reshape it to (1,)
            masks_loss  <- div_ masks_loss num_pos_avg >>= reshape [1]
            masks_loss  <- makeLoss masks_loss 1.0

            result_sym <- group $ [fr_outputs, masks_loss]
            return $ (MaskRCNN {
                _faster_rcnn_result = fr,
                _masks_loss = masks_loss
            }, result_sym)

graphI :: NumericDType a
       => FasterRCNN.RcnnConfiguration -> Layer (MaskRCNN a, Symbol a)
graphI conf@(FasterRCNN.RcnnConfigurationInference{..}) = do
    (fr@FasterRCNN.FasterRCNNInferenceOnly{..}, fr_outputs) <- FasterRCNN.graphI conf
    feature <- rearrange _top_feature "(b n) c h w -> b n c h w" [#b .== batch_size]
    let num_fcn_conv = case backbone of
                         FasterRCNN.RESNET50FPN -> 4
                         _                      -> 0
        num_fg_classes = rcnn_num_classes-1
    masks <- unique "mask_head" $ maskHead feature num_fcn_conv num_fg_classes batch_size 256
    masks <- prim T._sigmoid (#data := masks .& Nil)
    res_sym <- group [fr_outputs, masks]
    let res_data = MaskRCNNInferenceOnly fr masks
    return (res_data, res_sym)
