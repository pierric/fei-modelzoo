{-# LANGUAGE ViewPatterns #-}
module MXNet.NN.ModelZoo.VGG where

import           RIO
import           RIO.List       (scanl, zip3)

import           MXNet.Base
import           MXNet.NN.Layer

{-
VGG(
  (features): HybridSequential(
    (0): Conv2D(3 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Activation(relu)
    (2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Activation(relu)
    (4): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (5): Conv2D(64 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): Activation(relu)
    (7): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): Activation(relu)
    (9): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (10): Conv2D(128 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): Activation(relu)
    (12): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): Activation(relu)
    (14): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): Activation(relu)
    (16): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (17): Conv2D(256 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): Activation(relu)
    (19): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): Activation(relu)
    (21): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): Activation(relu)
    (23): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (24): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): Activation(relu)
    (26): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): Activation(relu)
    (28): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): Activation(relu)
    ** (30): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (31): Dense(25088 -> 4096, Activation(relu))
    (32): Dropout(p = 0.5, axes=())
    (33): Dense(4096 -> 4096, Activation(relu))
    (34): Dropout(p = 0.5, axes=())
  )
  (output): Dense(4096 -> 1000, linear)
)

** It appears only if `with_last_pooling` is True.
 -}


getFeature :: SymbolHandle -> [Int] -> [Int] -> Bool -> Bool -> Layer SymbolHandle
getFeature dat layers filters with_batch_norm with_last_pooling = do
    sym <- foldM build1 dat specs
    -- inlining the build1 below, and omit pooling depending on the with_last_pooling
    case last_group of
        (idx, num, filter) -> do
            sym <- foldM build2 sym $ zip [idx..] $ replicate num filter
            if not with_last_pooling
            then return sym
            else pooling (#data := sym
                       .& #pool_type := #max
                       .& #kernel := [2,2]
                       .& #stride := [2,2] .& Nil)

  where
    idxes = scanl (+) 0 layers
    last_group:groups = reverse $ zip3 idxes layers filters
    specs = reverse groups

    build1 sym (idx, num, filter) = do
        sym <- foldM build2 sym $ zip [idx..] $ replicate num filter
        pooling (#data := sym
              .& #pool_type := #max
              .& #kernel := [2,2]
              .& #stride := [2,2] .& Nil)

    build2 sym (idx, filter) = do
        sym <- convolution (#data := sym
                         .& #kernel := [3,3]
                         .& #pad := [1,1]
                         .& #num_filter := filter
                         .& #workspace := 2048 .& Nil)
        sym <- if with_batch_norm
                  then batchnorm (#data := sym .& Nil)
                  else return sym
        activation (#data := sym .& #act_type := #relu .& Nil)

getTopFeature :: SymbolHandle -> Layer SymbolHandle
getTopFeature input = do
    sym <- unique' $ flatten input
    sym <- fullyConnected (#data := sym .& #num_hidden := 4096 .& Nil)
    -- sym <- activation (#data := sym .& #act_type := #relu .& Nil)
    sym <- dropout sym 0.5
    sym <- fullyConnected (#data := sym .& #num_hidden := 4096 .& Nil)
    -- sym <- activation (#data := sym .& #act_type := #relu .& Nil)
    dropout sym 0.5

symbol :: SymbolHandle -> Int -> Bool -> Layer SymbolHandle
symbol dat num_layers with_batch_norm =
    getFeature dat layers filters with_batch_norm True >>= getTopFeature
  where
    (layers, filters) = case num_layers of
                            11 -> ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512])
                            13 -> ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512])
                            16 -> ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
                            19 -> ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])

vgg16 dat num_classes = do
    sym <- sequential "features" $ symbol dat 16 False
    named "output" $ fullyConnected (#data := sym .& #num_hidden := num_classes .& Nil)
