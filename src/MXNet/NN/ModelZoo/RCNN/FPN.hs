module MXNet.NN.ModelZoo.RCNN.FPN where

import           RIO
import qualified RIO.NonEmpty                as NE (reverse, unzip, zip, (<|))

import           MXNet.Base                  (ArgOf (..), HMap (..),
                                              SymbolHandle, at', internals,
                                              (.&))
import           MXNet.Base.Operators.Symbol (_UpSampling)
import           MXNet.NN.Layer

-- TODO
-- no_bias ?
-- batchnorm args ?

fpnFeatureExpander :: SymbolHandle -> NonEmpty (Text, Int) -> Layer (NonEmpty SymbolHandle)
fpnFeatureExpander sym output_layers = do
    sym <- internals sym
    layers <- mapM (at' sym) layer_names

    outputs <- liftIO $ newIORef (error "empty")
    sequential "fpn" $ do
        foldM_ (topDownPass outputs) Nothing (NE.zip layer_filters layers)
    liftIO $ readIORef outputs

  where
    (layer_names, layer_filters) = NE.unzip $ NE.reverse output_layers
    topDownPass outputs Nothing (nflt, layer) = subscope_next_name $ unique' $ do
        y <- named "conv1" $
             convolution (#data := layer
                       .& #num_filter := nflt
                       .& #kernel := [1,1]
                       .& #pad := [0, 0]
                       .& #stride := [1,1]
                       .& #no_bias := True .& Nil)
        y <- named "bn0" $
             batchnorm   (#data := y .& Nil)
        out <- named "conv2" $
               convolution (#data := y
                         .& #num_filter := nflt
                         .& #kernel := [3, 3]
                         .& #pad := [1, 1]
                         .& #stride := [1,1]
                         .& #no_bias := True .& Nil)
        out <- named "bn1" $
               batchnorm   (#data := out .& Nil)
        writeIORef outputs [out]
        return (Just y)
    topDownPass outputs (Just prev) (nflt, layer) = subscope_next_name $ unique' $ do
        y <- named "conv1" $
             convolution (#data := layer
                       .& #num_filter := nflt
                       .& #kernel := [1,1]
                       .& #pad := [0, 0]
                       .& #stride := [1,1]
                       .& #no_bias := True .& Nil)
        y <- named "bn0" $
             batchnorm   (#data := y .& Nil)
        prev_up <- prim _UpSampling
                         (#data := [prev]
                       .& #num_args := 1
                       .& #scale := 2
                       .& #sample_type := #nearest .& Nil)
        y <- add_ prev_up y
        out <- named "conv2" $
               convolution (#data := y
                         .& #num_filter := nflt
                         .& #kernel := [3, 3]
                         .& #pad := [1, 1]
                         .& #stride := [1,1]
                         .& #no_bias := True .& Nil)
        out <- named "bn1" $
               batchnorm   (#data := out .& Nil)
        modifyIORef outputs (out NE.<|)
        return (Just y)
