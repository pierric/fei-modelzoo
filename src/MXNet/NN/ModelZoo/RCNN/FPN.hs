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

    outputs <- liftIO $ newIORef []
    foldM_ (topDownPass outputs) Nothing (NE.zip layer_filters layers)
    liftIO $ readIORef outputs

  where
    (layer_names, layer_filters) = NE.unzip $ NE.reverse output_layers
    topDownPass outputs Nothing (nflt, layer) = do
        y <- convolution (#data := layer
                       .& #num_filter := nflt
                       .& #kernel := [1,1]
                       .& #pad := [0, 0]
                       .& #stride := [1,1]
                       .& #no_bias := True .& Nil)
        y <- batchnorm   (#data := y .& Nil)
        y <- convolution (#data := y
                       .& #num_filter := nflt
                       .& #kernel := [3, 3]
                       .& #pad := [1, 1]
                       .& #stride := [1,1]
                       .& #no_bias := True .& Nil)
        y <- batchnorm   (#data := y .& Nil)
        modifyIORef outputs (y NE.<|)
        return (Just y)
    topDownPass outputs (Just prev) (nflt, layer) = do
        y <- convolution (#data := layer
                       .& #num_filter := nflt
                       .& #kernel := [1,1]
                       .& #pad := [0, 0]
                       .& #stride := [1,1]
                       .& #no_bias := True .& Nil)
        y <- batchnorm   (#data := y .& Nil)
        y <- prim _UpSampling
                         (#data := [y]
                       .& #num_args := 1
                       .& #scale := 2
                       .& #sample_type := #nearest .& Nil)
        y <- add_ prev y
        y <- convolution (#data := y
                       .& #num_filter := nflt
                       .& #kernel := [3, 3]
                       .& #pad := [1, 1]
                       .& #stride := [1,1]
                       .& #no_bias := True .& Nil)
        y <- batchnorm   (#data := y .& Nil)
        modifyIORef outputs (y NE.<|)
        return (Just y)
