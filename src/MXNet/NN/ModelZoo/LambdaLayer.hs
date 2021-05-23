{-# LANGUAGE ExplicitForAll      #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.NN.ModelZoo.LambdaLayer where

import           RIO

import           Fei.Einops
import           MXNet.Base
import           MXNet.NN.Initializer
import           MXNet.NN.Layer

-- | The LambdaNetworks' layer
lambdaLayer :: forall a . (FloatDType a, InEnum (DTypeName a) BasicFloatDTypes)
            => Int -- ^ output dimension
            -> Int -- ^ key dimension
            -> Int -- ^ intra-depth dimension
            -> Int -- ^ number of heads
            -> Int -- ^ receptive window, max of height and width of the input feature
            -> Symbol a -- ^ input feature
            -> Layer (Symbol a)
lambdaLayer dim_out dim_k dim_u heads window x
  | dim_out `mod` heads /= 0 = error "dim_out must be divisible by heads"
  | otherwise = do
      let dim_v  = dim_out `div` heads
      _q <- convolution (#data := x .& #kernel := [1,1] .& #num_filter := dim_k * heads .& #no_bias := True .& Nil)
      _k <- convolution (#data := x .& #kernel := [1,1] .& #num_filter := dim_k * dim_u .& #no_bias := True .& Nil)
      _v <- convolution (#data := x .& #kernel := [1,1] .& #num_filter := dim_v * dim_u .& #no_bias := True .& Nil)

      _q <- sequential "_q" $ rearrange _q "b (h k) hh ww -> b h k (hh ww)" [#h .== heads]
      _k <- sequential "_k" $ rearrange _k "b (u k) hh ww -> b u k (hh ww)" [#u .== dim_u]
      _v <- sequential "_v" $ rearrange _v "b (u v) hh ww -> b u v (hh ww)" [#u .== dim_u, #v .== dim_v]

      _k <- softmax (#data := _k .& #axis := (-1) .& Nil)

      λc <- einsum "b u k m, b u v m -> b k v" [_k, _v] False
      υc <- einsum "b h k n, b k v -> b h v n" [_q, λc] False

      rel_pos_emb <- parameter "rel_pos_emb" ReqWrite (Just [2 * window - 1, 2 * window - 1, dim_k, dim_u])
                     >>= initWith (InitNormal 1.0)
      rel_pos     <- calcRelPos @a window
      rel_pos_lku <- gather rel_pos_emb rel_pos

      λp <- einsum "n m k u, b u v m -> b n k v" [rel_pos_lku, _v] False
      υp <- einsum "b h k n, b n k v -> b h v n" [_q, λp] False

      _υ <- add_ υc υp
      _υ <- rearrange _υ "b h v hhww -> b (h v) hhww" []
      reshapeLike _υ (Just (-1)) Nothing x (Just (-2)) Nothing

-- | Calculate the relative distance of every pair of points on a grid of size 'n', (2, n*n, n*n)
calcRelPos :: forall a t . (HasCallStack, PrimTensorOp t, MXTensor t, FloatDType a, InEnum (DTypeName a) BasicFloatDTypes)
           => Int -> TensorMonad t (t a)
calcRelPos n = do
    r <- arangeF Proxy 0 (Just $ fromIntegral n) Nothing
    grid_x <- expandDims 1 r >>= broadcastAxis [1] [n]
    grid_y <- expandDims 0 r >>= broadcastAxis [0] [n]
    mesh   <- stack 0 [grid_x, grid_y]
    mesh   <- rearrange mesh "n i j -> n (i j)" []
    ypos   <- expandDims 1 mesh
    xpos   <- expandDims 2 mesh
    sub_ ypos xpos >>= addScalar (fromIntegral $ n - 1)
