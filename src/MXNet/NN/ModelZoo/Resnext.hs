module MXNet.NN.ModelZoo.Resnext where

import           Formatting
import           RIO

import           MXNet.Base
import           MXNet.NN.Layer

-- ResNet
-- #layer: 164
-- #stage: 3
-- #layer per stage: 18
-- #filter of stage 1: 64
-- #filter of stage 2: 128
-- #filter of stage 3: 256

rootName = "resnext0"

symbol :: DType a => (Symbol a) -> Layer (Symbol a)
symbol dat = unique rootName $ do
    bnx <- named "batchnorm0" $
           batchnorm (#data := dat
                   .& #eps := eps
                   .& #momentum := bn_mom
                   .& #fix_gamma := True .& Nil)

    cvx <- named "conv0" $
           convolution (#data := bnx
                     .& #kernel := [3,3]
                     .& #num_filter := 16
                     .& #stride := [1,1]
                     .& #pad := [1,1]
                     .& #workspace := conv_workspace
                     .& #no_bias := True .& Nil)

    bdy <- foldM (\layer (num_filter, stride, dim_match, stage_id, unit_id) ->
                    unique (sformat ("stage" % int) stage_id) $ residual unit_id
                        (#data       := layer
                      .& #num_filter := num_filter
                      .& #stride     := stride
                      .& #dim_match  := dim_match .& resargs))
                 cvx
                 residual'parms

    pool1 <- pooling (#data := bdy
                   .& #kernel := [7,7]
                   .& #pool_type := #avg
                   .& #global_pool := True .& Nil)
    flat <- flatten pool1
    named "dense0" $ fullyConnected (#data := flat
                                  .& #num_hidden := 10 .& Nil)
  where
    bn_mom = 0.9 :: Float
    conv_workspace = 256 :: Int
    eps = 2e-5 :: Double
    residual'parms =
        [(64,  [1,1], False, 1::Int, 1::Int)] ++ [(64,  [1,1], True, 1, i) | i <- [2..18]]
     ++ [(128, [2,2], False, 2, 1)] ++ [(128, [1,1], True, 2, i) | i <- [2..18]]
     ++ [(256, [2,2], False, 3, 1)] ++ [(256, [1,1], True, 3, i) | i <- [2..18]]
    resargs = #bottle_neck := True .& #workspace := conv_workspace .& #memonger := False .& Nil

type instance ParameterList "_residual_layer(resnext)" t =
  '[ '("data"       , 'AttrReq t)
   , '("num_filter" , 'AttrReq Int)
   , '("stride"     , 'AttrReq [Int])
   , '("dim_match"  , 'AttrReq Bool)
   , '("bottle_neck", 'AttrOpt Bool)
   , '("num_group"  , 'AttrOpt Int)
   , '("bn_mom"     , 'AttrOpt Float)
   , '("workspace"  , 'AttrOpt Int)
   , '("memonger"   , 'AttrOpt Bool) ]
residual :: (Fullfilled "_residual_layer(resnext)" (Symbol a) args, DType a)
         => Int -> ArgsHMap "_residual_layer(resnext)" (Symbol a) args
         -> Layer (Symbol a)
residual _id args = do
    let dat        = args ! #data
        num_filter = args ! #num_filter
        stride     = args ! #stride
        dim_match  = args ! #dim_match
        bottle_neck= fromMaybe True $ args !? #bottle_neck
        num_group  = fromMaybe 32   $ args !? #num_group
        bn_mom     = fromMaybe 0.9  $ args !? #bn_mom
        workspace  = fromMaybe 256  $ args !? #workspace
        memonger   = fromMaybe False$ args !? #memonger
        eps = 2e-5 :: Double
    if bottle_neck
    then do
        conv1 <- named (sformat ("conv" % int) _id) $
                 convolution (#data      := dat
                           .& #kernel    := [1,1]
                           .& #num_filter:= num_filter `div` 2
                           .& #stride    := [1,1]
                           .& #pad       := [0,0]
                           .& #workspace := workspace
                           .& #no_bias   := True .& Nil)
        bn1   <- named (sformat ("batchnorm" % int) _id) $
                 batchnorm (#data      := conv1
                         .& #eps       := eps
                         .& #momentum  := bn_mom
                         .& #fix_gamma := False .& Nil)
        act1  <- activation (#data      := bn1
                          .& #act_type  := #relu .& Nil)
        conv2 <- named (sformat ("conv" % int) (_id + 1)) $
                 convolution (#data      := act1
                           .& #kernel    := [3,3]
                           .& #num_filter:= num_filter `div` 2
                           .& #stride    := stride
                           .& #pad       := [1,1]
                           .& #num_group := num_group
                           .& #workspace := workspace
                           .& #no_bias   := True .& Nil)
        bn2   <- named (sformat ("batchnorm" % int) (_id + 1)) $
                 batchnorm (#data      := conv2
                         .& #eps       := eps
                         .& #momentum  := bn_mom
                         .& #fix_gamma := False .& Nil)
        act2  <- activation (#data      := bn2
                          .& #act_type  := #relu .& Nil)
        conv3 <- named (sformat ("conv" % int) (_id + 2)) $
                 convolution (#data      := act2
                           .& #kernel    := [1,1]
                           .& #num_filter:= num_filter
                           .& #stride    := [1,1]
                           .& #pad       := [0,0]
                           .& #workspace := workspace
                           .& #no_bias   := True .& Nil)
        bn3   <- named (sformat ("batchnorm" % int) (_id + 2)) $
                 batchnorm (#data      := conv3
                         .& #eps       := eps
                         .& #momentum  := bn_mom
                         .& #fix_gamma := False .& Nil)
        shortcut <-
            if dim_match
            then return dat
            else do
                shortcut_conv <- named (sformat ("conv" % int) (_id + 3)) $
                                 convolution (#data        := dat
                                           .& #kernel      := [1,1]
                                           .& #num_filter  := num_filter
                                           .& #stride      := stride
                                           .& #workspace   := workspace
                                           .& #no_bias     := True .& Nil)
                named (sformat ("conv" % int) (_id + 3)) $
                    batchnorm (#data        := shortcut_conv
                            .& #eps         := eps
                            .& #momentum    := bn_mom
                            .& #fix_gamma   := False .& Nil)
        when memonger $
          void $ setAttr shortcut "mirror_stage" "true"
        eltwise <- add_ bn3 shortcut
        activation (#data := eltwise .& #act_type := #relu .& Nil)
    else do
        conv1 <- named (sformat ("conv" % int) _id) $
                 convolution (#data        := dat
                           .& #kernel      := [3,3]
                           .& #num_filter  := num_filter
                           .& #stride      := stride
                           .& #pad         := [1,1]
                           .& #workspace   := workspace
                           .& #no_bias     := True .& Nil)
        bn1   <- named (sformat ("batchnorm" % int) _id) $
                 batchnorm (#data        := conv1
                         .& #eps         := eps
                         .& #momentum    := bn_mom
                         .& #fix_gamma   := False .& Nil)
        act1  <- activation (#data        := bn1
                          .& #act_type    := #relu .& Nil)
        conv2 <- named (sformat ("conv" % int) (_id + 1)) $
                 convolution (#data        := act1
                           .& #kernel      := [3,3]
                           .& #num_filter  := num_filter
                           .& #stride      := [1,1]
                           .& #pad         := [1,1]
                           .& #workspace   := workspace
                           .& #no_bias     := True .& Nil)
        bn2   <- named (sformat ("batchnorm" % int) (_id + 1)) $
                 batchnorm (#data        := conv2
                         .& #eps         := eps
                         .& #momentum    := bn_mom
                         .& #fix_gamma   := False .& Nil)
        shortcut <-
            if dim_match
            then return dat
            else do
                shortcut_conv <- named (sformat ("conv" % int) (_id + 2)) $
                                 convolution (#data        := act1
                                           .& #kernel      := [1,1]
                                           .& #num_filter  := num_filter
                                           .& #stride      := stride
                                           .& #workspace   := workspace
                                           .& #no_bias     := True .& Nil)
                named (sformat ("batchnorm" % int) (_id + 2)) $
                    batchnorm (#data        := shortcut_conv
                            .& #eps         := eps
                            .& #momentum    := bn_mom
                            .& #fix_gamma   := False .& Nil)
        when memonger $
          void $ setAttr shortcut "mirror_stage" "true"
        eltwise <- add_ bn2 shortcut
        activation (#data := eltwise .& #act_type := #relu .& Nil)
