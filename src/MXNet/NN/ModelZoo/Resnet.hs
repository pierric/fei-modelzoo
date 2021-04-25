module MXNet.NN.ModelZoo.Resnet where

import           Data.Typeable  (Typeable)
import           RIO
import           RIO.List       (zip3)
import qualified RIO.NonEmpty   as RNE

import           MXNet.Base
import           MXNet.NN.Layer

data NoKnownExperiment = NoKnownExperiment Int
    deriving (Typeable, Show)
instance Exception NoKnownExperiment

-------------------------------------------------------------------------------
-- ResNet

resnet50Args :: ArgsHMap "resnet" (Symbol a) _
resnet50Args = (#num_stages := 4
             .& #filter_list := [64, 256, 512, 1024, 2048]
             .& #units := [3,4,6,3]
             .& #bottle_neck := True
             .& #workspace := 256 .& Nil)

resnet50 :: DType a => Int -> Symbol a -> Layer (Symbol a)
resnet50 num_classes x = do
    flt <- sequential "features" $ do
        u0 <- getFeature x resnet50Args
        u1 <- getTopFeature u0 resnet50Args
        flatten u1
    named "output"  $ fullyConnected (#data := flt .& #num_hidden := num_classes .& Nil)

resnet101Args :: ArgsHMap "resnet" (Symbol a) _
resnet101Args = (#num_stages := 4
             .& #filter_list := [64, 256, 512, 1024, 2048]
             .& #units := [3,4,23,3]
             .& #bottle_neck := True
             .& #workspace := 256
             .& Nil)

resnet101 :: DType a => Int -> Symbol a -> Layer (Symbol a)
resnet101 num_classes x = do
    flt <- sequential "features" $ do
        u0 <- getFeature x resnet101Args
        u1 <- getTopFeature u0 resnet101Args
        flatten u1
    named "dense0"  $ fullyConnected (#data := flt .& #num_hidden := num_classes .& Nil)

symbol :: DType a => Int -> Int -> Int -> Layer (Symbol a)
symbol num_classes num_layers image_size = do
    let args = if image_size <= 28 then args_small_image else args_large_image

    x <- variable "x"
    y <- variable "y"

    flt <- sequential "features" $ do
        u0 <- getFeature x args
        u1 <- getTopFeature u0 args
        flatten u1

    logits <- named "output" $ fullyConnected (#data := flt .& #num_hidden := num_classes .& Nil)
    ret    <- named "softmax" $ softmaxoutput  (#data := logits .& #label := y .& Nil)
    return ret

  where
    args_common = #workspace := 256 .& Nil
    unit0 = (num_layers - 2) `div` 9
    unit1 = (num_layers - 2) `div` 6
    args_small_image
        | (num_layers - 2) `mod` 9 == 0 && num_layers >= 164 = #num_stages := 3
                                                           .& #filter_list := [64, 64, 128, 256]
                                                           .& #units := [unit0, unit0, unit0]
                                                           .& #bottle_neck := True
                                                           .& args_common
        | (num_layers - 2) `mod` 6 == 0 && num_layers < 164 = #num_stages := 3
                                                          .& #filter_list := [64, 64, 32, 64]
                                                          .& #units := [unit1, unit1, unit1]
                                                          .& #bottle_neck := False
                                                          .& args_common

    args_large_image
        | num_layers == 18  = #num_stages := 4
                          .& #filter_list := [64, 64, 128, 256, 512]
                          .& #units := [2,2,2,2]
                          .& #bottle_neck := False
                          .& args_common
        | num_layers == 34  = #num_stages := 4
                          .& #filter_list := [64, 64, 128, 256, 512]
                          .& #units := [3,4,6,3]
                          .& #bottle_neck := False
                          .& args_common
        | num_layers == 50  = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,4,6,3]
                          .& #bottle_neck := True
                          .& args_common
        | num_layers == 101 = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,4,23,3]
                          .& #bottle_neck := True
                          .& args_common
        | num_layers == 152 = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,8,36,3]
                          .& #bottle_neck := True
                          .& args_common
        | num_layers == 200 = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,24,36,3]
                          .& #bottle_neck := True
                          .& args_common
        | num_layers == 269 = #num_stages := 4
                          .& #filter_list := [64, 256, 512, 1024, 2048]
                          .& #units := [3,30,48,8]
                          .& #bottle_neck := True
                          .& args_common

eps :: Double
eps = 2e-5

bn_mom :: Float
bn_mom = 0.9

type instance ParameterList "resnet" t =
  '[ '("num_stages" , 'AttrReq Int)
   , '("filter_list", 'AttrReq (NonEmpty Int))
   , '("units"      , 'AttrReq (NonEmpty Int))
   , '("bottle_neck", 'AttrReq Bool)
   , '("workspace"  , 'AttrReq Int)]

getFeature :: (Fullfilled "resnet" (Symbol a) args, DType a)
           => (Symbol a)
           -> ArgsHMap "resnet" (Symbol a) args
           -> Layer (Symbol a)
getFeature inp args = do
    bnx <- batchnorm   (#data := inp
                     .& #eps := eps
                     .& #momentum := bn_mom
                     .& #fix_gamma := True .& Nil)

    bdy <- convolution (#data      := bnx
                     .& #kernel    := [7,7]
                     .& #num_filter:= filter0
                     .& #stride    := [2,2]
                     .& #pad       := [3,3]
                     .& #workspace := conv_workspace
                     .& #no_bias   := True .& Nil)

    bdy <- batchnorm   (#data      := bdy
                     .& #fix_gamma := False
                     .& #eps       := eps
                     .& #momentum  := bn_mom .& Nil)

    bdy <- activation  (#data      := bdy
                     .& #act_type  := #relu .& Nil)

    bdy <- pooling     (#data      := bdy
                     .& #kernel    := [3,3]
                     .& #stride    := [2,2]
                     .& #pad       := [1,1]
                     .& #pool_type := #max
                     .& Nil)

    foldM (buildLayer bottle_neck conv_workspace) bdy (zip3 [0::Int ..2] filter_list units)

  where
    filter0 :| filter_list = args ! #filter_list
    units = RNE.toList $ args ! #units
    bottle_neck = args ! #bottle_neck
    conv_workspace = args ! #workspace

getTopFeature :: (Fullfilled "resnet" (Symbol a) args, DType a)
              => Symbol a -> ArgsHMap "resnet" (Symbol a) args -> Layer (Symbol a)
getTopFeature inp args = do
    bdy <- buildLayer bottle_neck conv_workspace inp (3, filter, unit)
    bn1 <- batchnorm   (#data := bdy -- 9
                     .& #eps := eps
                     .& #momentum := bn_mom
                     .& #fix_gamma := False .& Nil)
    ac1 <- unique' $
           activation (#data := bn1 -- 10
                    .& #act_type := #relu .& Nil)
    unique' $ pooling (#data := ac1 -- 11
                    .& #kernel := [7,7]
                    .& #pool_type := #avg
                    .& #global_pool := True .& Nil)
  where
    filter = RNE.last $ args ! #filter_list
    unit = RNE.last $ args ! #units
    bottle_neck = args ! #bottle_neck
    conv_workspace = args ! #workspace

buildLayer :: DType a
           => Bool -> Int -> (Symbol a) -> (Int, Int, Int) -> Layer (Symbol a)
buildLayer bottle_neck workspace bdy (stage_id, filter_size, unit) =
    -- unique (sformat ("stage" % int) (stage_id + 1)) $ do
    subscope_next_name $ sequential' $ do
        bdy <- residual (0,0)
                        (#data := bdy
                      .& #num_filter := filter_size
                      .& #stride := stride0
                      .& #dim_match := False
                      .& resargs)
        let conv_id = if bottle_neck then 4 else 3
            bn_id = 3
        foldM (\bdy unit_id ->
                residual (conv_id + (unit_id - 1) * 3, bn_id + (unit_id - 1) * 3)
                         (#data := bdy
                       .& #num_filter := filter_size
                       .& #stride := [1,1]
                       .& #dim_match := True
                       .& resargs)) -- unit_id
              bdy
              ([1..unit-1] :: [Int])
  where
    stride0 = if stage_id == 0 then [1,1] else [2,2]
    -- name unit_id = sformat ("features." % int % "." % int) (stage_id+5) unit_id
    resargs = #bottle_neck := bottle_neck .& #workspace := workspace .& #memonger := False .& Nil

type instance ParameterList "_residual_layer(resnet)" t =
  '[ '("data"       , 'AttrReq t)
   , '("num_filter" , 'AttrReq Int)
   , '("stride"     , 'AttrReq [Int])
   , '("dim_match"  , 'AttrReq Bool)
   , '("bottle_neck", 'AttrOpt Bool)
   , '("bn_mom"     , 'AttrOpt Float)
   , '("workspace"  , 'AttrOpt Int)
   , '("memonger"   , 'AttrOpt Bool) ]
residual :: (Fullfilled "_residual_layer(resnet)" (Symbol a) args, DType a)
         => (Int, Int)
         -> ArgsHMap "_residual_layer(resnet)" (Symbol a) args
         -> Layer (Symbol a)
residual (conv_id, bn_id) args = subscope_next_name $ do
    let dat        = args ! #data
        num_filter = args ! #num_filter
        stride     = args ! #stride
        dim_match  = args ! #dim_match
        bottle_neck= fromMaybe True $ args !? #bottle_neck
        bn_mom     = fromMaybe 0.9  $ args !? #bn_mom
        workspace  = fromMaybe 256  $ args !? #workspace
        memonger   = fromMaybe False$ args !? #memonger
    if bottle_neck
    then do
        bn1  <- -- named (sformat ("batchnorm" % int) bn_id) $
                named "bn1" $
                batchnorm   (#data := dat
                          .& #eps  := eps
                          .& #momentum  := bn_mom
                          .& #fix_gamma := False .& Nil)
        act1 <- unique' $
                activation  (#data := bn1
                          .& #act_type := #relu .& Nil)
        conv1<- -- named (sformat ("conv" % int) conv_id) $
                named "conv1" $
                convolution (#data := act1
                          .& #kernel := [1,1]
                          .& #num_filter := num_filter `div` 4
                          .& #stride := [1,1]
                          .& #pad := [0,0]
                          .& #workspace := workspace
                          .& #no_bias   := True .& Nil)

        bn2  <- -- named (sformat ("batchnorm" % int) (bn_id + 1)) $
                named "bn2" $
                batchnorm   (#data := conv1
                          .& #eps  := eps
                          .& #momentum  := bn_mom
                          .& #fix_gamma := False .& Nil)
        act2 <- unique' $
                activation  (#data := bn2
                          .& #act_type := #relu .& Nil)
        conv2<- -- named (sformat ("conv" % int) (conv_id + 1)) $
                named "conv2" $
                convolution (#data := act2
                          .& #kernel := [3,3]
                          .& #num_filter := (num_filter `div` 4)
                          .& #stride    := stride
                          .& #pad       := [1,1]
                          .& #workspace := workspace
                          .& #no_bias   := True .& Nil)

        bn3  <- -- named (sformat ("batchnorm" % int) (bn_id + 2)) $
                named "bn3" $
                batchnorm  (#data      := conv2
                         .& #eps       := eps
                         .& #momentum  := bn_mom
                         .& #fix_gamma := False .& Nil)
        act3 <- unique' $
                activation (#data := bn3
                         .& #act_type := #relu .& Nil)
        conv3<- -- named (sformat ("conv" % int) (conv_id + 2)) $
                named "conv3" $
                convolution(#data := act3
                         .& #kernel := [1,1]
                         .& #num_filter := num_filter
                         .& #stride    := [1,1]
                         .& #pad       := [0,0]
                         .& #workspace := workspace
                         .& #no_bias   := True .& Nil)
        shortcut <-
            if dim_match
            then return dat
            else -- named (sformat ("conv" % int) (conv_id + 3)) $
                 named "downsample" $
                 convolution (#data       := act1
                           .& #kernel     := [1,1]
                           .& #num_filter := num_filter
                           .& #stride     := stride
                           .& #workspace  := workspace
                           .& #no_bias    := True .& Nil)
        when memonger $
          void $ setAttr shortcut "mirror_stage" "true"

        named "plus" $ add_ conv3 shortcut

      else do
        bn1  <- -- named (sformat ("batchnorm" % int) bn_id) $
                named "bn1" $
                batchnorm    (#data      := dat
                           .& #eps       := eps
                           .& #momentum  := bn_mom
                           .& #fix_gamma := False .& Nil)
        act1 <- unique' $
                activation   (#data      := bn1
                           .& #act_type  := #relu .& Nil)
        conv1<- --named (sformat ("conv" % int) conv_id) $
                named "conv1" $
                convolution  (#data      := act1
                           .& #kernel    := [3,3]
                           .& #num_filter:= num_filter
                           .& #stride    := stride
                           .& #pad       := [1,1]
                           .& #workspace := workspace
                           .& #no_bias   := True .& Nil)

        bn2  <- --named (sformat ("batchnorm" % int) (bn_id + 1)) $
                named "bn2" $
                batchnorm    (#data      := conv1
                           .& #eps       := eps
                           .& #momentum  := bn_mom
                           .& #fix_gamma := False .& Nil)
        act2 <- unique' $
                activation  (#data      := bn2
                                           .& #act_type  := #relu .& Nil)
        conv2<- --named (sformat ("conv" % int) (conv_id + 1)) $
                named "conv2" $
                convolution  (#data      := act2
                           .& #kernel    := [3,3]
                           .& #num_filter:= num_filter
                           .& #stride    := [1,1]
                           .& #pad       := [1,1]
                           .& #workspace := workspace
                           .& #no_bias   := True .& Nil)
        shortcut <-
            if dim_match
            then return dat
            else -- named (sformat ("conv" % int) (conv_id + 2)) $
                 named "downsample" $
                 convolution (#data      := act1
                           .& #kernel    := [1,1]
                           .& #num_filter:= num_filter
                           .& #stride    := stride
                           .& #workspace := workspace
                           .& #no_bias   := True .& Nil)
        when memonger $
          void $ setAttr shortcut "mirror_stage" "true"

        named "plus" $ add_ conv2 shortcut

