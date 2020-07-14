module MXNet.NN.ModelZoo.Lenet where

import           MXNet.Base
import           MXNet.NN.Layer
import           RIO

-- # first conv
-- conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
-- tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
-- pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
-- # second conv
-- conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
-- tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
-- pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
-- # first fullc
-- flatten = mx.symbol.Flatten(data=pool2)
-- fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
-- tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
-- # second fullc
-- fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
-- # loss
-- lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

symbol :: Layer SymbolHandle
symbol = do
    x  <- variable "x"
    y  <- variable "y"

    logit <- sequential "features" $ do
        v1 <- convolution (#data := x  .& #kernel := [5,5] .& #num_filter := 20 .& Nil)
        a1 <- activation  (#data := v1 .& #act_type := #tanh .& Nil)
        p1 <- pooling     (#data := a1 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

        v2 <- convolution (#data := p1 .& #kernel := [5,5] .& #num_filter := 50 .& Nil)
        a2 <- activation  (#data := v2 .& #act_type := #tanh .& Nil)
        p2 <- pooling     (#data := a2 .& #kernel := [2,2] .& #pool_type := #max .& Nil)

        fl <- flatten     (#data := p2 .& Nil)

        v3 <- fullyConnected (#data := fl .& #num_hidden := 500 .& Nil)
        a3 <- activation     (#data := v3 .& #act_type := #tanh .& Nil)

        fullyConnected    (#data := a3 .& #num_hidden := 10  .& Nil)

    named "output" $ softmaxoutput (#data := logit .& #label := y .& Nil)
