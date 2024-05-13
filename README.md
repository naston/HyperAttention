# Hyperbolic Attention

This project aims to investigate the use of `tanh` as the basis for formulating `softmax` instead of the `logistic function (sigmoid)`. To do this we use the equality defined over scaler $x$ below:

$$ tanh(x) = 2 * sigmoid(2 * x) - 1 $$

As such we define `hypermax` over the vector $X$ as follows:

$$ hypermax(X) = 2 * softmax(2 * X) - 1 $$

The motivation for doing so is 2 fold, 1st is that it creates more generalized form of attention that allows for negation rather than just rejection. This means that Transformer hidden-states can be pushed away from one another in addition to being pulled towards one another. The 2nd motivation stems from prior NLP work before the Transformer, where `tanh ` was seen as preferable to `sigmoid` due to its empiracally improved gradients. 

It appears as if this may not currently be mathematically sound, i.e. the sum of all values of `hypermax` is `2-n`, a behavior which is not desired. 

There are 2 properties of softmax which we want to emulate:
- The sum of outputs is equal to one (well conditioned sum of outputs)
- Each output is bound between (0,1)

In the creation of hypermax we want:
- The sum of outputs is equal to a constant c (ideally still 1)
- Each output is bound between (-1,1)


## TODO:
- Run tests
- update repo with results