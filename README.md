# Hyperbolic Attention

This project aims to investigate the use of `tanh` as the basis for formulating `softmax` instead of the `logistic function (sigmoid)`. To do this we use the equality defined over scaler $x$ below:

$$ tanh(x) = 2 * sigmoid(2 * x) - 1 $$

As such we define `hypermax` over the vector $X$ as follows:

$$ hypermax(X) = 2 * softmax(2 * X) - 1 $$

The motivation for doing so is 2 fold, 1st is that it creates more generalized form of attention that allows for negation rather than just rejection. This means that Transformer hidden-states can be pushed away from one another in addition to being pulled towards one another. The 2nd motivation stems from prior NLP work before the Transformer, where `tanh ` was seen as preferable to `sigmoid` due to its empiracally improved gradients. 


## TODO:
- Finalize model code(done)
- Create training code (done)
- Finalize dataset selection
    - C4 or TinyStories (start with C4)
- Finalize model hyper-params
    - model size (60M to start, maybe 500M later)
    - number of training tokens (determined by steps currently)
    - number of training steps (10K to start)
- Initial Test run and debugging
- Run tests