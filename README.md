# Hyperbolic Attention

This project aims to investigate the use of `tanh` as the basis for formulating `softmax` instead of the `logistic function (sigmoid)`. To do this we use the equality defined over scaler $x$ below:

$$ tanh(x) = 2 * sigmoid(2 * x) - 1 $$

As such we define `hypermax` over the vector $X$ as follows:

$$ hypermax(X) = 2 * softmax(2 * X) - 1 $$

The motivation for doing so is 2 fold, 1st is that it creates more generalized form of attention that allows for negation rather than just rejection. This means that Transformer hidden-states can be pushed away from one another in addition to being pulled towards one another. The 2nd motivation stems from prior NLP work before the Transformer, where `tanh ` was seen as preferable to `sigmoid` due to its empiracally improved gradients. 


## TODO:
- Finalize model code
    - make sure implementation can be trained (done)
    - decoder (done)
    - change config structure (last item before PR)
- Create training code (done)
- Finalize dataset selection
    - C4 or TinyStories (probably has to be C4 tbh)
- Finalize model hyper-params
    - model size (60M - 500M?)
    - number of training tokens
    - number of training steps (10K?)
- Finalize poetry stuff?
- Run tests

- Do I want to change this to be GaLoRE tho? (https://github.com/jiaweizzhao/GaLore/tree/master)
    - This would mean I have to change my training code, which is fine, just maybe a bit annoying?
    - They have code for the llama model and training code which I can borrow from
    - I have imported the GaLoRE code and will go through it this weekend or next week
        - Training code
        - Model code