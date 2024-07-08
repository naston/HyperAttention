# Masked Causal LM

This method is combining the objectives of Masked Language Modeling with Causal Language Modeling. This combination of objectives is motivated by Attention Sinks and my previous work augmenting the COGMEN model for multi-modal emotion recognition. 

We experiment on a base Llama model as well as extend this analysis to the new MLA architecture proposed by DeepSeekV2. We also consider a goldfish objective extension where we mask the prediction loss of certain tokens at random as well.

**Hypothesis**: This will not create any real performance gains as the volume of data means we do not train over multiple epochs and thus creates this form of regularization anyway. We find this investigation to still be worthwhile due to its easy lift and chance benefit.

Code snippets
'''
python main.py --mask
python main.py
'''

## TODO:
- Convert mask to torch Tensor type
- Make sure mask is on correct device
- Run initial tests
- Add Input masks to MLA


Code works for training a 250M parameter Llama model on a single 2070 super GPU w/ 8GB of VRAM

different rank?
different hyper-params

https://www.deepspeed.ai/getting-started/
https://www.deepspeed.ai/tutorials/zero-offload/
https://discuss.pytorch.org/t/small-model-size-but-high-gpu-memory-usage/35075/2