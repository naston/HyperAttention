# Galactic Transformer

This is a method is highly inspired by the work of the Universal Transformer as well as discoveries made around the KV cache by MiniCache. In Universal Transformer the authors show the value of parameter and information sharing across Transformer layers, they do this by repeating Transformer blocks at certain positions in depth. In MiniCache the authors show that the KV cache across proximal layers is quite similar, allowing for compression in depth. We use these two findings among others to motivate the Galactic Transformer, a model which uses a shared KV projection accross certain proximal layers. This means that not only are parameters being shared, but the KV cache is exactly the same. For Llamas this is somewhat similar to GQA in depth, however, the Queries are able to be made using new information generated from the previous layer. We also extend this work to MLA in the hopes that varied KV up-projections create better modeling results.

Code snippets
'''
python main.py
'''

## TODO:
- Convert mask to torch Tensor type
- Make sure mask is on correct device
- Run initial tests
- Add Input masks to MLA
