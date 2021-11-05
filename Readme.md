# Extending a Newton-CG Second-order Optimizer to Natural Language Processing

While Convolutional Neural Networks (CNNs) are a prominent class of machine learning models that are mainly applied to analyze visual imagery, Recurrent Neural Networks (RNNs) and the cutting-edge Attention Networks, Transformer Networks are another significant class of machine learning models that are mainly applied to deal with Natural Language Processing problems (NLP). Training these networks requires vast computing resources: due to a large amount of training data and due to the many training iterations. To speed up learning, many specialized algorithms have been developed. First-order methods (using just the gradient) are the most popular, but second-order algorithms (using Hessian information) are gaining importance.

We have a second-order optimizer called Newton-CG that has already shown speed-up and accuracy benefits compared with first-order optimizers for image classification problems in Mihai Zorca's bachelor thesis [(link to the thesis)](#https://mediatum.ub.tum.de/doc/1554837/1554837.pdf). In this thesis, we continue the comparison between Newton-CG and first-order optimizers, but we focus on NLP problems or Sentiment Analysis problems more specifically. We implemented totally two models: One is RNN based and the other is Self-Attention based. We trained these two models using Newton-CG optimizer and other first-order optimizers and recorded their loss and accuracy. We also tried to improve Newton-CG's performance by using Adam to pretrain.

In contrast, the performance of Newton-CG on sentiment analysis is not as good as on image classification. The performance of Newton-CG on the RNN model is very unstable, and the accuracy is only higher than that of SGD. On the Attention model, the performance of Newton-CG is more stable and the accuracy is higher.

## Environment

- numpy: 1.19.5
- tensorflow: 1.15
- keras: 2.3.0
- pandas: 1.1.5
- matplotlib: 3.2.2



## Set up environment and download the pretrained word embeddings
<u>It's recommended to use a new conda environment:</u>
```bash=
conda create -n tf1 python=3.7
conda activate tf1
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ newton-cg==0.0.3
```

Download the pretrained word embeddings: https://mega.nz/file/agtDWA5S#8TEEqSqXBKfCNcu6KRD65GWM3gJnvZZaLXs1A2UEQ1E

Make sure the file `cc.en.300.vec` is stored under folder `data`.


## Notes
It's recommended to run Self-Attention model because it's fast and doesn't require much memory.

[scripts for running Self-Attention model](#att)

To run RNN model in the SCCS environment, it's recommended to choose a time when the host is relatively idle, because it requires large memory and sometimes the process can be killed. Also, the program could need a long time to run due to unrolling RNN.

[scripts for running RNN-Attention model](#rnn)

<h2 id = "att">Self-Attention model</h2>

```bash=
# hyperparameters comparison
python Self_Attention_model/hyperparameters_comparison.py

# comparison with other optimizers
python Self_Attention_model/hyperparameters_comparison.py

# comparison between pure Newton-CG and Newton-CG with Adam pretrained
python Self_Attention_model/adam_pretrained.py 
```

The result figures are stored in `./results/attention_model/`






<h2 id = "rnn">RNN model</h2>

```bash=
# hyperparameters comparison
python RNN_model/hyperparameters_comparison.py 

# comparison with other optimizers
python RNN_model/comparison_with_other_optimizers.py 

# comparison between pure Newton-CG and Newton-CG with Adam pretrained
python RNN_model/adam_pretrained.py 
```

The result figures are stored in `./results/rnn_model/`
