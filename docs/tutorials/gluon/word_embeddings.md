# Word Embeddings

Research into word embeddings is one of the most interesting in the deep learning
world at the moment. The concept of word embeddings originate in the domain of NLP.

In this tutorial, we will discuss about embeddings, why they are needed and commonly
used model architectures to produce distributed words representation.
We will also implement one of the most widely used word2vec algorithm, called
"Continuous bag of words"(CBOW).

In creating this tutorial, I've borrowed heavily from PyTorch:

[Word Embeddings using Pytorch](http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

## Prerequisites
To complete this tutorial, we need:  

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/get_started/install.html).  

- [Python Requests](http://docs.python-requests.org/en/master/) and [Jupyter Notebook](http://jupyter.org/index.html).

```
$ pip install requests jupyter
```

## Introduction

Word embeddings are dense vectors of real numbers, one per word in your vocabulary.
In NLP, it is almost always the case that your features are words!
So where can we use these word embeddings? We can use these as features
for many machine learning and NLP applications like sentiment analysis, document classification.
The semantic information in the vectors can be efficiently used for these tasks.
We can also measure the semantic similarity between two words are by calculating the
distance between corresponding word vectors.
Let’s see an example.

Suppose we are building a language model. Suppose we have seen the sentences

- The mathematician ran to the store.
- The physicist ran to the store.
- The mathematician solved the open problem.

in our training data. Now suppose we get a new sentence never before seen in our training data:

- The physicist solved the open problem.

Our language model might do OK on this sentence, but wouldn’t it be much better if we could use the following two facts:

- We have seen mathematician and physicist in the same role in a sentence. Somehow they have a semantic relation.
- We have seen mathematician in the same role in this new unseen sentence as we are now seeing physicist.

and then infer that physicist is actually a good fit in the new unseen sentence? This is what we mean by a notion of similarity: we mean semantic similarity, not simply having similar orthographic representations. It is a technique to combat the sparsity of linguistic data, by connecting the dots between what we have seen and what we haven’t. This example of course relies on a fundamental linguistic assumption: that words appearing in similar contexts are related to each other semantically. This is called the distributional hypothesis.

## Getting Dense Word Embeddings

Now let's see how we can actually encode semantic similarity in words. Maybe we think up some semantic attributes. For example, we see that both mathematicians and physicists can run, so maybe we give these words a high score for the “is able to run” semantic attribute. Think of some other attributes, and imagine what you might score some common words on those attributes.

If each attribute is a dimension, then we might give each word a vector, like this:

qmathematician=⎡⎣⎢⎢2.3⏞can run,9.4⏞likes coffee,−5.5⏞majored in Physics,…⎤⎦⎥⎥
qmathematician=[2.3⏞can run,9.4⏞likes coffee,−5.5⏞majored in Physics,…]
qphysicist=[2.5⏞can run,9.1⏞likes coffee,6.4⏞majored in Physics,…]
qphysicist=[2.5⏞can run,9.1⏞likes coffee,6.4⏞majored in Physics,…]
Then we can get a measure of similarity between these words by doing:

Similarity(physicist,mathematician)=qphysicist⋅qmathematician
Similarity(physicist,mathematician)=qphysicist⋅qmathematician
Although it is more common to normalize by the lengths:

Similarity(physicist,mathematician)=qphysicist⋅qmathematician‖q\physicist‖‖qmathematician‖=cos(ϕ)
Similarity(physicist,mathematician)=qphysicist⋅qmathematician‖q\physicist‖‖qmathematician‖=cos⁡(ϕ)
Where ϕϕ is the angle between the two vectors. That way, extremely similar words (words whose embeddings point in the same direction) will have similarity 1. Extremely dissimilar words should have similarity -1.

You can think of the sparse one-hot vectors from the beginning of this section as a special case of these new vectors we have defined, where each word basically has similarity 0, and we gave each word some unique semantic attribute. These new vectors are dense, which is to say their entries are (typically) non-zero.

But these new vectors are a big pain: you could think of thousands of different semantic attributes that might be relevant to determining similarity, and how on earth would you set the values of the different attributes? Central to the idea of deep learning is that the neural network learns representations of the features, rather than requiring the programmer to design them herself. So why not just let the word embeddings be parameters in our model, and then be updated during training? This is exactly what we will do. We will have some latent semantic attributes that the network can, in principle, learn. Note that the word embeddings will probably not be interpretable. That is, although with our hand-crafted vectors above we can see that mathematicians and physicists are similar in that they both like coffee, if we allow a neural network to learn the embeddings and see that both mathematicians and physicisits have a large value in the second dimension, it is not clear what that means. They are similar in some latent semantic dimension, but this probably has no interpretation to us.

In summary, word embeddings are a representation of the *semantics* of a word, efficiently encoding semantic information that might be relevant to the task at hand. You can embed other things too: part of speech tags, parse trees, anything! The idea of feature embeddings is central to the field.


## Word Embeddings using Gluon package

Before we get to examples, a few quick notes about how to use embeddings in mxnet and in deep learning programming in general. Similar to how we defined a unique index for each word when making one-hot vectors, we also need to define an index for each word when using embeddings. These will be keys into a lookup table. That is, embeddings are stored as a |V|×D|V|×D matrix, where DD is the dimensionality of the embeddings, such that the word assigned index ii has its embedding stored in the ii‘th row of the matrix.

The module that allows you to use embeddings is `nn.Embedding`, which takes two arguments: the vocabulary size, and the dimensionality of the embeddings.

```python
# Mapping of word to indices
word_to_ix = {"hello": 0, "world": 1}

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Embedding(2, 5))  # 2 words in vocab, 5 dimensional embeddings

ctx = [mx.cpu(0), mx.cpu(1)]
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

data = mx.nd.array([word_to_ix["hello"]])
with autograd.record():
    z = net(data)
    z.backward()

print(z)
```

# N-Gram Language Modeling

N-gram language model is a type of probabilistic language model for predicting next
word given a sequence of words.

  P(wi|wi−1,wi−2,…,wi−n+1)

Where wi is the i'th word of the sequence.

In this example, we will compute the loss function on training data and update the parameters with backpropagation and we will show decreasing loss over iterations.


```python

# [Todo] Add code explanation

# import dependencies
from __future__ import print_function
import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from mxnet import gluon, autograd
from mxnet.gluon import nn
import logging
logging.getLogger().setLevel(logging.INFO)

context_size = 2
embeding_dim = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
vocab_size = len(vocab)
print(vocab_size)

word_to_ix = {word: i for i, word in enumerate(vocab)}


class Net(nn.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = nn.Embedding(vocab_size, embeding_dim)
            self.fc1 = nn.Dense(embeding_dim * context_size)
            self.fc2 = nn.Dense(vocab_size)

    def forward(self, x):
        x = self.embed(x)
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((1, -1))
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        #out = F.log_softmax(out)
        return out



net = Net()
net.collect_params().initialize(mx.init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})

loss = gluon.loss.SoftmaxCrossEntropyLoss()

losses = []
for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:        
        context_idxs = [word_to_ix[w] for w in context]
        context_var = mx.nd.array(context_idxs)
        label = mx.nd.array([word_to_ix[target]])
        with autograd.record():        
            log_probs = net(context_var)
            L = loss(log_probs, label)
            L.backward()

        trainer.step(2)
        total_loss += mx.nd.array(L).asscalar()  
    losses.append(total_loss)
print(losses)
```

## Computing Word Embeddings: Continuous Bag-of-Words

The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. It is a model that tries to predict words given the context of a few words before and a few words after the target word. This is distinct from language modeling, since CBOW is not sequential and does not have to be probabilistic. Typically, CBOW is used to quickly train word embeddings, and these embeddings are used to initialize the embeddings of some more complicated model. Usually, this is referred to as pretraining embeddings. It almost always helps performance a couple of percent.

The CBOW model is as follows. Given a target word wi and an NN context window on each side, wi−1,…,wi−N and wi+1,…,wi+N, referring to all context words collectively as C, CBOW tries to minimize

  $$ - \log P(w_i | C) = - \log Softmax(A(\sum_{w \in C} q_w) + b)$$

where q_w is the embedding for word ww.

So basically, CBOW predicts a word given its context. The target word vector is now the output vector of the word at index i, ; the predicted word vector is the sum over all context input vectors.

Now let's implement this model using mxnet's gluon package.


```python

# [Todo] Add code explanation

context_size = 4  # 2 words to the left, 2 to the right
embeding_dim = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
context_arr = []
target_arr = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]

    context_arr.append([word_to_ix[word] for word in context])
    target_arr.append([word_to_ix[target]])

    data.append((context, target))
print(data[:5])
print(vocab_size)

batch_size=5
def get_batch(data, batch_size, i):
    return mx.nd.array(data[i:i+batch_size])

class Net(nn.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = nn.Embedding(vocab_size, embeding_dim)
            self.fc1 = nn.Dense(vocab_size, in_units = embeding_dim)

    def forward(self, x):
        x = self.embed(x)
        x = F.sum(data = x, axis=0)  
        x = x.reshape((1, -1))
        out = self.fc1(x)
        return out

ctx = mx.cpu(0)

net = Net()
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.003})

losses = []
loss = gluon.loss.SoftmaxCrossEntropyLoss()

for epoch in range(10):
    total_loss = 0
    total = 0
    for ibatch in range(0, len(context_arr)-1, batch_size):
        context_batch = get_batch(context_arr, batch_size, ibatch)
        target_batch = get_batch(target_arr, batch_size, ibatch)
        with autograd.record():
            for x, label in zip(context_batch,target_batch):
                log_probs = net(x)                
                L = loss(log_probs, label)
                L.backward()

                total_loss += mx.nd.array(L).asscalar()

        trainer.step(batch_size)
    losses.append(total_loss)
print(losses)
```
