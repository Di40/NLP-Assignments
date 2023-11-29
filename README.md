# Natural Language Processing - Assignments

## [1 - Static Word Embeddings with Word2Vec](https://github.com/Di40/NLP-Assignments/blob/main/1%20-%20Static%20Word%20Embeddings%20with%20Word2Vec.ipynb)

- Part 1: Using pretrained word embeddings with gensim
    - Download already pretrained embeddings
    - Nearest neighbour similarity search
    - Word embedding visualization via PCA
    - Intrisic evaluation with word analogy and word similarity benchmarks
- Part 2: Pretraining our **own** embeddings
    - Training choices
    - Saving and loading our embeddings
- Part 3: Extrinsic evaluation of word embeddings
    - Using word2vec embeddings for spam classification
 
## [2- Neural Dependency Parsing](https://github.com/Di40/NLP-Assignments/blob/main/2%20-%20Neural%20Dependency%20Parsing.ipynb)

Transition-based dependency parsing is one of the most popular methods for implementing a dependency parsers.  We used the **arc-standard** model, and augmented the parser with neural machinery for contextual word embeddings and for choosing the most appropriate parser actions.  

![image](https://github.com/Di40/NLP-Assignments/assets/57565142/d002cb0d-3498-458f-b036-f3fe6bc43412)


Implemented the following features:
* LSTM representation for stack tokens
* MLP for next transition classification, based on two top-most stack tokens and first token in the buffer
* training under static oracle

## [3 - Introduction to Transformers with Huggingface](https://github.com/Di40/NLP-Assignments/blob/main/3%20-%20Introduction%20to%20Transformers%20with%20Huggingface.ipynb)

Covered parts of chapter 1-3 of the [HuggingFace Course](https://huggingface.co/learn/nlp-course/chapter1/1).

* Huggingface Hub: the ecosystem of Huggingface
* Huggingface 4 main libraries: [transformers](https://huggingface.co/docs/transformers), [tokenizers](https://huggingface.co/docs/tokenizers), [datasets](https://huggingface.co/docs/datasets), [evaluate](https://huggingface.co/docs/evaluate).

We used the MRPC (Microsoft Research Paraphrasing Corpus) dataset that is part of the [GLUE](https://huggingface.co/datasets/glue) (General Language Understanding Evaluation) benchmark.

Task: *given two sentences, assign positive class (1) if the two sentences are paraphrases of one another (assign 0 otherwise)*. To do this, we fine-tuned **BERT** on the MRPC dataset.


## [4 - Text Summarization](https://github.com/Di40/NLP-Assignments/blob/main/4%20-%20Text%20Summarization.ipynb)

In this notebook, we explored how to train and test a transformer-based model for automatic summarization using the powerful Hugging Face libraries.

Text summarization is a challenging task in the field of Natural Language Processing, aiming to condense lengthy pieces of text into shorter summaries while preserving the most important information. It finds numerous applications in areas such as news summarization, document summarization, and information retrieval.

We used the [T5 model](https://huggingface.co/docs/transformers/model_doc/t5) which is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format. In particular summarization was also included in the pre-training.

For this notebook we used the [samsum](https://huggingface.co/datasets/samsum) dataset. It contains 16k messenger-like conversations with annotated summaries.

As evaluation metric we used ROUGE (Recall-Oriented Understudy for Gisting Evaluation). It measures the overlap between the generated summary and one or more reference summaries. The key idea behind ROUGE is to capture the recall of important information in the generated summary by comparing it with the reference summaries. [Here](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) is a link with a brief explanation.

```
Jack: Cocktails later?
May: YES!!!
May: You read my mind...
Jack: Possibly a little tightly strung today?
May: Sigh... without question.
Jack: Thought so.
May: A little drink will help!
Jack: Maybe two!

Gold summary:
Jack and May will drink cocktails later.

Generated summary:
Jack and May will have a drink together.
```
