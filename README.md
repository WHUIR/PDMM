# PDMM
The implementation of "Enhancing Topic Modeling for Short Texts with Auxiliary Word Embeddings", TOIS 2017

The model PDMM proposed in this paper is an effective topic model for short texts.

The code is contributed by Yu Duan, who is a master in Wuhan University. If you want to contact him, please email to **duanyu@whu.edu.cn**

## Description

This repository doesn't contain the preprocess steps. So if you want to use this code, you should prepare the data by yourself. 

Also this repository doesn't contain the metric code for classification and PMI score. 

The classification algorithm we used is `SVM` provided by [scikit](http://scikit-learn.org/stable/).

The PMI Coherence should calculated in external corpus, such as `Wikipedia` for English or `Baidu Baike` for Chinese.

The data format is described as follows:
> docID \t category | content

**example**:

> 0	business|manufacture manufacturers suppliers supplier china directory products taiwan manufacturer

***
Anonther file you should prepare is the `word similarity` file. In our paper, we use the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) calculated on [word embeddings](https://code.google.com/archive/p/word2vec/). This can be prepared in advance.

You also need to provide word2id file for launching the program, the data format is described as follows:
> word,id

Each line represents a word , the word and its id is separated by a comma

**example**:
> apple,0

## Parameter Explanation

`beta`: the hyper-parameter beta, and the alpha is calculated as 50/numTopic

`namda`: the hyper-parameter for Poisson distribution

`maxTd`: the max number of topics in a document

`searchTopk` : the search size of heuristic pruning strategy

`initialFileName` : the file of model's initlization 

`similarityFileName`: the file of words' similarity

`weight`: the promotion of similar word

`threshold`: the similar threshold for constructing similar word set

`filterSize`: the filter size for filtering similar word set

`numIter`: the number of iteration for gibbs sampling progress



## Model Result Explanation
`*_pdz.txt`: the topic-level representation for each document. Each line is a topic distribution for one document. This is used for classification task.

`*_phi.txt`: the word-level representation for each topic. Each line is a word distribution for one topic. This is used for PMI Coherence task.

`*_words.txt`: word, wordID map information. This is used for PMI Coherence task.



