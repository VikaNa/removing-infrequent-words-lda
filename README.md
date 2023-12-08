<h1 align="center">removing-infrequent-words-lda</h1> 


Latent Dirichlet Allocation (LDA) algorithm is frequently used in various text-as-data applications to automatically detect latent structures behind a given text collection. To prepare the data the so-called text preprocessing is performed. Many rule-of-thums have been developed over the last decade. However, only few of them are based on comprehensive simulation studies and allow for generalizable conclusions. The following analysis focuses on removing infrequent words, which make up the majority of unique terms occurring in a corpus, and the impact of this preprocessing step on the resulted topics in LDA applications.        

<h2 align="center">Monte Carlo Study</h2>

To analyse the impact of removing infrequent words in the context of LDA in a systematic way, we conduct a Monte Carlo (MC) simulation study. In proposed MC procedure, the data with known characterictics (actual number of topics) is generated. We then focus on the differences between true and estimated topics.  

<h2 align="center">Procedure</h2>

1. Define DGPs and generate corpora
We start by defining data generating processes (DGPs) to be considered in the Monte Carlo study. Number of documents in a corpus, the average text length, number of uniques terms in the corpus as the number of topics habe to be defined.

In our analysis, we consider two DGPs with the following characteristics:
<center>
  
|    |#documents|#words per document|#unique terms|#topics, K|
|----|----------|-------------------|-------------|----------|
|DGP1|1,000     |3,000              |30,000       | 50       |
|DGP2|10,000    |150                |20,000       | 15       |
</center>
The algorithm for generating a synthetic text corpus is presented in [here](generate_corpus.py).

2. Remove infrequent words
In our analysis, we consider the folloqing vocabulary pruning technques. 
- Document based frequency: remove words for which the frequency of showing up across the documents in the corpus is below the defined relative threshold.
- TF-IDF values: remove words with low TF-IDF values. 
- Absolute terms frequency: remove terms from the vocabulary the frequency of which in the corpus is below the defined absolute threshold.

After a corpus has been generated, different metrics could be applied to remove the infrequent words and to subsequently build a document-term-frequency (DTM) matrix which is needed as input for LDA models. To ensure the comparability of the metrics, we control for vocabulary size resulted when applied one of the techniques. For exanple, we start with document-based vocabulary pruning. If applying a cut-off value of 6% results in a vocabulary size of about 10,000 terms, we consider only 10,0000 terms with the highest TF-IDF values (TF-IDF technique) and highest absolute probability (absolute term frequency technique).    

3. Estimate models
For each of the defined DGPs, we 

4. Calculate metrics

<h2 align="center">References</h2>

Preliminary results are available on ArXiv: 

Bystrov, V., Naboka, V., Staszewska-Bystrova, A. and Winker, P. (2023). Analysing the Impact of Removing Infrequent Words on Topic Quality in LDA Models. ArXiv, [abs/2311.14505](https://arxiv.org/abs/2311.14505).
