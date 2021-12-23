---
layout: post
title: BERT를 처음 사용하기 위한 시각적 가이드
subtitle: A Visual Guide to Using BERT for the First Time
categories: translation
tags: [BERT, DistilBERT, Logistic Regression, Sentence Classification]
---

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-sentence-classification.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
지난 몇년간 언어를 처리하는 머신러닝 모델의 발전이 빠르게 가속화되었습니다. 이런 발전은 연구실을 떠나 주요 디지털 제품 중 일부에 힘을 공급하기 시작했습니다. 이에 대한 좋은 예는 [BERT 모델이 이제 구글 검색의 원동력이 되었다는 최근 발표](https://www.blog.google/products/search/search-language-understanding-bert/)입니다. 구글은 이 단계 (또는 검색에 적용한 자연어 이해의 발전)가 "지난 5년간의 가장 큰 도약이자, 검색  역사상 가장 큰 도약 중 하나"를 의미한다고 믿습니다. 
<span class="tooltiptext">
Progress has been rapidly accelerating in machine learning models that process language over the last couple of years. This progress has left the research lab and started powering some of the leading digital products. A great example of this is the [recent announcement of how the BERT model is now a major force behind Google Search](https://www.blog.google/products/search/search-language-understanding-bert/). Google believes this step (or progress in natural language understanding as applied in search) represents "the biggest leap forward in the past five years, and one of the biggest leaps forward in the history of Search".
</span>
</div>

<div class="tooltip" markdown="1">
이번 포스트에서는 BERT의 변형을 이용하여 문장을 분류하는 간단한 튜토리얼을 다루겠습니다. 이 것은 첫번째 소개로 충분히 기본적이지만 관련된 주요 개념 중 일부를 보여주기에 고급이기도 합니다. 
<span class="tooltiptext">
This post is a simple tutorial for how to use a variant of BERT to classify sentences. This is an example that is basic enough as a first intro, yet advanced enough to showcase some of the key concepts involved.
</span>
</div>

<div class="tooltip" markdown="1">
이 포스트와 함께 notebook도 준비했습니다. [notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)을 보거나 또는 [colab에서 실행](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) 해보실 수 있습니다.
<span class="tooltiptext">
Alongside this post, I've prepared a notebook. You can see it here [the notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) or [run it on colab](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb).
</span>
</div>
<!--more-->


<div class="tooltip" markdown="1">
## 데이터셋: SST2
<span class="tooltiptext">
Dataset: SST2
</span>
</div>

<div class="tooltip" markdown="1">
이 예제에서 사용할 데이터셋은 [SST2](https://nlp.stanford.edu/sentiment/index.html)이고, 이 것은 영화 리뷰 문장들과, 각 문장 별 긍정(값이 1) 또는 부정(값이 0)으로 레이블링 되어 있습니다:
<span class="tooltiptext">
The dataset we will use in this example is [SST2](https://nlp.stanford.edu/sentiment/index.html), which contains sentences from movie reviews, each labeled as either positive (has the value 1) or negative (has the value 0):
</span>
</div>

<table class="features-table">
  <tr>
    <th class="mdc-text-light-green-600">
    sentence (문장)
    </th>
    <th class="mdc-text-purple-600">
    label (레이블)
    </th>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      apparently reassembled from the cutting room floor of any given daytime soap
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      they presume their audience won't sit still for a sociology lesson
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      this is a visually stunning rumination on love , memory , history and the war between art and commerce
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      jonathan parker 's bartleby should have been the be all end all of the modern office anomie films
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
</table>


<div class="tooltip" markdown="1">
## 모델: 문장 감정 분류
<span class="tooltiptext">
Models: Sentence Sentiment Classification
</span>
</div>

<div class="tooltip" markdown="1">
우리의 목표는 (우리의 데이터셋과 같은) 문장을 받아서 (긍정적 감정을 가진 문장을 의미하는) 1 또는 (부정적 감정을 가진 문장을 의미하는) 0을 생성하는 것입니다. 다음과 같이 생겼다고 생각할 수 있습니다:
<span class="tooltiptext">
Our goal is to create a model that takes a sentence (just like the ones in our dataset) and produces either 1 (indicating the sentence carries a positive sentiment) or a 0 (indicating the sentence carries a negative sentiment). We can think of it as looking like this:
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/sentiment-classifier-1.png"/>
  <br />
</div>


<div class="tooltip" markdown="1">
모델은 사실 내부적으로 두개의 모델로 구성되어 있습니다. 
<span class="tooltiptext">
Under the hood, the model is actually made up of two model.
</span>
</div>

 <div class="tooltip" markdown="1">
 * [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)는 문장을 처리하고, 문장에서 추출한 몇가지 정보를 다음 모델에 전달합니다. DistilBERT는 [HuggingFace](https://huggingface.co/) 팀에서 개발하고 오픈소스로 제공하는 BERT의 작은 버전입니다. 성능은 BERT와 대략 비슷하지만 가볍고 빠른 버전입니다. 
 * 다음 모델인 scikit learn의 기본 Logistic Regression 모델은 DistilBERT 처리 결과를 받아 문장을 긍정 또는 부정 (각각 1 또는 0)으로 분류합니다. 
 <span class="tooltiptext">
 <span>*</span> [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)  processes the sentence and passes along some information it extracted from it on to the next model. DistilBERT is a smaller version of BERT developed and open sourced by the team at [HuggingFace](https://huggingface.co/). It's a lighter and faster version of BERT that roughly matches its performance.
 <span>*</span> The next model, a basic Logistic Regression model from scikit learn will take in the result of DistilBERT's processing, and classify the sentence as either positive or negative (1 or 0, respectively).
 </span>
 </div>

<div class="tooltip" markdown="1">
두 모델 간 전달하는 데이터는 768 크기의 벡터입니다. 벡터를 분류하려는 문장에 대한 임베딩으로 생각할 수 있습니다.
<span class="tooltiptext">
The data we pass between the two models is a vector of size 768. We can think of this of vector as an embedding for the sentence that we can use for classification.
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/distilbert-bert-sentiment-classifier.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
제 이전 글인 [Illustrated BERT](https://jalammar.github.io/illustrated-bert/)를 읽으셨다면, 벡터는 첫번째 위치(입력으로 [CLS] 토큰을 수신)의 결과임을 아실 수 있습니다. 
<span class="tooltiptext">
If you've read my previous post, [Illustrated BERT](https://jalammar.github.io/illustrated-bert/), this vector is the result of the first position (which receives the [CLS] token as input).
</span>
</div>

<div class="tooltip" markdown="1">
## 모델 학습
<span class="tooltiptext">
Model Training
</span>
</div>
<div class="tooltip" markdown="1">
두개의 모델을 사용하지만, logistic regression 모델만을 학습시킬 것 입니다. DistilBERT의 경우 이미 pre-train되었고 영어를 이해할 수 있는 모델을 사용할 것입니다. 하지만 이 모델은 문장 분류를 위해 fine-tune되지 않았습니다. 그러나 우리는 BERT가 훈련된 일반적인 objectives에서 일부 문장 분류 능력을 얻습니다. 이 것은 특히 첫번째 위치에 대한 BERT의 출력([CLS] 토큰과 관련됨)이 그렇습니다. BERT의 두번째 훈련 object(다음 문장 분류) 때문에 이 능력을 약간 갖게 되었다고 생각합니다. 이 objective는 첫번째 위치의 출력에 문장 전체의 의미를 캡슐화하도록 모델을 훈련시키는 것 같습니다. [transformers](https://github.com/huggingface/transformers) 라이브러리는 pretrain된 버전의 모델 뿐만 아니라 DistilBERT의 구현도 제공합니다. 
<span class="tooltiptext">
While we'll be using two models, we will only train the logistic regression model. For DistilBERT, we'll use a model that's already pre-trained and has a grasp on the English language. This model, however is neither trained not fine-tuned to do sentence classification. We get some sentence classification capability, however, from the general objectives BERT is trained on. This is especially the case with BERT's output for the first position (associated with the [CLS] token). I believe that's due to BERT's second training object -- Next sentence classification. That objective seemingly trains the model to encapsulate a sentence-wide sense to the output at the first position. The [transformers](https://github.com/huggingface/transformers) library provides us with an implementation of DistilBERT as well as pretrained versions of the model.
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/model-training.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
## 튜토리얼 개요
<span class="tooltiptext">
Tutorial Overview
</span>
</div>

<div class="tooltip" markdown="1">
이 튜토리얼의 전략은 다음과 같습니다. 우리는 먼저 2,000 문장에 대한 문장 임베딩을 생성하기 위해 훈련된 distilBERT를 사용할 것입니다. 
<span class="tooltiptext">
So here's the game plan with this tutorial. We will first use the trained distilBERT to generate sentence embeddings for 2,000 sentences.
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-tutorial-sentence-embedding.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이 단계 이후로는 distilBERT를 건드리지 않습니다. 여기부터는 모두 Scikit Learn입니다. 이 데이터셋에 대해 일반적인 train/test 분할을 수행합니다. 
<span class="tooltiptext">
We will not touch distilBERT after this step. It's all Scikit Learn from here. We do the usual train/test split on this dataset:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png"/>
  <br />
  <div class="tooltip" markdown="1">
  distilBert(모델 #1)의 출력에 대한 Train/test 분할은, logistric regression(모델 #2)을 훈련하고 평가할 데이터셋을 생성합니다. 현실에서는, sklearn의 train/test split이 분할 전에 example들을 섞기 때문에, 데이터셋에서 위쪽 75%의 example들을 (훈련에) 사용하는 것은 아닙니다. (이 포스팅에서 example은 dataset을 구성하고 있는 문장-레이블 조합을 의미합니다. 즉, dataset의 한 row 입니다.)
  <span class="tooltiptext">
    Train/test split for the output of distilBert (model #1) creates the dataset we'll train and evaluate logistic regression on (model #2). Note that in reality, sklearn's train/test split shuffles the examples before making the split, it doesn't just take the first 75% of examples as they appear in the dataset.
  </span>
</div>
</div>

<div class="tooltip" markdown="1">
그 다음 우리는 train set을 가지고 logistic regression 모델을 훈련합니다:
<span class="tooltiptext">
Then we train the logistic regression model on the training set:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-training-logistic-regression.png"/>
  <br />

</div>



<div class="tooltip" markdown="1">
## (단일) 예측을 계산하는 방법
<span class="tooltiptext">
How a single prediction is calculated
</span>
</div>

<div class="tooltip" markdown="1">
code로 들어가서 어떻게 모델이 훈련되는지 설명하기 전에, 훈련된 모델이 예측을 어떻게 계산하는지 알아봅시다.
<span class="tooltiptext">
Before we dig into the code and explain how to train the model, let's look at how a trained model calculates its prediction.
</span>
</div>

<div class="tooltip" markdown="1">
“a visually stunning rumination on love”(사랑에 대한 시각적으로 굉장히 놀라운 반추)라는 문장을 분류해봅시다. 첫번째 단계는 BERT 토크나이저를 사용하여 단어를 토큰으로 나누는 것입니다. 그 다음, 문장 분류를 위해 필요한 스페셜 토큰(첫번째 위치에 [CLS], 문장의 끝에 [SEP]이 있음)을 추가합니다. 
<span class="tooltiptext">
Let's try to classify the sentence “a visually stunning rumination on love”. The first step is to use the BERT tokenizer to first split the word into tokens. Then, we add the special tokens needed for sentence classifications (these are [CLS] at the first position, and [SEP] at the end of the sentence).
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-tokenization-1.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
토크나이저가 하는 세번째 단계는 각 토큰을 훈련된 모델의 일부분인 임베딩 테이블을 이용하여 id로 치환하는 것입니다. 단어 임베딩에 대한 배경정보는 [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)를 읽어보세요.
<span class="tooltiptext">
The third step the tokenizer does is to replace each token with its id from the embedding table which is a component we get with the trained model. Read [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) for a background on word embeddings.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-tokenization-2-token-ids.png"/>
  <br />
</div>
<div class="tooltip" markdown="1">
토크나이저는 이 모든 과정을 한줄의 코드로 수행합니다:
<span class="tooltiptext">
Note that the tokenizer does all these steps in a single line of code:
</span>
</div>
```python
tokenizer.encode("a visually stunning rumination on love", add_special_tokens=True)
```

<div class="tooltip" markdown="1">
이제 우리의 입력 문장은 DistilBERT로 전달되기에 적합한 shape이 되었습니다.
<span class="tooltiptext">
Our input sentence is now the proper shape to be passed to DistilBERT.
</span>
</div>

<div class="tooltip" markdown="1">
만약 [Illustrated BERT](https://jalammar.github.io/illustrated-bert/)를 읽으셨다면, 이 단계 또한 아래와 같이 시각화할 수 있습니다. 
<span class="tooltiptext">
If you've read [Illustrated BERT](https://jalammar.github.io/illustrated-bert/), this step can also be visualized in this manner:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-input-tokenization.png"/>
  <br />
</div>


<div class="tooltip" markdown="1">
## DistilBERT 이전 단계
<span class="tooltiptext">
Flowing Through DistilBERT
</span>
</div>

<div class="tooltip" markdown="1">
입력 벡터를 DistilBERT로 전달하는 것은 [BERT](https://jalammar.github.io/illustrated-bert/)와 동일합니다. 출력은 각 입력 토큰의 벡터가 됩니다. 각 벡터는 768개의 숫자(float)으로 구성됩니다. 
<span class="tooltiptext">
Passing the input vector through DistilBERT works [just like BERT](https://jalammar.github.io/illustrated-bert/). The output would be a vector for each input token. each vector is made up of 768 numbers (floats).
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-model-input-output-1.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
문장 분류 task이므로, 첫번째 벡터([CLS] 토큰과 관련된 것)를 제외하고 모두 무시합니다. logistic regression 모델에 대한 입력으로 전달하는 하나의 벡터입니다.
<span class="tooltiptext">
Because this is a sentence classification task, we ignore all except the first vector (the one associated with the [CLS] token). The one vector we pass as the input to the logistic regression model.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-model-calssification-output-vector-cls.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
여기에서, 훈련 단계에서 배운 것을 기반으로 이 벡터를 분류하는 것이 logistic regression 모델이 해야하는 일 입니다. 예측 계산은 다음과 같습니다:
<span class="tooltiptext">
From here, it's the logistic regression model's job to classify this vector based on what it learned from its training phase. We can think of a prediction calculation as looking like this:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-sentence-classification-example.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
훈련은 전체적인 프로세스의 코드와 함께 다음 섹션에서 알아보겠습니다.
<span class="tooltiptext">
The training is what we'll discuss in the next section, along with the code of the entire process.
</span>
</div>

<br />

<div class="tooltip" markdown="1">
## 코드
<span class="tooltiptext">
The Code
</span>
</div>

<div class="tooltip" markdown="1">
이번 섹션에서는 문장 분류 모델을 훈련하기 위한 코드를 살펴보겠습니다. 모든 코드를 포함하는 notebook은 [colab](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)과 [github](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)에서 확인하실 수 있습니다.
<span class="tooltiptext">
In this section we'll highlight the code to train this sentence classification model. A notebook containing all this code is available on [colab](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) and [github](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb).
</span>
</div>

<div class="tooltip" markdown="1">
import 부터 시작해봅시다
<span class="tooltiptext">
Let's start by importing the tools of the trade
</span>
</div>

```python
import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
```


<div class="tooltip" markdown="1">
데이터셋은 [이 github](https://github.com/clairett/pytorch-sentiment-classification/)에서 파일로 있으므로, pandas dataframe으로 바로 가져오기만 하면 됩니다. 
<span class="tooltiptext">
The dataset is [available](https://github.com/clairett/pytorch-sentiment-classification/) as a file on github, so we just import it directly into a pandas dataframe
</span>
</div>

```python
df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
```

<div class="tooltip" markdown="1">
dataframe의 첫 5행을 읽어서 데이터가 어떤 모양인지 확인하기위해 df.head()를 사용할 수 있습니다.
<span class="tooltiptext">
We can use df.head() to look at the first five rows of the dataframe to see how the data looks.
</span>
</div>

```python
df.head()
```
<div class="tooltip" markdown="1">
출력:
<span class="tooltiptext">
Which outputs:
</span>
<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/sst2-df-head.png"/>
  <br />
</div>
</div>


<div class="tooltip" markdown="1">
### pre-train된 DistilBERT 모델 및 토크나이저 import 하기
<span class="tooltiptext">
Importing pre-trained DistilBERT model and tokenizer
</span>
</div>

```python
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
## distilBERT 대신 BERT를 사용하고 싶으신가요? 그럼 아래의 주석처리를 해제하세요:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
# pretrain된 model/tokenizer를 로딩하기
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
```

<div class="tooltip" markdown="1">
이제 데이터셋을 토큰화할 수 있습니다. 이전 예제와는 조금 다른 것들을 해보려고 합니다. 위 예제에서는 한 문장에 대해서만 토큰화하고 처리를 했습니다. 지금은, 모든 문장을 배치로 묶어서 토큰화하고 처리하겠습니다 (notebook은 리소스 제약으로 작은 그룹의 example들만 처리합니다. 2000개 example이라고 가정해봅시다).
<span class="tooltiptext">
We can now tokenize the dataset. Note that we're going to do things a little differently here from the example above. The example above tokenized and processed only one sentence. Here, we'll tokenize and process all sentences together as a batch (the notebook processes a smaller group of examples just for resource considerations, let's say 2000 examples).
</span>
</div>

<div class="tooltip" markdown="1">
### 토큰화하기
<span class="tooltiptext">
Tokenization
</span>
</div>
```python
tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
```

<div class="tooltip" markdown="1">
이 코드는 모든 문장을 id로 바꿉니다. 
<span class="tooltiptext">
This turns every sentence into the list of ids.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/sst2-text-to-tokenized-ids-bert-example.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
데이터셋은 현재 중첩 리스트(또는 pandas의 Series/DataFrame)입니다. DistilBERT가 이 것을 입력으로 처리할 수 있기 전에, 짧은 문장에 토큰 id 0을 패딩(padding)해서 모든 벡터를 같은 크기로 만들 필요가 있습니다. 패딩 단계는 notebook을 참고하세요. 기본적인 파이썬 문자열(string) 및 배열(array) 조작(manipulation)입니다. 
<span class="tooltiptext">
The dataset is currently a list (or pandas Series/DataFrame) of lists. Before DistilBERT can process this as input, we'll need to make all the vectors the same size by padding shorter sentences with the token id 0. You can refer to the notebook for the padding step, it's basic python string and array manipulation.
</span>
</div>

<div class="tooltip" markdown="1">
패딩 이후에, BERT에 전달할 준비가 된 행렬/텐서가 있습니다. 
<span class="tooltiptext">
After the padding, we have a matrix/tensor that is ready to be passed to BERT:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-input-tensor.png"/>
  <br />
</div>



<div class="tooltip" markdown="1">
### DistilBERT로 처리하기
<span class="tooltiptext">
Processing with DistilBERT
</span>
</div>

<br />

<div class="tooltip" markdown="1">
패딩된 토큰 행렬에서 입력 텐서를 생성하여 DistilBERT로 전달합니다. 
<span class="tooltiptext">
We now create an input tensor out of the padded token matrix, and send that to DistilBERT
</span>
</div>

```python
input_ids = torch.tensor(np.array(padded))

with torch.no_grad():
    last_hidden_states = model(input_ids)
```

<div class="tooltip" markdown="1">
이 단계가 실행된 후 `last_hidden_states`는 DistilBERT의 출력을 가지고 있습니다. 이 것은 (example 수, sequence에서 최대 토큰 수, DistilBERT 모델에서 hidden unit 수) shape을 가진 tuple입니다. 우리의 경우에, 2000 (우리 스스로 2000개 example로 제한했으므로), 66 (2000 example로 부터 가장 긴 sequence의 토큰의 개수), 768 (DistilBERT 모델의 hidden unit 수) 입니다.
<span class="tooltiptext">
After running this step, `last_hidden_states` holds the outputs of DistilBERT. It is a tuple with the shape (number of examples, max number of tokens in the sequence, number of hidden units in the DistilBERT model). In our case, this will be 2000 (since we only limited ourselves to 2000 examples), 66 (which is the number of tokens in the longest sequence from the 2000 examples), 768 (the number of hidden units in the DistilBERT model).
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-output-tensor-predictions.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
### BERT 출력 텐서 펼쳐보기
<span class="tooltiptext">
Unpacking the BERT output tensor
</span>
</div>

<div class="tooltip" markdown="1">
이 3-차원 출력 텐서를 펼쳐서 보겠습니다. 이 것의 차원을 확인하는 것으로부터 시작할 수 있습니다.
<span class="tooltiptext">
Let's unpack this 3-d output tensor. We can first start by examining its dimensions:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-output-tensor.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
### 문장의 처리과정 요약
<span class="tooltiptext">
Recapping a sentence's journey
</span>
</div>
<div class="tooltip" markdown="1">
각 행은 우리 데이터셋으로 부터 문장과 관련이 있습니다. 첫번째 문장의 처리 경로를 요약하면 아래와 같습니다:
<span class="tooltiptext">
Each row is associated with a sentence from our dataset. To recap the processing path of the first sentence, we can think of it as looking like this:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-input-to-output-tensor-recap.png"/>
  <br />
</div>


<div class="tooltip" markdown="1">
### 필요한 부분만 자르기
<span class="tooltiptext">
Slicing the important part
</span>
</div>
<div class="tooltip" markdown="1">
문장 분류를 위해, 우리는 [CLS] 토큰에 대한 BERT의 출력에만 관심이 있습니다. 그래서, 큐브의 해당 조각만 취하고 나머지는 버립니다. 
<span class="tooltiptext">
For sentence classification, we're only only interested in BERT's output for the [CLS] token, so we select that slice of the cube and discard everything else.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-output-tensor-selection.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
3d 텐서에서 우리가 관심있는 2d 텐서를 얻기 위해 자르는 방법입니다:
<span class="tooltiptext">
This is how we slice that 3d tensor to get the 2d tensor we're interested in:
</span>
</div>

```python
 # Slice the output for the first position for all the sequences, take all hidden unit outputs
features = last_hidden_states[0][:,0,:].numpy()
```

<div class="tooltip" markdown="1">
이제 `features`는 데이터셋 모든 문장의 임베딩을 포함하고 있는 2차원 NumPy 배열입니다.
<span class="tooltiptext">
And now `features` is a 2d numpy array containing the sentence embeddings of all the sentences in our dataset.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-output-cls-senteence-embeddings.png"/>
  <br />
  <div class="tooltip" markdown="1">
  BERT의 출력에서 slice한 텐서
  <span class="tooltiptext">
  The tensor we sliced from BERT's output
  </span>
  </div>
</div>


<div class="tooltip" markdown="1">
## Logistic Regression을 위한 데이터셋
<span class="tooltiptext">
Dataset for Logistic Regression
</span>
</div>

<div class="tooltip" markdown="1">
이제 우리는 BERT의 출력을 가지고, logistic regression 모델을 훈련하는데 필요한 데이터셋을 조립했습니다. 768개의 열은 초기 데이터셋에서 방금 얻은 feature와 label입니다. 
<span class="tooltiptext">
Now that we have the output of BERT, we have assembled the dataset we need to train our logistic regression model. The 768 columns are the features, and the labels we just get from our initial dataset.
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/logistic-regression-dataset-features-labels.png"/>
  <br />
  <div class="tooltip" markdown="1">
  Logistic Regression을 훈련하는데 사용하는 레이블링된 데이터셋입니다. feature는, 이전 그림에서 슬라이스 했던 (위치 #0) [CLS] 토큰에 대한 BERT의 출력 벡터입니다. 각 행은 데이터셋의 문장에 해당하며, 각 열은 Bert/DistilBERT 모델의 가장 상단 transformer block에 있는 feed-forward neural network의 hidden unit의 출력에 해당합니다. 
  <span class="tooltiptext">
  The labeled dataset we use to train the Logistic Regression. The features are the output vectors of BERT for the [CLS] token (position #0) that we sliced in the previous figure. Each row corresponds to a sentence in our dataset, each column corresponds to the output of a hidden unit from the feed-forward neural network at the top transformer block of the Bert/DistilBERT model.
  </span>
  </div>
</div>

<div class="tooltip" markdown="1">
머신러닝에서의 전통적인 train/test split을 한 뒤, Logistic Regression 모델을 선언하고, 데이터셋에 대해 훈련할 수 있습니다. 
<span class="tooltiptext">
After doing the traditional train/test split of machine learning, we can declare our Logistic Regression model and train it against the dataset.
</span>
</div>


```python
labels = df[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

```
<div class="tooltip" markdown="1">
데이터셋을 training set과 testing set으로 분할합니다:
<span class="tooltiptext">
Which splits the dataset into training/testing sets:
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
그 다음, training set에 대해 Logistic Regression 모델을 훈련합니다. 
<span class="tooltiptext">
Next, we train the Logistic Regression model on the training set.
</span>
</div>

```python
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
```

<div class="tooltip" markdown="1">
이제 모델이 훈련됐고, test set에 대한 점수(score)를 계산할 수 있습니다:
<span class="tooltiptext">
Now that the model is trained, we can score it against the test set:
</span>
</div>

```python
lr_clf.score(test_features, test_labels)
```

<div class="tooltip" markdown="1">
모델이 약 81%의 정확도(accuracy)를 달성했음을 보여줍니다. 
<span class="tooltiptext">
Which shows the model achieves around 81% accuracy.
</span>
</div>

<br />

<div class="tooltip" markdown="1">
## 스코어 벤치마크
<span class="tooltiptext">
Score Benchmarks
</span>
</div>


<div class="tooltip" markdown="1">
참고로, 이 데이터셋에 대한 최고 정확도 스코어는 **96.8**였습니다. DistilBERT는 이 task에 대해 score를 개선하기 위해 훈련될 수 있습니다 -- 이 프로세스는 fine-tuning이라고 부르며(*downstream task*라고도 부름), BERT의 weight를 문장 분류에서 더 좋은 성능을 달성하도록 업데이트 합니다. fine-tune된 DistilBERT는 **90.7**이 정확도를 달성하는 것으로 나왔습니다. full size BERT 모델은 **94.9**를 달성했습니다. 
<span class="tooltiptext">
For reference, the highest accuracy score for this dataset is currently **96.8**. DistilBERT can be trained to improve its score on this task -- a process called fine-tuning which updates BERT's weights to make it achieve a better performance in the sentence classification (which we can call the *downstream task*). The fine-tuned DistilBERT turns out to achieve an accuracy score of **90.7**. The full size BERT model achieves **94.9**.
</span>
</div>

<div class="tooltip" markdown="1">
## Notebook 실습
<span class="tooltiptext">
The Notebook
</span>
</div>

<br />

<div class="tooltip" markdown="1">
[notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) 또는 [colab에서 실행](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)하여 바로 확인해보세요.
<span class="tooltiptext">
Dive right into [the notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) or [run it on colab](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb).
</span>
</div>


<div class="tooltip" markdown="1">
해냈습니다! 첫번째로 BERT를 다뤄보는 좋은 만남이었습니다. 다음 단계에서는 [fine-tuning](https://huggingface.co/transformers/examples.html#glue)하는 것입니다. 돌아가서 distilBERT와 BERT를 바꿔가면서 어떻게 동작하는지 확인할 수도 있습니다. 
<span class="tooltiptext">
And that's it! That's a good first contact with BERT. The next step would be to head over to the documentation and try your hand at [fine-tuning](https://huggingface.co/transformers/examples.html#glue). You can also go back and switch from distilBERT to BERT and see how that works.
</span>
</div>

<div class="tooltip" markdown="1">
[Clément Delangue](https://twitter.com/ClementDelangue), [Victor Sanh](https://twitter.com/SanhEstPasMoi), Huggingface 팀께 이 튜토리얼의 초기 버전에 대한 피드백을 주셔서 감사합니다. 
<span class="tooltiptext">
Thanks to [Clément Delangue](https://twitter.com/ClementDelangue), [Victor Sanh](https://twitter.com/SanhEstPasMoi), and the Huggingface team for providing feedback to earlier versions of this tutorial.
</span>
</div>

---

* 이 글은 Numpy에 대해 이해하기 쉽게 그림으로 설명한 Jay Alammar님의 [블로그](https://jalammar.github.io)의 글을 저자의 허락을 받고 번역한 글 입니다. 원문은 [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)에서 확인하실 수 있습니다.
* 원서/영문블로그를 보실 때 term에 대한 정보 호환을 위해, 이 분야에서 사용하고 있는 단어, 문구에 대해 가급적 번역하지 않고 원문 그대로 두었습니다. 그리고, 직역(번역체) 보다는 개념에 대한 설명을 쉽게 하는 문장으로 표현하는 쪽으로 더 무게를 두어 번역 했습니다.
* 번역문에 대응하는 영어 원문을 보고싶으신 분들을 위해 [찬](https://nlpinkorean.github.io)님께서 만들어두신 툴팁 도움말 기능(해당 문단에 마우스를 올리면 (모바일의 경우 터치) 원문을 확인할 수 있는 기능)을 가져와서 적용했습니다. 감사합니다.  
