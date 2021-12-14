---
layout: post
title: 그림으로 설명하는 GPT-2 (Transformer Language Model 시각화)
subtitle: The Illustrated GPT-2 (Visualizing Transformer Language Models)
categories: translation
tags: [gpt2, language model]
---

* 이 글은 GPT-2에 대해 이해하기 쉽게 그림으로 설명한 Jay Alammar님의 [블로그](https://jalammar.github.io)의 글을 저자의 허락을 받고 번역한 글 입니다.
 원문은 [The Illustrated GPT-2 (Visualizing Transformer Language Models)
](https://jalammar.github.io/illustrated-gpt2/)에서 확인하실 수 있습니다.
* 원서/영문블로그를 보실 때 term에 대한 정보 호환을 위해, 이 분야에서 사용하고 있는 단어, 문구에 대해 가급적 번역하지 않고 원문 그대로 두었습니다. 그리고, 직역(번역체) 보다는 개념에 대한 설명을 쉽게 하는 문장으로 표현하는 쪽으로 더 무게를 두어 번역 했습니다.
* 번역문-원문 단락 비교를 원하시는 분들을 위해 [찬](nlpinkorean.github.io)님께서 만들어두신 원문 tooltip 확인 기능(번역 글에 마우스를 올리면 (모바일의 경우 터치) 원문을 확인할 수 있는 기능)을 가져와서 적용했습니다. 감사합니다.  
<p align="center">(이하 본문)</p>

---


<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/openAI-GPT-2-3.png"/>
  <br />
</div>


<div class="tooltip" markdown="1">
올 해, 우리는 눈부시게 빛나는 머신러닝 어플리케이션을 보았습니다. [OpenAI의 GPT-2](https://openai.com/blog/better-language-models/)는 조리있고 강렬한 에세이들을 써내는 엄청난 능력을 보여주었습니다. 우리가 현재의 language model들이 만들어낼 것으로 기대하는 수준 이상이었습니다. GPT-2는 특별히 새로운 아키텍처는 아닙니다 -- GPT-2의 아키텍처는 decoder로만 구성된 transformer와 매우 유사합니다. 하지만 GPT-2는 방대한 양의 dataset으로 훈련된, transformer 기반의 매우 큰 language model입니다. 이번 글에서, 이 모델이 이러한 결과를 만들어낼 수 있게 한 아키텍처를 알아보고자 합니다. self-attention 레이어를 깊이 있게 살펴보고, language model 그 이상의 decoder-only transformer를 위한 어플리케이션들을 살펴보도록 하겠습니다. 
<span class="tooltiptext">
This year, we saw a dazzling application of machine learning. [The OpenAI GPT-2](https://openai.com/blog/better-language-models/) exhibited impressive ability of writing coherent and passionate essays that exceed what we anticipated current language models are able to produce. The GPT-2 wasn't a particularly novel architecture -- it's architecture is very similar to the decoder-only transformer. The GPT2 was, however, a very large, transformer-based language model trained on a massive dataset. In this post, we'll look at the architecture that enabled the model to produce its results. We will go into the depths of its self-attention layer. And then we'll look at applications for the decoder-only transformer beyond language modeling.
</span>
</div>


<div class="tooltip" markdown="1">
저의 이번 목표는, 이전 글인 [The Illustrated Transformer](/illustrated-transformer/)에 더 많은 시각적 설명을 더하여 transformer의 내부 동작 원리를 설명하고, 최초의 논문으로 부터 어떻게 발전되어 왔는지에 대해 설명하는 것 입니다. 이러한 시각적 설명을 통해, 내부 동작 방식이 계속 진화되고 있는 transformer 기반의 후속 모델들이 더 쉽게 설명이 되었으면 하는 바람이 있습니다.
<span class="tooltiptext">
My goal here is to also supplement my earlier post, [The Illustrated Transformer](/illustrated-transformer/), with more visuals explaining the inner-workings of transformers, and how they've evolved since the original paper. My hope is that this visual language will hopefully make it easier to explain later Transformer-based models as their inner-workings continue to evolve.
</span>
</div>



<!--more-->


<div class="tooltip" markdown="1">
<div style="font-size:75%; background-color:#eee; border: 1px solid #bbb; display: table; padding: 7px" markdown="1">

  <div style="text-align:center" markdown="1">  
  **목차**
  </div>

  * **[파트 1: GPT2와 Language Modeling](#part-1-got-and-language-modeling)**
    * Language Model이란
    * Language Modeling을 위한 Transformers 
    * BERT와 다른점 한가지
    * Transformer Block의 발전
    * 뇌 외과 집중 과정: GPT-2의 내부를 살펴보기
    * 더 깊이 살펴보기
    * 파트 1의 끝: GPT-2, 신사 숙녀 여러분
  * **[파트 2: Self-Attention의 설명](#part-2-illustrated-self-attention)**
    * (masking 없는) Self-Attention
    * 1- Query, Key, Value 벡터 만들기
    * 2- Score
    * 3- Sum
    * 그림으로 설명하는 Masked Self-Attention
    * GPT-2 Masked Self-Attention
    * Language modeling을 넘어
    * 해냈습니다!
  * **[파트 3: Language Modeling을 넘어](#part-3-beyond-language-modeling)**
    * 기계 번역(Machine Translation)
    * 요약(Summarization)
    * 전이 학습(Transfer Learning)
    * 음악 생성(Music Generation)
</div>
<span class="tooltiptext">
* Part 1: GPT2 And Language Modeling
  * What is a Language Model
  * Transformers for Language Modeling
  * One Difference From BERT
  * The Evolution of The Transformer Block
  * Crash Course in Brain Surgery: Looking Inside GPT-2
  * A Deeper Look Inside
  * End of part #1: The GPT-2, Ladies and Gentlemen
* Part 2: The Illustrated Self-Attention
  * Self-Attention (without masking)
  * 1- Create Query, Key, and Value Vectors
  * 2- Score
  * 3- Sum
  * The Illustrated Masked Self-Attention
  * GPT-2 Masked Self-Attention
  * Beyond Language modeling
  * You've Made it!
* Part 3: Beyond Language Modeling
  * Machine Translation
  * Summarization
  * Transfer Learning
  * Music Generation
</span>
</div>

## 파트 #1: GPT2와 Language Modeling <a href="#part-1-got-and-language-modeling" name="part-1-got-and-language-modeling">#</a>

<div class="tooltip" markdown="1">
그래서, language model이 정확히 무엇일까요?
<span class="tooltiptext">
So what exactly is a language model?
</span>
</div>

### Language Model 이란
<div class="tooltip" markdown="1">
이전 글인 [The Illustrated Word2vec](/illustrated-word2vec/)([한국어 번역본](https://databreak.netlify.app/2019-04-25-Illustrated_word2vec/))에서 language model이 무엇인지 살펴보았습니다 -- 기본적으로는, 문장의 일부를 보고 다음 단어를 예측하는 것을 할 수 있는 머신 러닝 모델입니다. 가장 유명한 language model로는 현재까지 입력된 것을 보고 다음 단어를 제안하는 스마트폰 키보드가 있습니다. 
<span class="tooltiptext">
In [The Illustrated Word2vec](/illustrated-word2vec/), we've looked at what a language model is -- basically a machine learning model that is able to look at part of a sentence and predict the next word. The most famous language models are smartphone keyboards that suggest the next word based on what you've currently typed.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/swiftkey-keyboard.png"/>
  <br />
</div>




<div class="tooltip" markdown="1">
이런 측면에서는 GPT-2가 키보드 앱의 다음단어 예측 기능이라고 말할 수 있지만, 스마트폰이 가진 것 이상으로 훨씬 더 크고 더욱 복잡합니다. GPT-2는 OpenAI 연구원들이 연구를 위한 노력의 일환으로 인터넷을 크롤링한 WebText라는 대량의 dataset(40GB)으로 학습 되었습니다. 스토리지 크기 측면에서 비교를 해본다면, 제가 사용하고 있는 키보드 앱인 SwiftKey가 78MB의 공간을 차지합니다. 훈련된 GPT-2 중에서 가장 작은 것의 경우에, 모든 파라미터를 저장하기 위해 500MB의 저장공간을 차지합니다. 가장 큰 GPT-2의 경우에는 사이즈가 13배 크기 때문에, 6.5GB 이상의 저장 공간을 차지할 수 있습니다. 
<span class="tooltiptext">
In this sense, we can say that the GPT-2 is basically the next word prediction feature of a keyboard app, but one that is much larger and more sophisticated than what your phone has. The GPT-2 was trained on a massive 40GB dataset called WebText that the OpenAI researchers crawled from the internet as part of the research effort. To compare in terms of storage size, the keyboard app I use, SwiftKey, takes up 78MBs of space. The smallest variant of the trained GPT-2, takes up 500MBs of storage to store all of its parameters. The largest GPT-2 variant is 13 times the size so it could take up more than 6.5 GBs of storage space.
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-sizes.png"/>
  <br />
</div>


<div class="tooltip" markdown="1">
GPT-2를 실험하는 가장 좋은 방법은 [AllenAI GPT-2 Explorer](https://gpt2.apps.allenai.org/?text=Joel%20is)를 이용하는 것 입니다. GPT-2를 사용하여, (확률 점수와 함께) 다음 단어로 가능한 10개의 예측을 표시해줍니다. 단어를 선택한 뒤 예측된 단어 리스트를 보고 계속해서 글의 구절을 작성할 수 있습니다. 
<span class="tooltiptext">
One great way to experiment with GPT-2 is using the [AllenAI GPT-2 Explorer](https://gpt2.apps.allenai.org/?text=Joel%20is). It uses GPT-2 to display ten possible predictions for the next word (alongside their probability score). You can select a word then see the next list of predictions to continue writing the passage.
</span>
</div>


### Transformers for Language Modeling

<div class="tooltip" markdown="1">
[Illustrated Transformer](/illustrated-transformer/)에서 본 것과 같이, 최초의 transformer model은 encoder와 decoder로 구성되어 있습니다 -- 그 각각은 우리가 transformer block들 이라고 부르는 것들을 쌓아놓은 것(stack) 입니다. 이 architecture는 원래 기계 번역을 다뤘었기 때문에, 적합했었습니다 -- encoder-decoder architecture가 과거에 성공적이었던 문제였습니다. (?)
<span class="tooltiptext">
As we've seen in The [Illustrated Transformer](/illustrated-transformer/), the original transformer model is made up of an encoder and decoder -- each is a stack of what we can call transformer blocks. That architecture was appropriate because the model tackled machine translation  -- a problem where encoder-decoder architectures have been successful in the past.
</span>
</div>


<div class="img-div" markdown="0">
  <image src="/images/xlnet/transformer-encoder-decoder.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이후 연구들에서 architecture에서 encoder 또는 decoder 중 하나를 없애고, 단 하나의 transformer block들의 stack을 사용합니다 -- block들을 현실적으로 가능한 한 높이 쌓아 올리고, 대량의 텍스트들을 학습에 이용하고, 엄청난 양의 컴퓨팅을 투하합니다. (이러한 language model들을 학습하는데 수십만 달러, [AlphaStar](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/)의 경우 수백만 달러)
<span class="tooltiptext">
A lot of the subsequent research work saw the architecture shed either the encoder or decoder, and use just one stack of transformer blocks -- stacking them up as high as practically possible, feeding them massive amounts of training text, and throwing vast amounts of compute at them (hundreds of thousands of dollars to train some of these language models, likely millions in the case of [AlphaStar](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/)).
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt-2-transformer-xl-bert-3.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
얼마나 이 block들을 높에 쌓을 수 있을까요? 이것이 GPT2 model 크기 간의 주요 구별 요소임이 밝혀졌습니다.
<span class="tooltiptext">
How high can we stack up these blocks? It turns out that's one of the main distinguishing factors between the different GPT2 model sizes:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-sizes-hyperparameters-3.png"/>
  <br />
</div>


### One Difference From BERT
<blockquote class='subtle'>
<strong>First Law of Robotics</strong><br />
<div class="tooltip" markdown="1">
로봇은 사람을 해치거나, 행동(action)하지 않아서 인간이 해를 입도록 내버려두어서는 안됩니다.
<span class="tooltiptext">
A robot may not injure a human being or, through inaction, allow a human being to come to harm.
</span>
</div>
</blockquote>


<div class="tooltip" markdown="1">
GPT-2는 transformer의 decoder 블럭으로 구성됩니다. 반대로 BERT는, transformer의 encoder 블럭을 사용합니다. 다음 섹션에서 이 차이를 실험해보겠습니다. 하지만, 이 둘 간의 중요한 차이 하나는 GPT2가 다른 전통적인 language model들 처럼 한번에 하나의 token을 출력한다는 것 입니다. 잘 훈련된 GPT-2가 로보틱스의 제 1법칙을 낭독(출력)하도록 해봅시다.
<span class="tooltiptext">
The GPT-2 is built using transformer decoder blocks. BERT, on the other hand, uses transformer encoder blocks. We will examine the difference in a following section. But one key difference between the two is that GPT2, like traditional language models, outputs one token at a time. Let's for example prompt a well-trained GPT-2 to recite the first law of robotics:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/gpt-2-output.gif"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이러한 모델들이 실제로 동작하는 방식은, 각 token이 생성된 후에 입력 시퀀스에 더해지는 것 입니다. 그러한 새 시퀀스는, 다음 스텝에서, 모델의 입력으로 들어갑니다. 이 것을 "auto-regression"이라고 부릅니다. 이 것은 [RNN을 과도하게 효과적으로 만든](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) idea 중 하나 입니다.
<span class="tooltiptext">
The way these models actually work is that after each token is produced, that token is added to the sequence of inputs. And that new sequence becomes the input to the model in its next step. This is an idea called "auto-regression". This is one of the ideas that [made RNNs unreasonably effective](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/gpt-2-autoregression-2.gif"/>
  <br />
</div>

<div class="tooltip" markdown="1">
GPT2, 그리고 이후에 등장한 TransformerXL과 XLNet과 같은 모델들은 본질적으로 auto-regressive입니다. BERT는 그렇지 않습니다. 이 것은 상충관계(trade off)가 있습니다. auto-regression 특성을 잃는 대신, BERT는 더 좋은 결과를 내기 위해 단어의 양쪽 방향에서 context를 활용할 수 있는 능력을 얻었습니다. XLNet은 autoregression을 되돌리면서 양쪽 방향의 context를 활용하기 대안적 방법을 찾습니다. (?)
<span class="tooltiptext">
The GPT2, and some later models like TransformerXL and XLNet are auto-regressive in nature. BERT is not. That is a trade off. In losing auto-regression, BERT gained the ability to incorporate the context on both sides of a word to gain better results. XLNet brings back autoregression while finding an alternative way to incorporate the context on both sides.
</span>
</div>

### The Evolution of the Transformer Block

<div class="tooltip" markdown="1">
[initial transformer paper](https://arxiv.org/abs/1706.03762)에서는 두가지 타입의 transformer block에 대해 소개합니다. 
<span class="tooltiptext">
The [initial transformer paper](https://arxiv.org/abs/1706.03762) introduced two types of transformer blocks:
</span>
</div>

#### The Encoder Block

<div class="tooltip" markdown="1">
먼저 encoder block 입니다:
<span class="tooltiptext">
First is the encoder block:
</span>
</div>

<div class="img-div" markdown="0">
  <image src="/images/xlnet/transformer-encoder-block-2.png"/>
  <br />
  <div class="tooltip" markdown="1">
  최초의 transformer 논문에서 encoder block은 input을 특정 max sequence length(예: 512개의 토큰)까지 가질 수 있습니다. input sequence가 이 한계 보다 짧으면 괜찮습니다. sequence의 나머지 부분에 padding을 할 수 있습니다.
  <span class="tooltiptext">
  An encoder block from the original transformer paper can take inputs up until a certain max sequence length (e.g. 512 tokens). It's okay if an input sequence is shorter than this limit, we can just pad the rest of the sequence.
  </span>
  </div>
</div>


#### The Decoder Block
<div class="tooltip" markdown="1">
두번째로, encoder block의 architecture를 살짝 변형한 decoder block이 있습니다 -- encoder로 부터의 특정 segment에 attention을 줄 수 있도록 하는 layer 입니다.
<span class="tooltiptext">
Second, there's the decoder block which has a small architectural variation from the encoder block -- a layer to allow it to pay attention to specific segments from the encoder:
</span>
</div>

<div class="img-div" markdown="0">
  <image src="/images/xlnet/transformer-decoder-block-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
여기서의 self-attention layer에서의 주요 다른 특징은, 앞으로 나올 token들을 masking하는 것 입니다 -- BERT 처럼 word를 [mask]로 치환하는 것이 아니라, aelf-attention 계산 시에, 계산되어야 하는 위치의 오른쪽에 있는 token들로부터의 정보를 막는 방법으로 차단합니다.
<span class="tooltiptext">
One key difference in the self-attention layer here, is that it masks future tokens -- not by changing the word to [mask] like BERT, but by interfering in the self-attention calculation blocking information from tokens that are to the right of the position being calculated.
</span>
</div>

<div class="tooltip" markdown="1">
예를 들어, #4번의 경로를 강조 표시하면, 우리는 현재와 이전 token들에 대해선만 주의/주목(attention)을 줄 수 있습니다.
<span class="tooltiptext">
If, for example, we're to highlight the path of position #4, we can see that it is only allowed to attend to the present and previous tokens:
</span>
</div>

<div class="img-div" markdown="0">
  <image src="/images/xlnet/transformer-decoder-block-self-attention-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
(BERT가 사용하는) self-attention과 (GPT-2가 사용하는) masked self-attention이 확연히 다르다는 것은 중요합니다. 일반 self-attention block은 자신보다 오른쪽에 있는 token을 선택(pick의 오타로 추정 (?))할 수 있도록 합니다. masked self-attention의 경우에는, 이런 상황을 방지합니다.
<span class="tooltiptext">
It's important that the distinction between self-attention (what BERT uses) and masked self-attention (what GPT-2 uses) is clear. A normal self-attention block allows a position to peak at tokens to its right. Masked self-attention prevents that from happening:
</span>
</div>

<div class="img-div-any-size" markdown="0">
  <image src="/images/gpt2/self-attention-and-masked-self-attention.png"/>
  <br />
</div>

#### The Decoder-Only Block
<div class="tooltip" markdown="1">
최초의 논문에 이어, [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf)에서는 language modeling이 가능한, 다른 배열의 transformer block을 제안했습니다. 이 모델은 transformer의 encoder를 버렸습니다. 그러한 이유로, 이 모델을 "Transformer-Decoder"라고 부르겠습니다. 이 초창기의 transformer 기반의 language model은 6개의 transformer decoder block으로 구성되었습니다:
<span class="tooltiptext">
Subsequent to the original paper, [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf) proposed another arrangement of the transformer block that is capable of doing language modeling. This model threw away the Transformer encoder. For that reason, let's call the model the "Transformer-Decoder". This early transformer-based language model was made up of a stack of six transformer decoder blocks:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/transformer-decoder-intro.png"/>
  <br />
  <div class="tooltip" markdown="1">
  이 decoder block들은 동일합니다. 첫번째를 확대했기 때문에, self-attention layer가 mask된 변형임을 알 수 있습니다. 현재는, 이 모델이 특정 segment에서 최대 4,000개 token까지 참조할 수 있음을 주목하세요 -- 최초 transformer에서 512개 였던 것에서 크게 업그레이드 되었습니다.
  <span class="tooltiptext">
  The decoder blocks are identical. I have expanded the first one so you can see its self-attention layer is the masked variant. Notice that the model now can address up to 4,000 tokens in a certain segment -- a massive upgrade from the 512 in the original transformer.
  </span>
  </div>
</div>

<div class="tooltip" markdown="1">
이 block들은, 두번째 self-attention layer를 제거한 것을 제외하면, 최초의 decoder block들과 매우 유사합니다. 비슷한 architecture가 [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/pdf/1808.04444.pdf)에서, 한번에 하나의 문자(letter/character)를 예측하는 language model을 만들기 위해 실험되었습니다.
<span class="tooltiptext">
These blocks were very similar to the original decoder blocks, except they did away with that second self-attention layer. A similar architecture was examined in [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/pdf/1808.04444.pdf) to create a language model that predicts one letter/character at a time.
</span>
</div>


<div class="tooltip" markdown="1">
OpenAI의 GPT-2 모델은 이러한 decoder만으로 구성된 block들을 사용합니다.
<span class="tooltiptext">
The OpenAI GPT-2 model uses these decoder-only blocks.
</span>
</div>

### Crash Course in Brain Surgery: Looking Inside GPT-2

<blockquote class='subtle' markdown="block">
Look inside and you will see,
The words are cutting deep inside my brain.
Thunder burning, quickly burning,
Knife of words is driving me insane, insane yeah.
~<strong>[Budgie](https://en.wikipedia.org/wiki/Budgie_(band))</strong>
</blockquote>

<div class="tooltip" markdown="1">
훈련된 GPT-2를 수술대 위에 올려놓고, 어떻게 동작하는지 살펴봅시다.
<span class="tooltiptext">
Let's lay a trained GPT-2 on our surgery table and look at how it works.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt-2-layers-2.png"/>
  <br />
  <div class="tooltip" markdown="1">
  GPT-2는 1024개의 token을 처리할 수 있습니다. 각 token은 각자의 경로를 따라서 모든 decoder block으로 흘러갑니다.
  <span class="tooltiptext">
  The GPT-2 can process 1024 tokens. Each token flows through all the decoder blocks along its own path.
  </span>
  </div>
</div>

<div class="tooltip" markdown="1">
훈련된 GPT-2 모델을 돌렬보는 가장 간단한 방법은 그 것 자체의 램블링(rambling/패턴없이 되는 대로 퍼져나가는)을 하도록 하는 것 입니다 (엄밀히 따져 말하면, *generating unconditional samples*입니다) -- 또는, prompt를 주고 특정 토픽에 대해 말해보도록 할 수 있습니다. (*interactive conditional samples* 생성이라고도 알려져 있습니다). rambling 방법에서, 우리는 간단히 시작 token을 주고 단어들을 생성하기 시작하도록 할 수 있습니다 (훈련된 모델은 ```<|endoftext|>```를 시작 token으로 쓰지만, 여기서는 ```<s>```으로 사용하겠습니다).
<span class="tooltiptext">
The simplest way to run a trained GPT-2 is to allow it to ramble on its own (which is technically called *generating unconditional samples*) -- alternatively, we can give it a prompt to have it speak about a certain topic (a.k.a generating *interactive conditional samples*). In the rambling case, we can simply hand it the start token and have it start generating words (the trained model uses ```<|endoftext|>``` as its start token. Let's call it ```<s>``` instead).
</span>
</div>

<div class="img-div-any-width" markdown="1">
  <image src="/images/gpt2/gpt2-simple-output-2.gif"/>
  <br />
</div>

<div class="tooltip" markdown="1">
input token 1개 만을 가지고 있는 상황에서, 모델은 경로가 1개만 활성화 됩니다. token은 모든 레이어를 통해 연속적으로 거쳐 처리된 다음 vector가 생성됩니다. 그 vector는 모델의 vocab(어휘)에 대해 score가 매겨질 수 있습니다(?)(모델이 알고 있는 모든 단어들, GPT-2의 경우엔 50,000개의 단어). 이런 경우에 우리는 가장 높은 확률을 갖는 'the'를 선택합니다. But we can certainly mix things up -- 키보드 앱에서 제안하는 단어를 클릭하기를 계속하면, 가끔 반복 루프에 빠질 때가 있고, 이 때 유일한 탈출 방법은 두번째나 세번째로 제안한 단어를 선택하는 것 입니다. 같은 상황이 여기서도 있습니다. GPT-2는 top-k라는 parameter를 가지고 있어서 우리는 모델이 top 단어(top-k가 1인 경우)가 아닌 다른 단어를 샘플링하게 만들 수 있습니다.
<span class="tooltiptext">
The model only has one input token, so that path would be the only active one. The token is processed successively through all the layers, then a vector is produced along that path. That vector can be scored against the model's vocabulary (all the words the model knows, 50,000 words in the case of GPT-2). In this case we selected the token with the highest probability, 'the'. But we can certainly mix things up -- you know how if you keep clicking the suggested word in your keyboard app, it sometimes can stuck in repetitive loops where the only way out is if you click the second or third suggested word. The same can happen here. GPT-2 has a parameter called top-k that we can use to have the model consider sampling words other than the top word (which is the case when top-k = 1).
</span>
</div>

<div class="tooltip" markdown="1">
다음 단계에서, 첫번째 단계의 output을 input sequence에 덧붙인 뒤 모델이 다음 예측을 수행합니다:
<span class="tooltiptext">
In the next step, we add the output from the first step to our input sequence, and have the model make its next prediction:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt-2-simple-output-3.gif"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이번 계산에서는 두 번째 경로만 활성화되는 것을 주목하세요. GPT-2의 각 레이어는 첫번째 token의 interpretation을 재학습하고, 두번째 token을 처리할 때에 사용합니다 (뒤에서 self-attention 섹션에서 더 자세히 알아보겠습니다). GPT-2는 두번째 token에 비추어 첫번째 token을 재해석하지 않습니다. (?)
<span class="tooltiptext">
Notice that the second path is the only that's active in this calculation. Each layer of GPT-2 has retained its own interpretation of the first token and will use it in processing the second token (we'll get into more detail about this in the following section about self-attention). GPT-2 does not re-interpret the first token in light of the second token.
</span>
</div>

### A Deeper Look Inside

#### Input Encoding
<div class="tooltip" markdown="1">
모델에 대해 더 상세하게 알기 위해 더 자세한 사항들을 봅시다. input으로 부터 시작합시다. 이전에 논의했던 다른 NLP 모델들처럼, GPT-2 모델도 embedding matrix에서 input 단어의 embedding을 조회합니다 -- 이 embedding matrix는 학습된 모델의 일부로서, 얻을 수 있는 구성요소 중 하나 입니다. 
<span class="tooltiptext">
Let's look at more details to get to know the model more intimately. Let's start from the input. As in other NLP models we've discussed before, the model looks up the embedding of the input word in its embedding matrix -- one of the components we get as part of a trained model.
</span>
</div>

<div class="img-div" markdown="0">
  <image src="/images/gpt2/gpt2-token-embeddings-wte-2.png"/>
  <br />
  <div class="tooltip" markdown="1">
  각 행은 단어 embedding 입니다: 단어를 표현하고 그 의미를 캡쳐(포함)하는 숫자형태로 표현된 리스트(목록). 이 리스트의 크기는 GPT2 모델 사이즈에 따라 다릅니다. 제일 작은 모델은 단어(토큰) 당 768 embedding 크기를 사용합니다. 
  <span class="tooltiptext">
  Each row is a word embedding: a list of numbers representing a word and capturing some of its meaning. The size of that list is different in different GPT2 model sizes. The smallest model uses an embedding size of 768 per word/token.
  </span>
  </div>
</div>

<div class="tooltip" markdown="1">
처음에는, embedding matrix에서 start token ```<s>```의 embedding을 조회합니다. 모델의 첫번째 block에 이 정보를 전달하기 전에, 위치 encoding 정보를 통합해야 합니다. -- 위치 encoding은 transformer block으로 sequence 상에서의 word들의 순서 정보를 알려주는 신호(정보)입니다. 훈련된 모델을 구성하는 한 부분은 input의 1024개 위치 각각에 대한 위치 encoding vector입니다.
<span class="tooltiptext">
So in the beginning, we look up the embedding of the start token ```<s>``` in the embedding matrix. Before handing that to the first block in the model, we need to incorporate positional encoding -- a signal that indicates the order of the words in the sequence to the transformer blocks. Part of the trained model is a matrix that contains a positional encoding vector for each of the 1024 positions in the input.
</span>
</div>

<div class="img-div" markdown="0">
  <image src="/images/gpt2/gpt2-positional-encoding.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이로써 input 단어들이 transformer의 첫번쨰 block에 전달되기 전에 어떤 처리가 있는지 살펴봤습니다. 또한 훈련된 GPT-2를 구성하는 두개의 weight matrix(가중치 행렬)도 알게 되었습니다.
<span class="tooltiptext">
With this, we've covered how input words are processed before being handed to the first transformer block. We also know two of the weight matrices that constitute the trained GPT-2.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-input-embedding-positional-encoding-3.png"/>
  <br />
  <div class="tooltip" markdown="1">
  word를 transformer block의 첫번째로 전달하는 것은 그 word의 embedding과 #1번 위치의 encoding vector를 더하는 것을 의미합니다.
  <span class="tooltiptext">
  Sending a word to the first transformer block means looking up its embedding and adding up the positional encoding vector for position #1.
  </span>
  </div>
</div>

#### A journey up the Stack

<div class="tooltip" markdown="1">
첫번째 block은 이제 token을 self-attention 프로세스를 통해 전달하고, neural network 레이어로 전달하여 처리할 수 있습니다. 첫번째 transformer block이 이 token을 처리하면, 그 결과 vector를 다음 block에서 처리되도록 윗쪽 stack으로 올립니다. 프로세스는 각 block마다 동일하지만 각 block은 각자의 self-attention 및 neural network sublayer에 대한 weight들을 가지고 있습니다. 
<span class="tooltiptext">
The first block can now process the token by first passing it through the self-attention process, then passing it through its neural network layer. Once the first transformer block processes the token, it sends its resulting vector up the stack to be processed by the next block. The process is identical in each block, but each block has its own weights in both self-attention and the neural network sublayers.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-transformer-block-vectors-2.png"/>
  <br />
</div>



#### Self-Attention Recap

<div class="tooltip" markdown="1">
언어는 context(문맥)에 매우 의존적입니다. 예를들어, 제 2법칙을 봅시다:
<span class="tooltiptext">
Language heavily relies on context. For example, look at the second law:
</span>
</div>

<blockquote class='subtle'>

<strong>Second Law of Robotics</strong><br />
<div class="tooltip" markdown="1">
로봇은 <strong style="color:#6D4C41">제 1법칙</strong>에 위반되는 <strong style="color:#689F38">그러한 명령들</strong>을 제외하고는 인간이 <strong style="color:#D81B60">그것</strong>에게 내린 명령들에 복종해야 한다.
<span class="tooltiptext">
A robot must obey the orders given <strong style="color:#D81B60">it</strong> by human beings except where <strong style="color:#689F38">such orders</strong> would conflict with the <strong style="color:#6D4C41">First Law</strong>.
</span>
</div>

</blockquote>

<div class="tooltip" markdown="1">
문장에서 다른 word를 참조하는 단어들 3군데를 하이라이트 표기 했습니다. 이 단어들은 참조하는 context와 통합적으로 보지 않으면 이해 또는 처리할 수 없습니다. 모델이 이 문장을 처리할 때, 다음을 알 수 있어야 합니다.
<span class="tooltiptext">
I have highlighted three places in the sentence where the words are referring to other words. There is no way to understand or process these words without incorporating the context they are referring to. When a model processes this sentence, it has to be able to know that:
</span>
</div>

<div class="tooltip" markdown="1">
* <strong style="color:#D81B60">그것</strong>은 로봇을 가르킵니다.
* <strong style="color:#689F38">그러한 명령들</strong>은 이 법칙의 앞부분(참고: 한국어에서는 언어 구조 상 뒷쪽에 위치)인, "인간이 그것에게 내린 명령들"을 가르킵니다.
* <strong style="color:#6D4C41">제 1법칙</strong>은 제 1법칙 전체를 가르킵니다.
<span class="tooltiptext">
* <strong style="color:#D81B60">it</strong> refers to the robot
* <strong style="color:#689F38">such orders</strong> refers to the earlier part of the law, namely "the orders given it by human beings"
* <strong style="color:#6D4C41">The First Law</strong> refers to the entire First Law
</span>
</div>

<div class="tooltip" markdown="1">
이 것이 self-attention이 하는 일 입니다. (neural network를 통해 전달되는) 단어를 처리하기 전에, 특정 단어의 context를 설명하는 관련 word들에 대한 모델의 이해를 만듭니다. segment에서 각 word가 얼마나 관련되어 있는지 score를 할당하고, 그 vector representation을 합산하는 방법으로 이를 수행합니다.
<span class="tooltiptext">
This is what self-attention does. It bakes in the model's understanding of relevant and associated words that explain the context of a certain word before processing that word (passing it through a neural network). It does that by assigning scores to how relevant each word in the segment is, and adding up their vector representation.
</span>
</div>

<div class="tooltip" markdown="1">
예를 들어, 상단의 block에서 self-attention layer는 단어 "it"을 처리할 때 "a robot"에 attention을 줍니다. neural network으로 전달하는 vector는, 그 3개의 단어들의 vector에 각 score들이 곱해진 것의 합 입니다. 
<span class="tooltiptext">
As an example, this self-attention layer in the top block is paying attention to "a robot" when it processes the word "it". The vector it will pass to its neural network is a sum of the vectors for each of the three words multiplied by their scores.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-example-2.png"/>
  <br />
</div>

#### Self-Attention Process
<div class="tooltip" markdown="1">
Self-attention은 segment에서 각 token의 path를 따라 처리됩니다. 중요한 요소들은 다음 세가지 vector들 입니다:
<span class="tooltiptext">
Self-attention is processed along the path of each token in the segment. The significant components are three vectors:
</span>
</div>

<div class="tooltip" markdown="1">
* <span class="decoder">Query</span>: Query는 다른 모든 word들(해당 key 사용)과 score를 계산하는데 사용되는 현재 단어(word)를 나타냅니다. 우리는 현재 처리 중인 token의 query 값에만 관심이 있습니다.
* <span class="context">Key</span>: Key vector는 segment에서 모든 word들에 대한 label과 같습니다. 관련 word를 검색할 때 매칭되는 항목입니다.
* <span class="step_no">Value</span>: Value vector는 실제 word representation 입니다. 우리가 각 단어가 얼마나 관련이 있는지 score를 매기고 나면, 현재의 word를 표현하기 위해 합산한 값입니다.
<span class="tooltiptext">
* <span class="decoder">Query</span>: The query is a representation of the current word used to score against all the other words (using their keys). We only care about the query of the token we're currently processing.
* <span class="context">Key</span>: Key vectors are like labels for all the words in the segment. They're what we match against in our search for relevant words.
* <span class="step_no">Value</span>: Value vectors are actual word representations, once we've scored how relevant each word is, these are the values we add up to represent the current word.
</span>
</div>


<div class="img-div-any-size" markdown="0">
  <image src="/images/gpt2/self-attention-example-folders-3.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
대강 비유해보자면, 서류 캐비넷에서 어떤 서류를 찾는 것과 같다고 생각하는 것 입니다. Query는 찾고자 하는 주제를 적은 메모지 입니다. Key는 캐비넷 안의 서류 폴더들에 달린 label과 같습니다. 메모장과 tag를 매칭시키면, 폴더에서 내용물을 꺼내는데, 이 내용물이 바로 value vector 입니다. 단지 다른 점은, 하나의 value만 찾지 않고, 여러 폴더들에서 여러 value들을 찾는다는 것 입니다.
<span class="tooltiptext">
A crude analogy is to think of it like searching through a filing cabinet. The query is like a sticky note with the topic you're researching. The keys are like the labels of the folders inside the cabinet. When you match the tag with a sticky note, we take out the contents of that folder, these contents are the value vector. Except you're not only looking for one value, but a blend of values from a blend of folders.
</span>
</div>

<div class="tooltip" markdown="1">
Query vector를 각 key vector에 곱해서, 각 폴더 별 score 값을 만듭니다 (기술적으로: 내적(dot product) 연산 뒤 softmax 연산 수행).
<span class="tooltiptext">
Multiplying the query vector by each key vector produces a score for each folder (technically: dot product followed by softmax).
</span>
</div>


<div class="img-div-any-size" markdown="0">
  <image src="/images/gpt2/self-attention-example-folders-scores-3.png"/>
  <br />
</div>


<div class="tooltip" markdown="1">
각 value를 위에서 구한 score과 곱한 뒤, 합산합니다 -- self-attention 결과가 나오게 됩니다. 
<span class="tooltiptext">
We multiply each value by its score and sum up -- resulting in our self-attention outcome.
</span>
</div>

<div class="img-div-any-size" markdown="0">
  <image src="/images/gpt2/gpt2-value-vector-sum.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이 가중치 혼합된 value vector는, 50%는 단어 ```robot```에, 30%는 ```a```에, 19%는 ```it```에 attention을 준 vector를 생성합니다. 이 글의 뒷부분에서, self-attention에 대해 더 자세히 알아보겠습니다. 지금은, 모델의 output을 향한 stack의 여정을 계속합시다.
<span class="tooltiptext">
This weighted blend of value vectors results in a vector that paid 50% of its "attention" to the word ```robot```, 30% to the word ```a```, and 19% to the word ```it```. Later in the post, we'll got deeper into self-attention. But first, let's continue our journey up the stack towards the output of the model.
</span>
</div>

#### Model Output

<div class="tooltip" markdown="1">
모델의 최상위 block이 (최상위 block의 self-attention 및 neural network 계산을 거친 결과인) output vector를 생성할 때 , 모델은 그 vector와 embedding matrix를 곱합니다. 
<span class="tooltiptext">
When the top block in the model produces its output vector (the result of its own self-attention followed by its own neural network), the model multiplies that vector by the embedding matrix.
</span>
</div>

<div class="img-div-any-size" markdown="0">
  <image src="/images/gpt2/gpt2-output-projection-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
embedding matrix의 각 행은 모델 어휘(vocab)의 단어의 embedding에 해당합니다. 이 곱셈의 결과는 모델의 어휘에서 각 word에 대한 score로 해석됩니다.
<span class="tooltiptext">
Recall that each row in the embedding matrix corresponds to the embedding of a word in the model's vocabulary. The result of this multiplication is interpreted as a score for each word in the model's vocabulary.
</span>
</div>

<div class="img-div-any-size" markdown="0">
  <image src="/images/gpt2/gpt2-output-scores-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
가장 높은 score를 갖는 token을 선택해봅시다 (top_k = 1). 그러나 모델이 다른 word들도 고려한다면 더 좋은 결과를 얻을 수 있습니다. 더 좋은 전략은 전체 리스트에서 score를 어떤 word를 고르기 위한 확률값으로 사용하여 단어(word)를 선택하는 것 입니다 (그래서 높은 score를 갖는 word들이 선택될 가능성이 더 높습니다). 절충안은 top_k를 40으로 잡고, 모델이 가장 높은 score를 갖는 40개의 word를 고려하도록 하는 것 입니다.
<span class="tooltiptext">
We can simply select the token with the highest score (top_k = 1). But better results are achieved if the model considers other words as well. So a better strategy is to sample a word from the entire list using the score as the probability of selecting that word (so words with a higher score have a higher chance of being selected). A middle ground is setting top_k to 40, and having the model consider the 40 words with the highest scores.
</span>
</div>


<div class="img-div-any-size" markdown="0">
  <image src="/images/gpt2/gpt2-output.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
그렇게 함으로써, 모델은 하나의 word를 출력하면서 한 iteration을 종료합니다. 모델은 전체적으로 context가 생성(1024개의 token)될 때 까지 혹은 EOS(end-of-sequence) token이 생성될 떄까지 iteration을 계속 수행합니다.
<span class="tooltiptext">
With that, the model has completed an iteration resulting in outputting a single word. The model continues iterating until the entire context is generated (1024 tokens) or until an end-of-sequence token is produced.
</span>
</div>

### End of part #1: The GPT-2, Ladies and Gentlemen

<div class="tooltip" markdown="1">
And there we have it.(?) GPT2 동작 방식에 대한 요약입니다. 만약 self-attention 레이어의 안쪽에서 무슨 일이 일어나는지 궁금하다면, 아래의 보너스 섹션을 살펴보세요. 나중에 transformer 모델을 더 쉽게 조사하고 설명할 수 있도록, self-attention을 설명하는 더 시각적인 언어(설명)를 도입하기 위해 이 글을 썼습니다. (looking at you, TransformerXL와 XLNet).
<span class="tooltiptext">
And there we have it. A run down of how the GPT2 works. If you're curious to know exactly what happens inside the self-attention layer, then the following bonus section is for you. I created it to introduce more visual language to describe self-attention in order to make describing later transformer models easier to examine and describe (looking at you, TransformerXL and XLNet).
</span>
</div>

<div class="tooltip" markdown="1">
이 글에서 매우 단순화 시켰던 점들은 다음과 같습니다:
<span class="tooltiptext">
I'd like to note a few oversimplifications in this post:
</span>
</div>


<div class="tooltip" markdown="1">
* "word"와 "token"을 같은 의미로 사용했습니다. 하지만 실제로는, GPT2는 어휘(vocab)의 token들을 만들기 위해 BPE(Byte Pair Encoding)을 사용합니다. 이 것은 일만적으로 token이 word의 일부임을 의미합니다.
* 예로 든 GPT2는 추론(inference)/평가(evaluation) 모드입니다. (설명 과정에서) 한번에 하나의 word만을 처리하는 이유입니다. 학습(training) 시에는, 모델은 더 긴 text sequence에 대해 학습하며 한번에 여러개의 token을 처리할 것입니다. 또한, 모델은 evaluation 때 사용하는 배치 사이즈 보다 더 큰 배치 사이즈 (512)를 처리할 것 입니다.
* 그림에서 공간을 효과적으로 사용하기 위해 회전/치환을 자유롭게 사용했습니다. 하지만 구현 때에는, 보다 더 정확히 해야 합니다. 
* Transformer는 레이어 정규화(layer normalization)을 많이 사용하며, 꽤 중요합니다. 이전 블로그 포스팅 'Illustrated Transformer'에서는 이 것들 중 몇가지를 언급하긴 했지만, 이번 포스팅에서는 self-attention에 집중했습니다.
* vector를 표현하기 위해 더 많은 상자(box)들로 표현해야할 떄가 있습니다. 저는 이 상자들을 "zoom in"으로 표시했습니다. 예를 들어 다음과 같습니다:
<span class="tooltiptext">
* I used "words" and "tokens" interchangeably. But in reality, GPT2 uses Byte Pair Encoding to create the tokens in its vocabulary. This means the tokens are usually parts of words.
* The example we showed runs GPT2 in its inference/evaluation mode. That's why it's only processing one word at a time. At training time, the model would be trained against longer sequences of text and processing multiple tokens at once. Also at training time, the model would process larger batch sizes (512) vs. the batch size of one that evaluation uses.
* I took liberties in rotating/transposing vectors to better manage the spaces in the images. At implementation time, one has to be more precise.
* Transformers use a lot of layer normalization, which is pretty important. We've noted a few of these in the Illustrated Transformer, but focused more on self-attentionin this post.
* There are times when I needed to show more boxes to represent a vector. I indicate those as "zooming in". For example:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/zoom-in.png"/>
  <br />
</div>


## Part #2: The Illustrated Self-Attention <a name="part-2-illustrated-self-attention" href="#part-2-illustrated-self-attention">#</a>

<div class="tooltip" markdown="1">
이 글의 앞 부분에서 단어 ```it```을 처리하는 layer에 self-attention을 적용하는 것을 보여주기 위해 이 그림을 보여주었습니다.
<span class="tooltiptext">
Earlier in the post we showed this image to showcase self-attention being applied in a layer that is processing the word ```it```:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-1-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이번 섹션에서는, 어떻게 동작하는지 자세히 살펴보겠습니다. 각 개별 word에 무슨일이 일어나는지 이해하는 방향으로 알아보겠습니다. 많은 단일 vector들을 보여줄 것입니다. 실제 구현은 거대한 matrix를 서로 곱하여 수행됩니다. 하지만 여기서는, word 수준에서 어떤 일이 일어나는지 직관적 표현에 집중하겠습니다. 
<span class="tooltiptext">
In this section, we'll look at the details of how that is done. Note that we'll look at it in a way to try to make sense of what happens to individual words. That's why we'll be showing many single vectors. The actual implementations are done by multiplying giant matrices together. But I want to focus on the intuition of what happens on a word-level here.
</span>
</div>

### Self-Attention (without masking)
<div class="tooltip" markdown="1">
encoder block에서 계산된 최초의 self-attention을 살펴보는 것으로 부터 시작해봅시다. 한번에 4개의 token만 처리할 수 있는 작은(toy) transformer를 살펴보겠습니다.
<span class="tooltiptext">
Let's start by looking at the original self-attention as it's calculated in an encoder block. Let's look at a toy transformer block that can only process four tokens at a time.
</span>
</div>


<div class="tooltip" markdown="1">
Self-attention은 3개의 주요 단계가 있습니다:
<span class="tooltiptext">
Self-attention is applied through three main steps:
</span>
</div>

<div class="tooltip" markdown="1">
1. 각 경로 마다 Query, Key, Value 벡터를 생성합니다.
2. 각 input token 마다, query vector를 사용하여 모든 다른 key vector들에 대한 score를 계산합니다.
3. value vector에 그 조합된 score를 곱한 뒤 합산합니다.
<span class="tooltiptext">
1. Create the Query, Key, and Value vectors for each path.
2. For each input token, use its query vector to score against all the other key vectors
3. Sum up the value vectors after multiplying them by their associated scores.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/self-attention-summary.png"/>
  <br />
</div>

### 1- Create Query, Key, and Value Vectors
<div class="tooltip" markdown="1">
첫번째 경로를 봅시다. query를 받아서, 모든 key들과 비교할 것입니다. 각 key 별로 score를 계산해냅니다. self-attention에서의 첫번쨰 단계는 각 token 경로 별 3개의 vector를 계산하는 것 입니다 (지금부터는 attention head는 무시합니다):
<span class="tooltiptext">
Let's focus on the first path. We'll take its query, and compare against all the keys. That produces a score for each key. The first step in self-attention is to calculate the three vectors for each token path (let's ignore attention heads for now):
</span>
</div>


<div class="img-div-any-width" markdown="0">
  1) 각 input token 마다, weight matrix W^Q, W^K, W^V를 곱하여 query vector, key vector, value vector를 생성합니다. 
  <image src="/images/xlnet/self-attention-1.png"/>
  <br />
</div>

### 2- Score
<div class="tooltip" markdown="1">
이제 vector들을 갖게 되었고, #2번 단계를 위해서만 query 및 key vector를 사용합니다. 우리는 지금 첫번째 token을 집중해서 보고 있기 때문에, 그 token의 query를 모든 key vector들과 곱하여 4개의 token들 각각의 score를 얻습니다. 
<span class="tooltiptext">
Now that we have the vectors, we use the query and key vectors only for step #2. Since we're focused on the first token, we multiply its query by all the other key vectors resulting in a score for each of the four tokens.
</span>
</div>
<div class="img-div-any-width" markdown="0">  
  2) 현재의 query vector와 모든 key vector가 얼마나 잘 매칭되는지 score를 얻기 위해 곱셈(dot product) 연산을 합니다. 
  <image src="/images/xlnet/self-attention-2.png"/>
  <br />
</div>


### 3- Sum

<div class="tooltip" markdown="1">
우리는 이제 score들과 value vector들을 곱할 수 있습니다. 높은 score의 value는, 결과가 다 더해지고 난 뒤에, 결과 vector의 큰 비중을 차지하게 됩니다.
<span class="tooltiptext">
We can now multiply the scores by the value vectors. A value with a high score will constitute a large portion of the resulting vector after we sum them up.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  3) value vector들을 score들과 곱한 뒤 합산합니다. 
  <image src="/images/xlnet/self-attention-3-2.png"/>
  <br />
  <div class="tooltip" markdown="1">
  더 낮은 score일 수록 value vector가 더 투명하게 표시됩니다. 작은 수를 곱하는 것이 vector의 값을 희석하는지(작게 만드는지) 표현합니다.
  <span class="tooltiptext">
  The lower the score, the more transparent we're showing the value vector. That's to indicate how multiplying by a small number dilutes the values of the vector.
  </span>
  </div>
</div>

<div class="tooltip" markdown="1">
만약 각 경로마다 같은 동작을 수행한다면, 각 해당 token 마다, 적합한 context를 포함하는 token을 표현하는 vector를 얻게 됩니다. 그 값은 transformer block의 다음 하위 layer(feed-forward neural network)에 제공됩니다. 
<span class="tooltiptext">
If we do the same operation for each path, we end up with a vector representing each token containing the appropriate context of that token. Those are then presented to the next sublayer in the transformer block (the feed-forward neural network):
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/self-attention-summary.png"/>
  <br />
</div>

### The Illustrated Masked Self-Attention

<div class="tooltip" markdown="1">
우리는 지금까지 transformer의 self-attention 단계를 살펴보았고, 이제 masked self-attention에 대해 살펴보겠습니다. Masked self-attention은 self-attention과 같지만, #2 단계에서는 다릅니다. 모델이 2개의 token만을 input으로 가진다고 가정하고, 우리는 두번쨰 token을 처리하는 상황입니다. 이러한 경우에, 마지막 2개의 token은 masking 됩니다. 모델은 scoring 단계를 방해합니다. 즉, 기본적으로 앞으로 나올 token에 대한 score를 0으로 만들어서, 모델이 앞으로 나올 word를 반영할 수 없습니다:
<span class="tooltiptext">
Now that we've looked inside a transformer's self-attention step, let's proceed to look at masked self-attention. Masked self-attention is identical to self-attention except when it comes to step #2. Assuming the model only has two tokens as input and we're observing the second token. In this case, the last two tokens are masked. So the model interferes in the scoring step. It basically always scores the future tokens as 0 so the model can't peak to future words:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/masked-self-attention-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이러한 masking은 attention mask라고 불리는 matrix로 구현됩니다. 4개의 단어 sequence(예를 들어 "robot must obey orders")를 생각해보세요. language modeling 시나리오에서, 이 sequence는 4단계에 걸쳐 입력됩니다 -- word 당 하나씩 (모든 word는 token이라고 가정합니다). 모델은 배치로 동작하기 때문에, 전체 sequence (4단계를 갖는)를 한 배치로 처리하는 이 toy model의 배치 사이즈를 4로 가정할 수 있습니다.
<span class="tooltiptext">
This masking is often implemented as a matrix called an attention mask. Think of a sequence of four words ("robot must obey orders", for example). In a language modeling scenario, this sequence is absorbed in four steps -- one per word (assuming for now that every word is a token). As these models work in batches, we can assume a batch size of 4 for this toy model that will process the entire sequence (with its four steps) as one batch.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/transformer-decoder-attention-mask-dataset.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
matrix 형태에서, query matrix를 key matrix와 곱해서 score를 계산할 수 있습니다. cell에서 word 대신에 word와 관련된 query(또는 key) vector가 있다고 가정하고 다음과 같이 시각적으로 표현해보겠습니다. 
<span class="tooltiptext">
In matrix form, we calculate the scores by multiplying a queries matrix by a keys matrix. Let's visualize it as follows, except instead of the word, there would be the query (or key) vector associated with that word in that cell:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/queries-keys-attention-mask.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
곱셈 이후에, attention mask triangle을 적용합니다. masking 하고 싶은 cell들을 마이너스 무한대 또는 매우 큰 음수로 설정합니다 (예. GPT2에서는 -10억):
<span class="tooltiptext">
After the multiplication, we slap on our attention mask triangle. It set the cells we want to mask to -infinity or a very large negative number (e.g. -1 billion in GPT2):
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/transformer-attention-mask.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
각 행에 softmax를 취함으로써 self-attention에 사용하는 실제 score가 생성됩니다. 
<span class="tooltiptext">
Then, applying softmax on each row produces the actual scores we use for self-attention:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/transformer-attention-masked-scores-softmax.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이 score 테이블이 의미하는 것은 다음과 같습니다:
<span class="tooltiptext">
What this scores table means is the following:
</span>
</div>

<div class="tooltip" markdown="1">
* 모델이 dataset에서 첫번째 케이스(1번 행)를 처리할 때, 단 하나의 단어("robot")만을 포함하며, 그 단어에 모든(100%) attention을 갖습니다.
* 모델이 dataset에서 두번째 케이스(2번 행)을 처리할 때, "robot must"라는 단어들을 포함하며, "robot"에 48%, "must"에 52%의 attention을 갖으면서 단어 "must"를 처리합니다.
* 기타 등등
<span class="tooltiptext">
* When the model processes the first example in the dataset (row #1), which contains only one word ("robot"), 100% of its attention will be on that word.
* When the model processes the second example in the dataset (row #2), which contains the words ("robot must"), when it processes the word "must", 48% of its attention will be on "robot", and 52% of its attention will be on "must".
* And so on
</span>
</div>


### GPT-2 Masked Self-Attention
<div class="tooltip" markdown="1">
GPT-2의 masked attention에 대해 더 깊이 알아봅시다.
<span class="tooltiptext">
Let's get into more detail on GPT-2's masked attention.
</span>
</div>

#### Evaluation Time: Processing One Token at a Time
<div class="tooltip" markdown="1">
GPT-2가 masked self-attention이 동작하는 것과 똑같이 동작하도록 만들 수 있습니다. 하지만 evaluation 할 때에, 우리 모델이 각 iteration이 끝나고 하나의 새로운 word만 추가할 때, 이미 처리된 token에 대해 이전 경로를 따라 self-attention을 다시 계산하는 것은 비효율적 입니다.  
<span class="tooltiptext">
We can make the GPT-2 operate exactly as masked self-attention works. But during evaluation, when our model is only adding one new word after each iteration, it would be inefficient to recalculate self-attention along earlier paths for tokens which have already been processed.
</span>
</div>

<div class="tooltip" markdown="1">
이 경우에, 첫번째 token을 처리합니다 (지금은 `<s>`를 무시합니다).
<span class="tooltiptext">
In this case, we process the first token (ignoring `<s>` for now).
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-qkv-1-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
GPT-2는 ```a``` token의 key, value vector를 유지하고 있습니다. 모든 self-attention layer는 그 token에 대한 각각의 key, value vector를 유지합니다.
<span class="tooltiptext">
GPT-2 holds on to the key and value vectors of the the ```a``` token. Every self-attention layer holds on to its respective key and value vectors for that token:
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-qkv-2-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이제 다음 iteration에서, 모델이 단어 ```robot```을 처리할 때, query, key, value 쿼리(?)를 생성할 필요가 없습니다. 첫번째 iteration에서 저장한 것을 재사용 합니다. 
<span class="tooltiptext">
Now in the next iteration, when the model processes the word ```robot```, it does not need to generate query, key, and value queries for the ```a``` token. It just reuses the ones it saved from the first iteration:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-qkv-3-2.png"/>
  <br />
</div>

#### GPT-2 Self-attention: 1- Creating queries, keys, and values

<div class="tooltip" markdown="1">
단어 ```it```를 처리하는 모델을 가정해봅시다. 하단 block의 경우, 그 token의 input은 `it`의 embedding + 슬롯 #9에 대한 positional encoding이 됩니다. 
<span class="tooltiptext">
Let's assume the model is processing the word ```it```. If we're talking about the bottom block, then its input for that token would be the embedding of `it` + the positional encoding for slot #9:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-1.png"/>
  <br />
</div>


<div class="tooltip" markdown="1">
Transformer에서 모든 block은 각자의 weight를 갖습니다 (이 글의 후반부에서 설명하겠습니다). 가장 먼저 만나는 것은 query, key, value를 생성하는 데 사용하는 weight matrix입니다. (?)
<span class="tooltiptext">
Every block in a transformer has its own weights (broken down later in the post). The first we encounter is the weight matrix that we use to create the queries, keys, and values.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-2.png"/>
  <div class="tooltip" markdown="1">
  Self-attention은 input을 weight matrix 곱합니다 (그리고 여기서 표현하지는 않았지만, bias vector를 더해줍니다). 
  <span class="tooltiptext">
  <br />
  Self-attention multiplies its input by its weight matrix (and adds a bias vector, not illustrated here).
  </span>
  </div>
</div>

<div class="tooltip" markdown="1">
이 곱셈 연산은 기본적으로 단어 `it`에 대한 query, key, value vector의 연결된 vector를 생성합니다.
<span class="tooltiptext">
The multiplication results in a vector that's basically a concatenation of the query, key, and value vectors for the word `it`.
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-3.png"/>
  <div class="tooltip" markdown="1">
  input vector에 attention weight vector를 곱함으로써 (그리고 나중에 bias vector를 더함으로써), 이 token에 대한 key, value, query vector를 생성합니다.
  <span class="tooltiptext">
  <br />
  Multiplying the input vector by the attention weights vector (and adding a bias vector aftwards) results in the key, value, and query vectors for this token.
  </span>
  </div>
</div>

#### GPT-2 Self-attention: 1.5- Splitting into attention heads

<div class="tooltip" markdown="1">
이전 예제에서, "multi-head" 부분을 건너뛰고 self-attention을 바로 살펴봤습니다. 이제 그 개념에 대해 약간의 설명을 하는 것이 좋겠습니다. Self attention은 Q, K, V vector의 다른 부분들에 대해 여러번 수행됩니다. attention heads "분할(Splitting)"은 긴 vector를 matrix로 단순히 재구성하는 것 입니다. small GPT2는 12개의 attention head를 갖으며, 재구성된 matrix의 첫번째 차원(dimension)이 됩니다. 
<span class="tooltiptext">
In the previous examples, we dove straight into self-attention ignoring the "multi-head" part. It would be useful to shed some light on that concept now. Self attention is conducted multiple times on different parts of the Q,K,V vectors. "Splitting" attention heads is simply reshaping the long vector into a matrix. The small GPT2 has 12 attention heads, so that would be the first dimension of the reshaped matrix:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-split-attention-heads-1.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이전 예제에서, attention head 안에서 어떤 일이 일어나는지 살펴보았습니다. 다수의 attention-heads를 생각하는 방법은 아래와 같습니다 (만약 12개의 attention head의 3개만을 그림으로 표현한다면):
<span class="tooltiptext">
In the previous examples, we've looked at what happens inside one attention head. One way to think of multiple attention-heads is like this (if we're to only visualize three of the twelve attention heads):
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-split-attention-heads-2.png"/>
  <br />
</div>

#### GPT-2 Self-attention: 2- Scoring
<div class="tooltip" markdown="1">
우리는 이제 score를 계산하는 것을 처리합니다 -- 우리가 하나의 attention head를 바라보고 있음 (그리고 다른 것들은 비슷한 연산을 수행함)을 알고 있습니다:
<span class="tooltiptext">
We can now proceed to scoring -- knowing that we're only looking at one attention head (and that all the others are conducting a similar operation):
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-scoring.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이제 token은 다른 token들의 key에 대해 score 값을 얻을 수 있습니다 (이전 iteration에서 attention head 1번에서 계산되었습니다).
<span class="tooltiptext">
Now the token can get scored against all of keys of the other tokens (that were calculated in attention head #1 in previous iterations):
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-scoring-2.png"/>
  <br />
</div>


#### GPT-2 Self-attention: 3- Sum
<div class="tooltip" markdown="1">
이전에 살펴본 것과 같이, 각 value를 각 score와 곱하고, 그 결과들을 합산해서, attention-head #1를 위한 self-attention 결과를 만듭니다.
<span class="tooltiptext">
As we've seen before, we now multiply each value with its score, then sum them up, producing the result of self-attention for attention-head #1:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-multihead-sum-1.png"/>
  <br />
</div>


#### GPT-2 Self-attention: 3.5- Merge attention heads

<div class="tooltip" markdown="1">
여러 attention head를 다루기 위한 방법은, 먼저 이들을 하나의 vector로 접합(concat)하는 것 입니다.
<span class="tooltiptext">
The way we deal with the various attention heads is that we first concatenate them into one vector:
</span>
</div>
<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-merge-heads-1.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
하지만 이 vector는 아직 다음 하위 layer로 전달될 준비가 되지 않았습니다. 먼저 hidden state의 이 괴물을 동질적(homogenous) 표현(representation)으로 바꿔야 합니다. 
<span class="tooltiptext">
But the vector isn't ready to be sent to the next sublayer just yet. We need to first turn this Frankenstein's-monster of hidden states into a homogenous representation.
</span>
</div>

#### GPT-2 Self-attention: 4- Projecting

<div class="tooltip" markdown="1">
우리는 모델이 연결된 self-attention 결과를, feed-forward neural network가 처리할 수 있는 하나의 vector로 잘 mapping시킬 수 있도록 학습하도록 만들 것 입니다. 
<span class="tooltiptext">
We'll let the model learn how to best map concatenated self-attention results into a vector that the feed-forward neural network can deal with. Here comes our second large weight matrix that projects the results of the attention heads into the output vector of the self-attention sublayer:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-project-1.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
그리고 이렇게, 다음 layer로 보낼 수 있는 vector를 생성했습니다.
<span class="tooltiptext">
And with this, we have produced the vector we can send along to the next layer:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-self-attention-project-2.png"/>
  <br />
</div>

#### GPT-2 Fully-Connected Neural Network: Layer #1

<div class="tooltip" markdown="1">
fully-connected neural network은 self-attention이 representation에 적합한 context를 포함시킨 뒤, block이 input token을 처리하는 곳 입니다. 이 것은 두 개의 layer로 구성되어 있습니다. 첫 번째 layer는 모델 사이즈의 4배 입니다 (GPT2 small의 경우 768 이므로, 이 network는 768*4 = 3072 unit 입니다). 왜 4배 일까요? 그 것은 단순히 최초의 transformer에서 사용한 값과 같습니다 (모델 차원이 512 였고, layer #1은 2048 이었습니다). 이 것은 transformer 모델에 주어진(처리해야 하는) task들을 다루기에 충분한 representation 능력/용량을 주는 것으로 보입니다.
<span class="tooltiptext">
The fully-connected neural network is where the block processes its input token after self-attention has included the appropriate context in its representation. It is made up of two layers. The first layer is four times the size of the model (Since GPT2 small is 768, this network would have 768*4 = 3072 units). Why four times? That's just the size the original transformer rolled with (model dimension was 512 and layer #1 in that model was 2048). This seems to give transformer models enough representational capacity to handle the tasks that have been thrown at them so far.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-mlp1.gif"/>
  <br />
  <div class="tooltip" markdown="1">
  (bias vector 생략)
  <span class="tooltiptext">
  (Not shown: A bias vector)
  </span>
  </div>
</div>



#### GPT-2 Fully-Connected Neural Network: Layer #2 - Projecting to model dimension

<div class="tooltip" markdown="1">
두 번째 layer는 첫 번째 layer의 결과를 모델 차원(dimension; small GPT2의 경우 768)으로 다시 투영합니다. 이 곱셈 연산의 결과는 이 token에 대한 transformer block의 결과입니다.
<span class="tooltiptext">
The second layer projects the result from the first layer back into model dimension (768 for the small GPT2). The result of this multiplication is the result of the transformer block for this token.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-mlp-2.gif"/>
  <br />
  <div class="tooltip" markdown="1">
  (bias vector 생략)
  <span class="tooltiptext">
  (Not shown: A bias vector)
  </span>
</div>

### You've Made <span style="color:#7F34AA">It</span>!
<div class="tooltip" markdown="1">
이 것이 우리가 다룰 transformer block의 가장 상세한 버전입니다! 당신은 transformer language model 안에서 일어나는 대다수의 것들을 알게 되었습니다. 요약하자면, 우리의 input vector는 이러한 weight matrix들을 만납니다: 
<span class="tooltiptext">
That's the most detailed version of the transformer block we'll get into! You now pretty much have the vast majority of the picture of what happens inside of a transformer language model. To recap, our brave input vector encounters these weight matrices:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-transformer-block-weights-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
그리고 각 block 마다 이러한 weight들의 세트를 가지고 있습니다. 반면에, 모델은 하나의 token embedding matrix와 하나의 positional encoding matrix 만을 가지고 있습니다. 
<span class="tooltiptext">
And each block has its own set of these weights. On the other hand, the model has only one token embedding matrix and one positional encoding matrix:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-weights-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
만약 모델의 parameter 전체를 보고 싶다면, 여기에 집계해두었습니다:
<span class="tooltiptext">
If you want to see all the parameters of the model, then I have tallied them here:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/gpt2-117-parameters.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
어떤 이유로 117M개가 아닌, total 124M개의 parameter 수가 나왔습니다. 이유는 모르겠지만, publish된 code들에서 보여지는 숫자 입니다 (만약 제가 틀린 경우 수정해주세요). 
<span class="tooltiptext">
They add up to 124M parameters instead of 117M for some reason. I'm not sure why, but that's how many of them seems to be in the published code (please correct me if I'm wrong).
</span>
</div>

## Part 3: Beyond Language Modeling <a href="#part-3-beyond-language-modeling" name="part-3-beyond-language-modeling">#</a>
<div class="tooltip" markdown="1">
decoder-only transformer는 language modeling 이상의 가능성들을 계속 보여줍니다. 위와 유사한 그림으로 설명할 수 있는 성공을 보여준 application들이 많이 있습니다 (?). 이러한 application 몇 개를 보면서 이번 포스팅을 마치고자 합니다. 
<span class="tooltiptext">
The decoder-only transformer keeps showing promise beyond language modeling. There are plenty of applications where it has shown success which can be described by similar visuals as the above. Let's close this post by looking at some of these applications
</span>
</div>

### Machine Translation
<div class="tooltip" markdown="1">
번역(translation)을 하는데에 encoder가 필요하지 않습니다. 이 task를 decoder-only transformer로 처리할 수 있습니다:
<span class="tooltiptext">
An encoder is not required to conduct translation. The same task can be addressed by a decoder-only transformer:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/decoder-only-transformer-translation.png"/>
  <br />
</div>


### Summarization

<div class="tooltip" markdown="1">
요약(Summarization)은 첫 번째 decoder-only transformer가 학습된 task 입니다. 즉, (목차 앞쪽의 서두 부분을 제외하고) 위키피디아 아티클을 읽고 요약하도록 학습했습니다. 실제 서두 부분은 학습 dataset에서 label로 사용되었습니다:
<span class="tooltiptext">
This is the task that the first decoder-only transformer was trained on. Namely, it was trained to read a wikipedia article (without the opening section before the table of contents), and to summarize it. The actual opening sections of the articles were used as the labels in the training datasest:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/wikipedia-summarization.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이 논문에서는 위키피디아 아티클에 대해 모델을 학습시켰고, 학습된 모델은 아티클을 요약할 수 있었습니다:
<span class="tooltiptext">
The paper trained the model against wikipedia articles, and thus the trained model was able to summarize articles:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/decoder-only-summarization.png"/>
  <br />
</div>

### Transfer Learning
<div class="tooltip" markdown="1">
[Sample Efficient Text Summarization Using a Single Pre-Trained Transformer](https://arxiv.org/abs/1905.08836) 논문에서, decoder-only transformer는 먼저 language model에 대해 pre-train 하고, 요약(summary)에 대해 finetuning 했습니다. 이 것은 제한된 data 설정(?)에서 encoder-decoder transformer를 pre-train하는 것 보다 더 좋은 결과를 보였습니다.
<span class="tooltiptext">
In [Sample Efficient Text Summarization Using a Single Pre-Trained Transformer](https://arxiv.org/abs/1905.08836), a decoder-only transformer is first pre-trained on language modeling, then finetuned to do summarization. It turns out to achieve better results than a pre-trained encoder-decoder transformer in limited data settings.
</span>
</div>

<div class="tooltip" markdown="1">
GPT2 논문도 language modeling에 대해 pre-train한 뒤에 요약(summary)의 결과를 보여줍니다.
<span class="tooltiptext">
The GPT2 paper also shows results of summarization after pre-training the model on language modeling.
</span>
</div>

### Music Generation
<div class="tooltip" markdown="1">
[Music Transformer](https://magenta.tensorflow.org/music-transformer)는 decoder-only transformer를 사용하여 expressive timing과 dynamic한 음악을 생성합니다. "Music Modeling"은 language modeling과 같습니다 -- 모델이 unsupervise한 방법으로 음악을 학습하도록 하고, 샘플 출력하도록 합니다 (우리가 이전에 "rambling"이라고 불렀습니다). 
<span class="tooltiptext">
The [Music Transformer](https://magenta.tensorflow.org/music-transformer) uses a decoder-only transformer to generate music with expressive timing and dynamics. "Music Modeling" is just like language modeling -- just let the model learn music in an unsupervised way, then have it sample outputs (what we called "rambling", earlier).
</span>
</div>

<div class="tooltip" markdown="1">
이 시나리오에서 음악이 어떻게 reparesent 되는지 궁금할 것 입니다. language modeling은 단어의 일부인 문자(character), 단어(word), 토큰(token) 등이 vector representation을 통해 표현되었습니다. 음악 연주에서 (지금은 피아노를 생각해보세요), 우리는 음표를 표현해야 하지만 velocity도 표현해야 합니다 -- 피아노 건반이 얼마나 세게 눌렀는지 측정. 
<span class="tooltiptext">
You might be curious as to how music is represented in this scenario. Remember that language modeling can be done through vector representations of either characters, words, or tokens that are parts of words. With a musical performance (let's think about the piano for now), we have to represent the notes, but also velocity -- a measure of how hard the piano key is pressed.
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/music-transformer-performance-encoding-3.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
연주는 이러한 일련의 one-hot vector들일 뿐입니다. midi 파일은 이러한 포맷으로 변환된ㄹ 수 있습니다. 이 논문에서는 다음과 같은 입력 순서 예시가 있습니다:
<span class="tooltiptext">
A performance is just a series of these one-hot vectors. A midi file can be converted into such a format. The paper has the following example input sequence:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/music-representation-example.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이 입력 순서에 대한 one-hot vector는 이렇게 모양일 것 입니다. 
<span class="tooltiptext">
The one-hot vector representation for this input sequence would look like this:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/music-transformer-input-representation-2.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
저는 Music Transformer에서 self-attention을 표현하는 이 논문의 그림을 좋아합니다. 여기에 주석을 조금 달았습니다. 
<span class="tooltiptext">
I love a visual in the paper that showcases self-attention in the Music Transformer. I've added some annotations to it here:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/gpt2/music-transformer-self-attention-2.png"/>
  <br />
  <div class="tooltip" markdown="1">
  "그림 8: 이 작품은 반복되는 삼각형 형태를 가지고 있습니다. query는 뒷쪽 peak들 중 하나에 있고, 곡의 시작부분에 이르기까지 peak의 모든 이전 고음에 attention을 줍니다." ..."[이] 그림은 query (모든 attention 선의 source)와 attention을 받는 이전 메모리(더 많은 softmax 확률을 받는 음표(note)가 강조됨)를 보여줍니다 (?). attention line의 색상은 서로 다른 head에 해당하고 두께는 softmax 확률의 가중치(weight)에 해당합니다."
  <span class="tooltiptext">
  "Figure 8: This piece has a recurring triangular contour. The query is at one of the latter peaks and it attends to all of the previous high notes on the peak, all the way to beginning of the piece." ... "[The] figure shows a query (the source of all the attention lines) and previous memories being attended to (the notes that are receiving more softmax probabiliy is highlighted in). The coloring of the attention lines correspond to different heads and the width to the weight of the softmax probability."
  </span>
  </div>
</div>

<div class="tooltip" markdown="1">
악보 representation에 대해 부족하다면 [이 영상](https://www.youtube.com/watch?v=ipzR9bhei_o)을 참고해보세요.
<span class="tooltiptext">
If you're unclear on this representation of musical notes, [check out this video](https://www.youtube.com/watch?v=ipzR9bhei_o).
</span>
</div>

## Conclusion
<div class="tooltip" markdown="1">
이것으로 GPT2로의 여정과 상위 모델인 decoder-only transformer에 대한 탐색을 마치겠습니다. 이 포스팅을 통해 self-attention에 대한 더 깊은 이해와 transformer 내부에서 일어나는 것들에 대해 이해하는 데에 더 편안하기를 바랍니다.
<span class="tooltiptext">
This concludes our journey into the GPT2, and our exploration of its parent model, the decoder-only transformer. I hope that you come out of this post with a better understanding of self-attention and more comfort that you understand more of what goes on inside a transformer.
</span>
</div>

## Resources

<div class="tooltip" markdown="1">
* OpenAI의 [GPT2 구현](https://github.com/openai/gpt-2) 
* [Hugging Face](https://huggingface.co/)의 [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) 라이브러리 및 GPT2, BERT 구현, Transformer-XL, XLNet, 최신 transformer model들을 확인해보세요.
<span class="tooltiptext">
* The [GPT2 Implementation](https://github.com/openai/gpt-2) from OpenAI
* Check out the [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) library from [Hugging Face](https://huggingface.co/) in addition to GPT2, it implements BERT, Transformer-XL, XLNet and other cutting-edge transformer models.
</span>
</div>

## Acknowledgements
<div class="tooltip" markdown="1">
[Lukasz Kaiser](https://twitter.com/lukaszkaiser), [Mathias Müller](https://www.cl.uzh.ch/de/people/team/compling/mmueller.html), [Peter J. Liu](https://twitter.com/peterjliu), [Ryan Sepassi](https://twitter.com/rsepassi), [Mohammad Saleh](https://www.linkedin.com/in/mohammad-saleh-39614224/)님들께 이 포스팅의 이전 버전에서 피드백을 주셔서 감사합니다.
<span class="tooltiptext">
Thanks to [Lukasz Kaiser](https://twitter.com/lukaszkaiser), [Mathias Müller](https://www.cl.uzh.ch/de/people/team/compling/mmueller.html), [Peter J. Liu](https://twitter.com/peterjliu), [Ryan Sepassi](https://twitter.com/rsepassi) and [Mohammad Saleh](https://www.linkedin.com/in/mohammad-saleh-39614224/) for feedback on earlier versions of this post.
</span>
</div>

<div class="tooltip" markdown="1">
의견이나 수정이 있다면 [@JayAlammar](https://twitter.com/JayAlammar)로 tweet 해주세요.
<span class="tooltiptext">
Comments or corrections? Please tweet me at [@JayAlammar](https://twitter.com/JayAlammar)
</span>
</div>

<!--
### Just Add Memory

So far, our models have only considered the keys and values from the current segment. What's to stop us from adding a bunch more keys and values representing words from previous tokens? Nothing stops us! That's exactly what memory is in this context


<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/memory-self-attention.png"/>
  <br />
</div>

And there we have it! The model can now incorporate all previous tokens in previous segments into the self-attention calculation.


Let's go over an example to make sure we're on the same page. Say we want to process the first eight words of The Second Law using a toy memory-transformer with one block and four token segment length. From now on, we'll show vectors vertically rather than horizontally so we can squeeze them into matrices:


### Memory-Compression

In practice, we can very quickly run out memory if we memorize the keys and values of all previous tokens in a long text sequence. Here, we can turn to the idea of compressing this memory to save space. If we're to rotate our key and value vectors like the following:

<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/keys-and-values.png"/>
  <br />
</div>

We can compress it by compressing every three vectors into one:

<div class="img-div-any-width" markdown="0">
  <image src="/images/xlnet/transformer-memory-compression.png"/>
  <br />
</div>

The compression is done using a convolutional neural network which learns (during training time) how to effectively turn every three key vectors into a a single vector. Likewise with the values vectors. Again, in technical jargon, the compression is done using a CNN with a kernel size of 3 and a stride of 3.
-->
