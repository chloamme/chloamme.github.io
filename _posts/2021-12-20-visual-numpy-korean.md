---
layout: post
title: NumPy 및 데이터 표현에 대한 시각적 소개
subtitle: A Visual Intro to NumPy and Data Representation
categories: translation
tags: [NumPy, Data Representation]
---

<div class="tooltip" markdown="1">
> 이 글은 [Jay Alammar님의 글](https://jalammar.github.io/visual-numpy/)을 번역한 글입니다. [[추가정보](#additional-info)]
<span class="tooltiptext">
This post is a translated version of [A Visual Intro to NumPy and Data Representation](https://jalammar.github.io/visual-numpy/) by Jay Alammar.
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-array.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
[NumPy](https://www.numpy.org/) 패키지는 파이썬 생태계에서 데이터 분석, 머신러닝, 과학적 컴퓨팅의 핵심 요소입니다. 벡터(vector)와 행렬(matrix)의 조작과 크런칭(대량 고속 처리)을 엄청나게 단순화시킵니다. (scikit-learn, SciPy, pandas, tensorflow 등) 파이썬의 주요 패키지는 NumPy를 인프라스트럭쳐의 기본/근본 부분으로 사용합니다. 숫자 데이터를 쪼개어 분석하는 능력 외에도 NumPy를 마스터하는 것은 이러한 라이브러리의 고급 활용 시 처리하거나 디버깅할 때에 우위를 확보할 수 있습니다. 
<span class="tooltiptext">
The [NumPy](https://www.numpy.org/) package is the workhorse of data analysis, machine learning, and scientific computing in the python ecosystem. It vastly simplifies manipulating and crunching vectors and matrices. Some of python's leading package rely on NumPy as a fundamental piece of their infrastructure (examples include scikit-learn, SciPy, pandas, and tensorflow). Beyond the ability to slice and dice numeric data, mastering numpy will give you an edge when dealing and debugging with advanced usecases in these libraries.
</span>
</div>


<div class="tooltip" markdown="1">
이번 포스팅에서는 NumPy를 사용하기 위한 주요 방법들을 살펴보고, 머신러닝 모델에 넣기 전에 (테이블, 이미지, 텍스트 등) 다른 데이터 타입을 표현하는 방법에 대해 알아보겠습니다. 
<span class="tooltiptext">
In this post, we'll look at some of the main ways to use NumPy and how it can represent different types of data (tables, images, text...etc) before we can serve them to machine learning models.
</span>
</div>

<!--more-->

```python
import numpy as np
```


## Creating Arrays (배열 생성)


<div class="tooltip" markdown="1">
파이썬 리스트를 `np.array()`에 전달하여, NumPy 배열(즉, 강력한 [ndarray](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html))을 생성할 수 있습니다. 이 케이스에서 파이썬은 오른쪽에 보이는 배열을 생성합니다. 
<span class="tooltiptext">
We can create a NumPy array (a.k.a. the mighty [ndarray](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html)) by passing a python list to it and using `np.array()`. In this case, python creates the array we can see on the right here:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/create-numpy-array-1.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
NumPy가 배열의 값을 초기화해주기를 바라는 케이스들이 종종 있습니다. NumPy는 이런 경우에 ones(), zeros(), random.random()과 같은 메서드를 제공합니다. 우리는 생성하기 원하는 element의 개수만 전달하면 됩니다: 
<span class="tooltiptext">
There are often cases when we want NumPy to initialize the values of the array for us. NumPy provides methods like ones(), zeros(), and random.random() for these cases. We just pass them the number of elements we want it to generate:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/create-numpy-array-ones-zeros-random.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
배열을 생성하면, 재밌는 방법들로 배열을 조작(manipulate)할 수 있습니다. 
<span class="tooltiptext">
Once we've created our arrays, we can start to manipulate them in interesting ways.
</span>
</div>

## Array Arithmetic (배열 산술연산)

<div class="tooltip" markdown="1">
NumPy 배열의 유용성을 설명하기 위해 두 개의 NumPy 배열을 생성합니다. `data`와 `ones`라고 부르겠습니다:
<span class="tooltiptext">
Let's create two NumPy arrays to showcase their usefulness. We'll call them `data` and `ones`:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-arrays-example-1.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
position-wise 덧셈(즉, 각 행마다의 값들을 더하기)은 ```data + ones```를 입력하면 됩니다. 간단합니다. 
<span class="tooltiptext">
Adding them up position-wise (i.e. adding the values of each row) is as simple as typing ```data + ones```:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-arrays-adding-1.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
이러한 도구들을 배우기 시작했을 때, loop를 돌면서 계산하는 것을 프로그래밍할 필요가 없는 이런 추상화가 신선하게 느껴졌습니다. 상위 레벨에서 문제를 생각해볼 수 있게 만드는 멋진 추상화입니다.
<span class="tooltiptext">
When I started learning such tools, I found it refreshing that an abstraction like this makes me not have to program such a calculation in loops. It's a wonderful abstraction that allows you to think about problems at a higher level.
</span>
</div>

<div class="tooltip" markdown="1">
이런 방식으로 할 수 있는 것은 덧셈 뿐만이 아닙니다:
<span class="tooltiptext">
And it's not only addition that we can do this way:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-array-subtract-multiply-divide.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
배열과 하나의 숫자값을 연산하고 싶을 때도 종종 있습니다 (우리는 이 것을 벡터와 스칼라간 연산이라고 부릅니다). 예를 들어, 마일로 표현된 거리 값을 가지고 있는 배열을 킬로미터로 변환하려고 한다고 가정해 보겠습니다. 간단하게 ```data * 1.6```라고 하면 됩니다:
<span class="tooltiptext">
There are often cases when we want to carry out an operation between an array and a single number (we can also call this an operation between a vector and a scalar). Say, for example, our array represents distance in miles, and we want to convert it to kilometers. We simply say ```data * 1.6```:
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-array-broadcast.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
NumPy가 이 연산(`*`)이 곱셈이 각 셀에서 수행되어야 한다는 것을 의미하는 것임을 이해하는 방법을 주목/확인해보세요. 그 컨셉을 *브로트캐스팅*이라고 합니다. 매우 유용하죠.
<span class="tooltiptext">
See how NumPy understood that operation to mean that the multiplication should happen with each cell? That concept is called *broadcasting*, and it's very useful.
</span>
</div>


## Indexing (인덱싱)

<div class="tooltip" markdown="1">
파이썬 리스트에서 슬라이싱을 할 수 있는 모든 방법으로 NumPy에서도 인덱싱과 슬라이싱을 할 수 있습니다:
<span class="tooltiptext">
We can index and slice NumPy arrays in all the ways we can slice python lists:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-array-slice.png"/>
  <br />
</div>

## Aggregation (집계)

<div class="tooltip" markdown="1">
NumPy의 추가적인 이점은 집계 함수(aggregation function)들입니다. 
<span class="tooltiptext">
Additional benefits NumPy gives us are aggregation functions:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-array-aggregation.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
`min`, `max`, `sum` 뿐만아니라 평균을 구하는 `mean`, 모든 element들을 곱한 결과를 구하는 `prod`, 표준 편차를 구하는 `std`, [다른 많은 연산들](https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html)을 얻을 수 있습니다.
<span class="tooltiptext">
In addition to `min`, `max`, and `sum`, you get all the greats like `mean` to get the average, `prod` to get the result of multiplying all the elements together, `std` to get standard deviation, and [plenty of others](https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html).
</span>
</div>


## In more dimensions (더 높은 차원)

<div class="tooltip" markdown="1">
현재가지 우리가 살펴본 예제들은 1차원인 벡터들이었습니다. NumPy의 정수는, 우리가 지금까지 배운 능력들을 어떤 차원에도 적용할 수 있다는 것입니다. 
<span class="tooltiptext">
All the examples we've looked at deal with vectors in one dimension. A key part of the beauty of NumPy is its ability to apply everything we've looked at so far to any number of dimensions.
</span>
</div>

### Creating Matrices (배열 생성)

<div class="tooltip" markdown="1">
아래와 같은 모양의 파이썬의 중첩 리스트를 전달하여, NumPy가 이를 나타내는 행렬을 생성하도록 할 수 있습니다:
<span class="tooltiptext">
We can pass python lists of lists in the following shape to have NumPy create a matrix to represent them:
</span>
</div>

```python
np.array([[1,2],[3,4]])
```

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-array-create-2d.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
우리가 생성하려는 배열의 차원을 기술하는 튜플을 (파라미터로) 전달한다면, (`ones()`, `zeros()`, `random.random()` 등) 위에서 살펴본 메서드들을 사용할 수 있습니다. 
<span class="tooltiptext">
We can also use the same methods we mentioned above (`ones()`, `zeros()`, and `random.random()`) as long as we give them a tuple describing the dimensions of the matrix we are creating:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-matrix-ones-zeros-random.png"/>
  <br />
</div>


### Matrix Arithmetic (행렬 산술연산)

<div class="tooltip" markdown="1">
두 행렬의 크기가 같다면, 산술연산자 (`+-*/`)를 사용해서 행렬을 더하거나 곱할 수 있습니다. NumPy는 연산을 position-wise로 처리합니다:
<span class="tooltiptext">
We can add and multiply matrices using arithmetic operators (`+-*/`) if the two matrices are the same size. NumPy handles those as position-wise operations:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-matrix-arithmetic.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
(행렬이 오직 1개의 열 또는 1개의 행을 가지고 있는 경우 등) 서로 크기가 다른 차원이 1개인 경우에만, 크기가 다른 행렬에 대해 이러한 산술 연산을 수행할 수 있습니다. 이 경우에 NumPy는 연산에 브로드캐스트를 적용합니다.
<span class="tooltiptext">
We can get away with doing these arithmetic operations on matrices of different size only if the different dimension is one (e.g. the matrix has only one column or one row), in which case NumPy uses its broadcast rules for that operation:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-matrix-broadcast.png"/>
  <br />
</div>


### Dot Product (내적)


<div class="tooltip" markdown="1">
산술 연산과의 주요 차이점은 내적을 사용한 [행렬 곱셈](https://www.mathsisfun.com/algebra/matrix-multiplying.html)의 경우입니다. NumPy는 행렬에 다른 행렬과 내적 연산을 수행하는데 사용할 수 있는 `dot()` 메서드를 제공합니다:
<span class="tooltiptext">
A key distinction to make with arithmetic is the case of [matrix multiplication](https://www.mathsisfun.com/algebra/matrix-multiplying.html) using the dot product. NumPy gives every matrix a `dot()` method we can use to carry-out dot product operations with other matrices:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-matrix-dot-product-1.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
행렬의 차원 정보를 그림 하단에 추가하여, 두 행렬이 서로 마주하는 면이 서로 같은 차원을 가지고 있어야 함을 강조했습니다. 이 연산을 아래와 같이 시각화하여 표현할 수 있습니다.
<span class="tooltiptext">
I've added matrix dimensions at the bottom of this figure to stress that the two matrices have to have the same dimension on the side they face each other with. You can visualize this operation as looking like this:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-matrix-dot-product-2.png"/>
  <br />
</div>

### Matrix Indexing (행렬 인덱싱)

<div class="tooltip" markdown="1">
인덱싱과 슬라이싱은 행렬을 다룰 때 더 유용합니다:
<span class="tooltiptext">
Indexing and slicing operations become even more useful when we're manipulating matrices:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-matrix-indexing.png"/>
  <br />
</div>


### Matrix Aggregation (행렬 집계연산)

<div class="tooltip" markdown="1">
벡터를 집계했던 것과 동일한 방법은 행렬도 집계 연산을 할 수 있습니다:
<span class="tooltiptext">
We can aggregate matrices the same way we aggregated vectors:
</span>
</div>
<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-matrix-aggregation-1.png"/>
  <br />
</div>

<br />


<div class="tooltip" markdown="1">
행렬의 모든 값을 집계할 수 있을 뿐만 아니라, `axis` 파라미터를 사용하여 행 또는 영을 집계할 수도 있습니다.
<span class="tooltiptext">
Not only can we aggregate all the values in a matrix, but we can also aggregate across the rows or columns by using the `axis` parameter:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-matrix-aggregation-4.png"/>
  <br />
</div>


## Transposing and Reshaping (전치 및 재구조화/재배열)

<div class="tooltip" markdown="1">
행렬을 다룰 때 일반적으로(많은 경우) 회전을 시킬 필요가 있습니다. 우리가 두 행렬에 내적을 취할 필요가 있거나 두 행렬이 공유하는(맞대는) 차원을 맞출 필요가 있을 때입니다. NumPy 배열을 행렬의 전치를 구하는 `T`라는 편리한 기능을 가지고 있습니다. 
<span class="tooltiptext">
A common need when dealing with matrices is the need to rotate them. This is often the case when we need to take the dot product of two matrices and need to align the dimension they share. NumPy arrays have a convenient property called `T` to get the transpose of a matrix:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-transpose.png"/>
  <br />
</div>

<br />


<div class="tooltip" markdown="1">
고급 유즈케이스로, 특정 행렬의 차원을 서로 바꾸는(switch) 필요를 느낄 수 있습니다. 데이터셋과 다른 입력이 있고, 그 입력이 특정 shape일 것을 모델이 기대하는 머신러닝 어플리케이션에서 종종 발생합니다. NumPy의 `reshape()` 메서드는 이러한 케이스에서 유용합니다. 원하는 행렬의 차원을 전달하기만 하면 됩니다. 만약 -1을 차원 값으로 전달하면, NumPy는 그 행렬의 정보를 기반으로 정확한 차원 값을 계산해냅니다.
<span class="tooltiptext">
In more advanced use case, you may find yourself needing to switch the dimensions of a certain matrix. This is often the case in machine learning applications where a certain model expects a certain shape for the inputs that is different from your dataset. NumPy's `reshape()` method is useful in these cases. You just pass it the new dimensions you want for the matrix. You can pass -1 for a dimension and NumPy can infer the correct dimension based on your matrix:
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-reshape.png"/>
  <br />
</div>


## Yet More Dimensions (한층 더 높은 차원)

<div class="tooltip" markdown="1">
NumPy는 앞서 언급한 모든 것을 모든 차원수에 대해 수행할 수 있습니다. 중심이되는 자료 구조를 ndarray(N-Dimensional Array; N-차원 배열)이라고 부르는 이유입니다. 
<span class="tooltiptext">
NumPy can do everything we've mentioned in any number of dimensions. Its central data structure is called ndarray (N-Dimensional Array) for a reason.
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-3d-array.png"/>
  <br />
</div>


<div class="tooltip" markdown="1">
많은 방법들 중에서, 새로운 차원을 다루는 방법은 콤마(`,`)를 NumPy 함수의 파라미터에 추가하는 것입니다. 
<span class="tooltiptext">
In a lot of ways, dealing with a new dimension is just adding a comma to the parameters of a NumPy function:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-3d-array-creation.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
참고: 3-차원 NumPy 배열을 출력하려고 할 때, 텍스트 출력은 여기에 표현한 것과 다르게 배열을 시각화합니다. NumPy의 n-차원 배열을 출력하는 순서는 마지막 축(axis)이 가장 빠르게 반복되고, 첫번째 축이 가장 느립니다. 이 것은 ```np.ones((4,3,2))```이 아래와 같이 출력됨을 의미합니다:
<span class="tooltiptext">
Note: Keep in mind that when you print a 3-dimensional NumPy array, the text output visualizes the array differently than shown here.  NumPy's order for printing n-dimensional arrays is that the last axis is looped over the fastest, while the first is the slowest. Which means that ```np.ones((4,3,2))``` would be printed as:
</span>
</div>

```python
array([[[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]]])
```

## Practical Usage (실질적인 사용법)

<div class="tooltip" markdown="1">
마무리입니다. 다음은 NumPy가 도움이 될 만한 몇가지 유용한 것들의 예입니다. 
<span class="tooltiptext">
And now for the payoff. Here are some examples of the useful things NumPy will help you through.
</span>
</div>


### Formulas (수식)

<div class="tooltip" markdown="1">
행렬과 벡터를 다루는 수학적인 공식을 구현하는 것은 NumPy를 고려해야 하는 주요 사용 사례입니다. 이 것은 왜 NumPy가 과학적 파이썬 커뮤니티에서 사랑받는 이유 입니다. 예를 들어, 회귀 문제를 다루는 지도학습 머신러닝 모델의 중심인 평균 제곱 오차 수식을 고려해보세요:
<span class="tooltiptext">
Implementing mathematical formulas that work on matrices and vectors is a key use case to consider NumPy for. It's why NumPy is the darling of the scientific python community. For example, consider the mean square error formula that is central to supervised machine learning models tackling regression problems:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/mean-square-error-formula.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
NumPy에서 이 것을 구현하는 것은 간단합니다:
<span class="tooltiptext">
Implementing this is a breeze in NumPy:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-mean-square-error-formula.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
이 것의 정수는 NumPy가 `predictions`와 `labels`이 한개던 천개던 몇개의 값을 가지고 있던지 간에 신경쓰지 않는다는 것입니다 (그 것들이 서로 같은 크기이기만 하다면). 다음 코드에서 4개의 연산을 순차적으로 단계별로 실행하는 예제를 살펴볼 수 있습니다:
<span class="tooltiptext">
The beauty of this is that numpy does not care if `predictions` and `labels` contain one or a thousand values (as long as they're both the same size). We can walk through an example stepping sequentially through the four operations in that line of code:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-mse-1.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
predictions 및 labels 벡터는 모두 3개의 값을 가지고 있습니다. 이 것은 n이 3임을 의미합니다. 뺼셈을 수행한 뒤, 다음과 같은 값이 나옵니다:
<span class="tooltiptext">
Both the predictions and labels vectors contain three values. Which means n has a value of three. After we carry out the subtraction, we end up with the values looking like this:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-mse-2.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
그런 다음 벡터에 있는 값들을 제곱합니다:
<span class="tooltiptext">
Then we can square the values in the vector:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-mse-3.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
이제 이 값들을 더합니다:
<span class="tooltiptext">
Now we sum these values:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-mse-4.png"/>
  <br />
</div>

<br />

<div class="tooltip" markdown="1">
해당 예측에 대한 에러 값이자 모델 품질에 대한 점수가 산출됩니다. 
<span class="tooltiptext">
Which results in the error value for that prediction and a score for the quality of the model.
</span>
</div>


### Data Representation (데이터 표현)

<div class="tooltip" markdown="1">
모델을 처리하거나 빌드하는데 필요한 모든 데이터 유형에 대해 생각해보십시오 (스프레드시트, 이미지, 오디오 등). 대부분의 경우 n-차원 배열의 표현과 완벽하게 적합합니다. 
<span class="tooltiptext">
Think of all the data types you'll need to crunch and build models around (spreadsheets, images, audio...etc). So many of them are perfectly suited for representation in an n-dimensional array:
</span>
</div>


#### Tables and Spreadsheets

 <div class="tooltip" markdown="1">
 * 스프레드시트나 테이블은 2차원 행렬입니다. 스프레드시트의 각 시트가 행렬로 사용될 수 있습니다. 파이썬에서 이 것을 위한 가장 인기있는 추상화는 [판다스 데이터프레임](https://jalammar.github.io/gentle-visual-intro-to-data-analysis-python-pandas/)이며, 이 것은 NumPy를 사용했고 그 위에 빌드한 것입니다. 
 <span class="tooltiptext" style="display: inline-block; text-align: left;">
 <span>*</span> A spreadsheet or a table of values is a two dimensional matrix. Each sheet in a spreadsheet can be its own variable. The most popular abstraction in python for those is the [pandas dataframe](https://jalammar.github.io/gentle-visual-intro-to-data-analysis-python-pandas/), which actually uses NumPy and builds on top of it.
 </span>
 </div>

 <div class="img-div-any-width" markdown="0">
   <image src="/images/pandas-intro/0%20excel-to-pandas.png"/>
   <br />
 </div>


#### Audio and Timeseries(시계열)

 <div class="tooltip" markdown="1">
 * 오디오 파일은 오디오 샘플들의 1차원 배열입니다. 각 샘플들은 오디오 신호의 작은 조각을 숫자로 표현한 것 입니다. CD품질 오디오는 초당 44,100 샘플을 가지며 각 샘플은 -32767 ~ 32767 사이의 정수 값을 갖습니다. CD품질 10초 WAVE 파일을 가지고 있다면, 10 * 44,100 = 441,000 샘플 길이의 NumPy 배열로 로딩할 수 있습니다. 오디오의 첫 1초를 추출하고 싶으신가요? `audio`라고 명명할 파일을 NumPy 배열로 단순히 로드하고 `audio[:44100]`를 가져오면 됩니다. 
 <span class="tooltiptext" style="display: inline-block; text-align: left;">
 <span>*</span> An audio file is a one-dimensional array of samples. Each sample is a number representing a tiny chunk of the audio signal. CD-quality audio may have 44,100 samples per second and each sample is an integer between -32767 and 32768. Meaning if you have a ten-seconds WAVE file of CD-quality, you can load it in a NumPy array with length 10 * 44,100 = 441,000 samples. Want to extract the first second of audio? simply load the file into a NumPy array that we'll call `audio`, and get `audio[:44100]`.
 </span>
 </div>

 <div class="tooltip" markdown="1">
 다음은 오디오 파일의 일부분입니다:
 <span class="tooltiptext" style="display: inline-block; text-align: left;">
 Here's a look at a slice of an audio file:
 </span>
 </div>

  <div class="img-div-any-width" markdown="0">
    <image src="/images/numpy/numpy-audio.png"/>
    <br />
  </div>

<div class="tooltip" markdown="1">
시계열 데이터도 동일합니다 (예를 들어, 시간에 걸친 주식 가격)
<span class="tooltiptext">
The same goes for time-series data (for example, the price of a stock over time).
</span>
</div>


#### Images


 <div class="tooltip" markdown="1">
 * 이미지는 (높이 x 너비) 크기의 픽셀 행렬입니다.

   * 만약 이미지가 흑백(회색조라고도 함)이라면, 각 픽셀은 1개의 숫자로 표현됩니다(주로 0(검정), 255(흰색) 사이의 값). 이미지의 왼쪽 상단의 10 x 10 픽셀을 자르고 싶은가요? NumPy에게 `image[:10,:10]`라고 하시면 됩니다. 

 <span class="tooltiptext" style="display: inline-block; text-align: left;">
 <span>*</span>  An image is a matrix of pixels of size (height x width).

   <span>*</span>  If the image is black and white (a.k.a. grayscale), each pixel can be represented by a single number (commonly between 0 (black) and 255 (white)). Want to crop the top left 10 x 10 pixel part of the image? Just tell NumPy to get you `image[:10,:10]`.
 </span>
 </div>

<div class="tooltip" markdown="1">
다음은 이미지 파일의 일부입니다:
<span class="tooltiptext">
Here's a look at a slice of an image file:
</span>
</div>


 <div class="img-div-any-width" markdown="0">
   <image src="/images/numpy/numpy-grayscale-image.png"/>
   <br />
 </div>


 <div class="tooltip" markdown="1">
 * 만약 이미지가 컬러이미지라면, 각 픽셀은 3개의 숫자로 표현됩니다 - 빨강, 초록, 파랑 각각의 값. 이 경우에 우리는 3차원이 필요합니다 (각 셀이 한 숫자만 표현할 수 있기 때문에). 그래서 컬러이미지는 (높이 x 너비 x 3) 차원의 ndarray로 표현됩니다. 
 <span class="tooltiptext" style="display: inline-block; text-align: left;">
 <span>*</span> If the image is colored, then each pixel is represented by three numbers - a value for each of red, green, and blue. In that case we need a 3rd dimension (because each cell can only contain one number). So a colored image is represented by an ndarray of dimensions: (height x width x 3).
 </span>
 </div>
   


    <div class="img-div-any-width" markdown="0">
      <image src="/images/numpy/numpy-color-image.png"/>
      <br />
    </div>


#### Language

<div class="tooltip" markdown="1">
만약 텍스트를 다룬다면, 얘기가 조금 달라집니다. 텍스트의 숫자 표현은 어휘(vocab; 모델이 아는 모든 고유한 단어 목록)이 만들어져야 하고 [임베딩 단계](https://jalammar.github.io/illustrated-word2vec/)가 있어야 합니다. 고대에 쓰여진 (번역된) 아래 인용구를 수치적으로 나타내는 단계를 살펴보겠습니다.
<span class="tooltiptext">
If we're dealing with text, the story is a little different. The numeric representation of text requires a step of building a vocabulary (an inventory of all the unique words the model knows) and an [embedding step](https://jalammar.github.io/illustrated-word2vec/). Let us see the steps of numerically representing this (translated) quote by an ancient spirit:
</span>
</div>

"Have the bards who preceded me left any theme unsung?"
("내 이전 음유시인들이 노래를 부르지 않고 남겨둔 주제가 있었던가?")

<div class="tooltip" markdown="1">
모델은 이 전사 시인의 불안의 말(단어들)을 숫자로 나타내기 전에 대량의 텍스트를 볼 필요가 있습니다. [작은 데이터셋](http://mattmahoney.net/dc/textdata.html)을 처리하여, (71,290 단어의) 어휘를 구축하는데 사용할 수 있습니다:
<span class="tooltiptext">
A model needs to look at a large amount of text before it can numerically represent the anxious words of this warrior poet. We can proceed to have it process a [small dataset](http://mattmahoney.net/dc/textdata.html) and use it to build a vocabulary (of 71,290 words):
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-nlp-vocabulary.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
문장은 토큰(일반적으로 단어나 단어의 일부분)의 배열로 나뉩니다.
<span class="tooltiptext">
The sentence can then be broken into an array of tokens (words or parts of words based on common rules):
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-nlp-tokenization.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
각 단어를 어휘 테이블의 id로 치환할 수 있습니다.
<span class="tooltiptext">
We then replace each word by its id in the vocabulary table:
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-nlp-ids.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이 id들은 여전히 모델에게 많은 정보 가치를 제공하지는 않습니다. 그래서 모델에게 단어의 시퀀스를 공급하기 전에, 토큰/워드는 임베딩으로 대체될 필요가 있습니다 (이 예에서는 50 차원 [word2vec 임베딩](https://jalammar.github.io/illustrated-word2vec/))
<span class="tooltiptext">
These ids still don't provide much information value to a model. So before feeding a sequence of words to a model, the tokens/words need to be replaced with their embeddings (50 dimension [word2vec embedding](https://jalammar.github.io/illustrated-word2vec/) in this case):
</span>
</div>

<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-nlp-embeddings.png"/>
  <br />
</div>



<div class="tooltip" markdown="1">
이 NumPy 배열이 [임베딩_차원 x 시퀀스_길이] 크기의 차원임을 알 수 있습니다. 실제로는 반대일 것이지만, 시각적 일관성을 위해 이렇게 표현하겠습니다. 성능 이유로, 딥러닝 모델은 첫번째 차원을 배치 사이즈로 남겨두는 경향이 있습니다 (왜냐하면 모델은 여러 예제를 병렬로 훈련하는 경우 빠르기 때문입니다). 이 것은 ```reshape()```이 매우 유용해지는 명백한 케이스입니다. 예를들어, [BERT](https://jalammar.github.io/illustrated-bert/)와 같은 모델은 이 것의 입력을 [배치_사이즈, 시퀀스_길이, 임베딩_크기] shape의 입력을 예상합니다.
<span class="tooltiptext">
You can see that this NumPy array has the dimensions [embedding_dimension x sequence_length]. In practice these would be the other way around, but I'm presenting it this way for visual consistency. For performance reasons, deep learning models tend to preserve the first dimension for batch size (because the model can be trained faster if multiple examples are trained in parallel). This is a clear case where ```reshape()``` becomes super useful. A model like [BERT](https://jalammar.github.io/illustrated-bert/), for example, would expect its inputs in the shape: [batch_size, sequence_length, embedding_size].
</span>
</div>


<div class="img-div-any-width" markdown="0">
  <image src="/images/numpy/numpy-nlp-bert-shape.png"/>
  <br />
</div>

<div class="tooltip" markdown="1">
이제 모델이 유용한 작업을 수행할 수 있는 숫자 볼륨(단위)입니다. 다른 행은 비워 두었지만 모델이 학습(또는 예측)할 다른 예제(숫자값)들로 채워질 것입니다. 
<span class="tooltiptext">
This is now a numeric volume that a model can crunch and do useful things with. I left the other rows empty, but they'd be filled with other examples for the model to train on (or predict).
</span>
</div>

<div class="tooltip" markdown="1">
위의 예에서 [이 시인(Antara)의 시](https://en.wikisource.org/wiki/The_Poem_of_Antara)은 어떤 다른 시들 보다 훨씬 더 불후의 명성을 얻었습니다. 아버지 소유의 노예로 태어난 Antarah(어머니가 포로/노예로 잡혀간 에티오피아 공주)는 용맹과 언어 구사력을 통해 자유를 얻었고, 그의 시가 이슬람 이전 아라비아의 [카바(사우디아라비아 메카 소재 '하람 성원'의 중심에 위치)에 게시된 7개의 시](https://en.wikipedia.org/wiki/Mu%27allaqat) 중 하나로 신화적 지위를 얻었습니다.
<span class="tooltiptext">
(It turned out the [poet's words](https://en.wikisource.org/wiki/The_Poem_of_Antara) in our example were immortalized more so than those of the other poets which trigger his anxieties. Born a slave owned by his father, [Antarah's](https://en.wikipedia.org/wiki/Antarah_ibn_Shaddad) valor and command of language gained him his freedom and the mythical status of having his poem as one of [seven poems suspended in the kaaba](https://en.wikipedia.org/wiki/Mu%27allaqat) in pre-Islamic Arabia).
</span>
</div>

---


## 추가 정보<a href="#additional-info" name="additional-info">.</a>

* 이 글은 GPT2에 대해 이해하기 쉽게 그림으로 설명한 포스팅을 저자인 Jay Alammar님의 허락을 받고 번역한 글 입니다. 원문은 [A Visual Intro to NumPy and Data Representation](https://jalammar.github.io/visual-numpy/)에서 확인하실 수 있습니다.
* 원서/영문블로그를 보실 때 term에 대한 정보 호환을 위해, 이 분야에서 사용하고 있는 단어, 문구에 대해 가급적 번역하지 않고 원문 그대로 두었습니다. 그리고, 직역 보다는 개념이나 의미에 대한 설명을 쉽게 하는 문장 쪽으로 더 무게를 두어 번역 했습니다.
* 번역문에 대응하는 영어 원문을 보고싶으신 분들을 위해 [찬](https://nlpinkorean.github.io)님께서 만들어두신 툴팁 도움말 기능(해당 문단에 마우스를 올리면 (모바일의 경우 터치) 원문을 확인할 수 있는 기능)을 가져와서 적용했습니다. 감사합니다.  

