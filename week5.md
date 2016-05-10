# Machine Learing Note (week #5)
* Neural Networks: Representation
* progress: 5주차
* 
* date: 2016.05.0ᆨ3

* Note:
 - 1주와 2주는 supervised learing 에서 Linear Regression(선형 회귀)을 공부했습니다.
 - 3주는 classification(군집화)에 대하여 공부했습니다. 
 - 4주는 기계학습에서 사용하는 신경망(neural network)을 공부했습니다.
 - 5주는 신경망이 어떻게 학습을 하는지에 대하여 공부합니다. 


## Week5 contents
* ᆱCost Funciton and Backpropagation
 - Cost Function 비용함수
 - Backpropagation Algorithm 역전파 알고리즘
 - Backpropagation Intuition 역전파 직관

* Backpropagation in Practice 역전파 예시
 - Implementation Note: Unrolling Parameters 구현노트: 파라메터 풀어쓰기
 - Gradient Checking: 경사도 검증
 - Random Initialization 임의의 초기화
 - Putting It Together 함께 넣기.

* Application of Neural Networks 신경망 응용
 - Autonomous Driving 자율주행

## ᆱCost Function and Backpropagation 비용함수와 역전파.
 - Cost Function 비용함수
 - Backpropagation Algorithm 역전파 알고리즘
 - Backpropagation Intuition 역전파 직관

### Cost Function 비용함수
![multi-0ᆸ3](https://github.com/hephaex/ML_class/blob/master/week4/week4_6_MulticlassClassification_03.png)

Neuroal Networks cost function 신경망 비용 함수

**신경망 (NNs, Neural Networksᆫ), 그리고 신경망 비용 함수**
- 가장 강력한 학습 알고리즘 중 하나입니다.
- 신경망에 사용된 매개 변수(theta)를 찾기 위해서 학습용 훈련용 셋트를 사용합니다.
- 신경 네트워크의 매개 변수(theta)를 최적화 하는 비용 함수를 알아보겠습니다.

**신경망(NN)의 여러가지 예시 중 분류(ᆱClassification)를 가지고 설명하겠습니다.**
* Training set is { (x1, y1), (x2, y2), (x3, y3) ..., (xn, ym) }
  - x^N: 입력에 사용한 feature 수
  - y^m: 분류하고자 하는 대상(object 수, 출력)
  - L: 네트워크 층 (layer 수)
  - S1: layer 1에 unit 수

![costfunction-01](https://github.com/hephaex/ML_class/blob/master/week5/week5_01_CostFunction_01.png)
>x^N: 3
>y^m: 4
>L: 4
>s1: 3
>s2: 5
>s3: 5
>s4: 4

4주에서 배웠던 NN 분류의 종류
![costfunction-07](https://github.com/hephaex/ML_class/blob/master/week5/week5_01_CostFunction_0ᆻ7.png) 
* 이진 분류 (binary classification
 - 분류 대상(출력, y^m)이 한가지만 나옴.
 - 출력 값은 거짓(0)이나 참(1)로 두가지 형태
 - 따라서 ᆫsL: 1 이 됨. 

* 다수 분류 (multiclasss classifcation)
 - 명확한 분류 대상(출력, y^m)은 k개
 - k가 2라면 이진 분류를 사용
 - 일반적으로 K는 3 이상임
 - 마지막 층(L)의 값은 k이다. (L = K)
 - 분류 대상 y는 실수 값으로 이루어진 k 차원 벡터가 된다. (k-dimensional vector)
![costfunction-02](https://github.com/hephaex/ML_class/blob/master/week5/week5_01_CostFunction_0ᆻ2.png) 

**신경 네트워크를 위한 비용 함수**

신경망의 비용함수 (neural network cost funciton) J(Ɵ)는 정규화된 선형 회귀 함수와 비슷하다. (regularized logistric regression)
![costfunction-03](https://github.com/hephaex/ML_class/blob/master/week5/week5_01_CostFunction_0ᆻ3.png) 

하나의 출력이 아닌 다수 k개의 출력이므로, 신경망 우리 비용 함수는 다음 식이 된다.
![costfunction-04](https://github.com/hephaex/ML_class/blob/master/week5/week5_01_CostFunction_0ᆻ4.png) 
예제에서 4개를 분류한다면 k는 4가 되고, K가 1~4까지 결과 값에 대한 비용함수가 된다. 

* 따라서 신경망에서 비용 함수는 K 차원의 벡터를 출력하게 된다. 
 - hƟ(ᆨᆨᆨx) : k dimensional vector
 - hƟ(x)i : 벡터의 i번째 값을 나타낸다.

비용함수를 두가지로 나누어서 보자.

첫번째 항은 다음처럼 쓸 수 있다. 
![costfunction-05](https://github.com/hephaex/ML_class/blob/master/week5/week5_01_CostFunction_0ᆻ5.png) 
처음 1부터 m까지 트레이닝에 사용한 데이터이며,
각각의 출력을 모두 더한 값이 될것이다.
이것은 logistic regression (로지스텍 회기법)과 같이 된다..

두번째 항도 다음처럼 쓸 수 있다.
![costfunction-06](https://github.com/hephaex/ML_class/blob/master/week5/week5_01_CostFunction_0ᆻ6.png) 
이 값은 매우 복잡한 과정을 거쳐서 구하게 된다.
Ɵ에 대한 i, j, l의 모든 값에 θ에 대하여 구한 값이 될 것이다. 
여기서 bias 값(θ0)는 계산에 포함되지 않는 것에 주의 하자. 

대규모로 정규화된 값들을 모두 더한 값이 될 것이다. 
이것을 중량감쇠(weight decay)라고 합니다. 
로지스틱 회기 분석와 유사하게 람다(λ) 값을 사용합니다. 


## 역 전파 알고리즘 Backpropagation Algorithm

신경망에서 비용함수를 쓰면
ᆽ![backpropa-01](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_01.png)
이고 여기서 비용함수 J(θ)를 최소화 하는 것이 목적이 될 것이다.
비용함수 J(θ)를 최소화 값을 구하려면,
J(θ)가 i, j, l의 요소로 구성된  함수이므로,
θ에 대한 편미분(증감) 값이 최소화하는 값을 구하면 된다.
ᆽ![backpropa-02](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_02.png)

그렇다면 예제를 가지고 이것을 구체적으로 알아보자.
입력x와 목적한 출력 y에 대하여 트레이닝 예시를 보면 다음처럼 될 것이다.
ᆽ![backpropa-03](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_03.png)

지난 주에서 했던 전진 전파(forward propagation)으로 이 신경망을 정리하면.

* layer 1
  - ᆼa^1 : x
  - z^2 : θ1 * a1
* layer 2
  - ᆼa^2 : g(z^2) (add ao ^ 2)
  - z^3 : θ2 * a2
* layer 3
  - ᆼa^3 : g(z^3) (add ao ^ 3)
  - z^4 : θ3 * a3
* output
  - a^4 : hθ(x) = g( z^4 )

ᆽ![backpropa-04](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_04.png)

**역전파 backpropagation algorithm**
layer l에서 노드 j의 원하는 값과의 차이를 delta(δ)라고 하자.
delta(δ)는 j와 l로 첨자를 쓸 수 있다.
ᆽ![backpropa-12](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_12.png)

>ᆼᆼaj ^ l: the activation node of node j in layer l
>δj ^ l: the error of node j in layer l

원하는 결과와 실제 나타난 결과의 차이가 우리가 구성한 신경망에서 오류값이 될 것이다.
이 오류값을 최소화하는 것이 우리가 하고자 하는 목적이며, 이것이 비용함수를 최소화 하는 값이 된다.

따라서 원하는 출력 값과 실제 출력 값이 어떻게 구하는지 수식으로 풀어 써 보자.

출력 레이어 4에 대한 오류 값 delta는 δj4는
>δj4 = aj4 - yj
>> ᆼaj4: acivation j4(신경망에서 실제 값)
>> yj: 원하는 결과 값
>> δj4: 실제 값과 결과 값의 차이 

이것을 벡터로 표현하면.
>δ^4 = a^4 - y 이다. 
>> δ^4 : 4번째 레이어에서 오류값
>> ᆼa^4:  4번째 레이어에서 신경망의 실제 값

δ^4 를 구했으므로 다른 레이어에 대하여도 구해보자.
ᆽ![backpropa-05](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_05.png)

> Ɵ^3 : 레이어 ᆸ3에서 4로 매핑될 때 매개변수 벡터이다.
> δ^4 : 레이어 4에서 계산한 값이다.
> g'(z3) : 입력으로 주어진 ᆷz3에 대하여 activation function으로 계산된 값이다.
> 즉 g'(z3) = a3 . * (1 - a3) 가 되며,
> 이것은 다시 레이어 3에서 오류값 delta로 정리하면.
> δ3 = (Ɵ3)T δ4 . *(a3 . * (1 - a3)) 가 된다.
>
> . * 는 matlab에서 두 벡터를 곱하는 연산자 이다.

그러면 전체 신경망을 수학으로 다시 분석해 보자.

ᆽ![backpropa-03](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_03.png)

* ᆼactivation vector ᆼᆼa^1, a^2, a^3, a^4는
 - a^1 : 3
 - a^2 : 5
 - a^3 : 5
 - a^4 : 4

* Ɵ^3: 레이어 3에서 레이어 4로 매핑되는 매개변수(parameter) 벡터
 - [ 4 x 5 ] 행렬 (matrix)이 된다. (bias를 포함 한다면 [ 4 x 6 ] 이다.)
 - (Ɵ3)T 의 행렬(matrix)는 [ 5 x 4 ] 가 된다.

* δ^4: 레이어 4에서 오류 값
 - [ 4 x 1 ] 벡터가 된다.

* (Ɵ^3)T δ^4 를 구하면
 - [ 5 x 4 ] * [ 4 x 1 ] 이므로 [5 x 1 ] 벡터가 된다.
 - 즉 a3 의 벡터 [ 5 x 1 ] 과 동일한 벡터 형태가 됨을 알 수 있다.

이 방법을 반복해서 레이어 3에서 오류값과 레이어 2에서 오류 값을 수식으로 정리할 수 있다.

> δ2 = (Ɵ2)T δ3 . *(a2 . * (1 - a2))

δ1은 입력이므로 구하지 않는다.

δ를 가지고 신경망을 바꾸서 쓸 수 있었다.

우리가 여기서 구하고자 하는 것은 신경망의 오류를 최소화 하는 비용함수였다.
즉 δ만으로 표현된 비용함수를 구할 수 있다.
비용함수를 최소화하면 가장 잘 학습된 신경망에서 결과값을 구할 수 있게 될 것이다.
즉 δ만으로 표현된 함수를 편미분을 취하여 비용함수가 최소가 되게 할 수 있다.
ᆽ![backpropa-13](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_13.png)

정리해 보자.
예시에서 trainin set은 
>ᆽ![backpropa-07](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_07.png)
이다.

delta 값은 초기화해서 0으로 놓자.
ᆽ![backpropa-08](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_08.png)

training set에 대하여 반복된 루프를 해서 delta를 계산해보자.

* ᆼa^1 : x ^ 1
 - 레이어 1은 입력이므로 a ^ 1은 x ^ 1 입력 값이 된다.

* 레이어 2부터 레이어 L까지 forward propagation을 통해서 a^l 을 계산할 수 있다.

* 마지막 레이어에서 원하는 결과 값과 실제 계산된 결과 값의 차이 δL 을 구할 수 있다.

* δL = a ^ l - y ^ i 이므로

* back propagation 을 통해서 δL -> δ(L-1) -> δ(L-2) -> ... -> δ2 를 계산할 수 있다.

* delta로 정리하면
> ᆽ![backpropa-09](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_09.png)

* 이것을 벡터로 표현하면. 
> ᆽ![backpropa-10](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_10.png)

* 즉 비용함수를 계산 하기 위한 루프에서 다음을 반복하면 된다.
> ᆽ![backpropa-11](https://github.com/hephaex/ML_class/blob/master/week5/week5_02_BackPropagation_11.png)

여기서 정규화 항을 고려하지 않는 다면 j가 0이 되기 때문에,
신경망에서 비용함수를 구할 수 있게 되었다.

## back propagation intuition

그림과 함께 forward propagation과 back propagation을 예를 들어 살펴보자.

![bpi-01](https://github.com/hephaex/ML_class/blob/master/week5/week5_03_BackPropagationIntuition_01.png)

여기에 input layer (xi, yi)를 입력해 보자.
그리고 각 레이어를 따라가면서 forward propagation을 구해 보자. 
![bpi-02](https://github.com/hephaex/ML_class/blob/master/week5/week5_03_BackPropagationIntuition_02.png)

sigmoid 함수를 적용하면 각 레이어에서 z값을 구할 수 있다.

![bpi-0ᆸ3](https://github.com/hephaex/ML_class/blob/master/week5/week5_03_BackPropagationIntuition_03.png)

forward propagation이 끝났다면 back propagation을 할 차례이다.
back propagation은 forward propagation과 비슷하게 나아 갈 것이지만 방향이 반대이다.

여기서 출력은 이진 분류이므로 마지막 레이어는 a1^4만 있게 될 것이다.

신경망에서 비용함수는
>![bpi-04](https://github.com/hephaex/ML_class/blob/master/week5/week5_03_BackPropagationIntuition_04.png)
이다. 

이것을 비용함수를 i에 대해서 다시 써 보면,
>![bpi-05](https://github.com/hephaex/ML_class/blob/master/week5/week5_03_BackPropagationIntuition_05.png)

δ는 오류값이므로 이것을 비용함수로 정리하면
>![bpi-06](https://github.com/hephaex/ML_class/blob/master/week5/week5_03_BackPropagationIntuition_06.png) 
로 쓸 수 있다.

그럼 back propagation을 해보자.

![bpi-07](https://github.com/hephaex/ML_class/blob/master/week5/week5_03_BackPropagationIntuition_07.png)

δᇂᇂ1^4 에서
> δᇂ1 ^ 4 = y ^ i - a1 ^ 4
이므로
> δ2 ^ 3 = (Ɵ12 ^ 3) * ( δ1 ^ 3) + (Ɵ22 ^ 3) * (δ2 ^3)  
이 된다.

![bpi-08](https://github.com/hephaex/ML_class/blob/master/week5/week5_03_BackPropagationIntuition_08.png)

