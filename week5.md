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

