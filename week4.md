# Machine Learing Note (week #4)
* Neural Networks: Representation
* progress: #4 week
* 
* date: 2016.05.0ᆨx

* Note:
 - 1주와 2주는 supervised learing 에서 Linear Regression(선형 회귀)을 공부했다.
 - 3주는 classification(군집화)에 대하여 공부했다.
 - 4주는 기계학습에서 사용하는 신경망(neural network)을 공부한다.

## Week4 contents
* Motivations
 - Non-linear Hypotheses
 - Neurons and the Brain

* Neural Networks
 - Model Representaion I
 - Model Representaion II

* Application
 - Examples and Intuition I
 - Examples and Intuition II
 - Multiclass classfication

## Motivations
- Non-linear Hypotheses
- Neurons and the Brain

### Non-linear Hypotheses
Nural network :
 - 뉴럴 네트워크는 고안된지 오래된 아이디어이다.
 - 기계학습을 더욱 심화할 수록 의미는 줄어 들수 있지만,
 - 많은 기계 학습 문제를 푸는데 사용되었습니다.

* 왜 다른 알고리즘을 배워야 할까?
 - 기계학습을 위해서 linear regression(선형 회기)와 logistic regression을 배웠다.
 - 복잡한 비선형 문제(complex non-linear hypothesis)를 생각해 보자.
 - [non-linear hypothesis 01](https://github.com/hephaex/ML_class/blob/master/week4/week4_1_Non-Linear_Hypotesis_01.png)
 - 지도 학습의 종류 구분 문제(supervised classification problem)으로 이 것을 푼다면.
 - logistic regression을 적용할 것이고,
 - 많은 비 선형 요소(non-linear feature)에 대하여 생각해야만 한다.
 - 또한 signoid 함수를 사용할 때 고차상에 대하여도 생각해야만 한다.
 - 이 문제 뿐만 아니라 앞에서 예를 든 집값 예측에서도
 - 이집이 가격을 예측했지만 6개월 안에 팔릴지 안팔릴지 하는 것은
 - 분류 문제(Classification problem)로 생각할 수 있게 된다.
 - 가설 함수를 세울 때에도 2차가 넘어가면 더욱더 문제는 복잡해진다.

정리해보자.

* Complex supervised learning classification problem
 - 지도학습 중 linear regression을 배웠고,
 - logistic regression또한 linear regresion 문제로 바꿔서 해결하는 방법을 배웠다.
 - linear regression은 ᇂ1-2개의 요소(feature) 문제에 대해서는 잘 적용된다.
 - 요소가 100개가 넘어가면 적용하기 어렵다.
 - [non-linear hypothesis 02](https://github.com/hephaex/ML_class/blob/master/week4/week4_1_Non-Linear_Hypotesis_02.png)
 - 고차항(many polynomial terms)에 logistic regression은 적용하기 어렵다. 

### 예시) 학습할 요소(feature) N이 큰 경우
 기계 학습을 이용한 사물인식 (computer vision)을 예로 들어보자.

사용자가 사진을 제시하고 자동차인지 아닌지 구분하는 기계 학습을 구현한다고 하자.
[non-linear hypothesis 03](https://github.com/hephaex/ML_class/blob/master/week4/week4_1_Non-Linear_Hypotesis_03.png)

자동차 사진을 학습을 시켰고, 자동차가 아닌 것도 함께 학습을 시켰다.
방법은 우리가 앞서 배운 logistic regression이 될 것이다.

여기서 새로운 사진을 입력을 하여 이것이 자동차인지 아닌지 구분한다고 하자.
어떻게 이 문제를 풀 수 있을까?

기계학습에서 사물을 인식할 때는 행렬로 표현된 화소(matrix pixel)를 사용한다.
[non-linear hypothesis 04](https://github.com/hephaex/ML_class/blob/master/week4/week4_1_Non-Linear_Hypotesis_04.png)


* 자동차 인식 (a car recongition)
 - 학습(training set)
   - 자동차 Cars 
   - 자동차가 아니다.Not cars 
 - 두개의 화소 요소를 추출해서 이것으로 자동차 인지 아닌지 판단한다고 할때.
   - 자동차 Cars를 그래프에 + 로 표시
   - 자동차가 아니다.Not cars 를 그래프에 -로 표시
 - [non-linear hypothesis 05](https://github.com/hephaex/ML_class/blob/master/week4/week4_1_Non-Linear_Hypotesis_05.png)   
 - 학습 요소에서 화소를 50 x 50 pixel을 사용한다고 하면.
   - 50 x 50 => 2500 pixels
   - 따라서 n = 2500 이 된다.
   - 삼색(RGB)를 사용하면 2500 x 3 (R, G, B 3요소) => 7500
   - 자동차 100개, 자동차 아니다 100개를 학습한다면.
   - 7500 x 100 x 100 => 50,000,000 이상의 요소를 학습에 사용.
   - 이것은 기존 linear regression으로 풀기는 어렵다.

linear regression으로 풀기 어려운 문제를 해결하기 위한 아이디어 중
신경망을 이용한 방법이 있다.
그럼 신경망에 대하여 알아보자. 

# Neurons
