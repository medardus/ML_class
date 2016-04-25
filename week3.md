# Machine Learing Note (week#3)
* progress: week #3
* 
* date: 2016.04.26

## week3 content
* Classification and Representation
 - Classification
 - Hypothesis Representation
 - Decision Boundary

* Logistic Regression Model
 - Cost Function
 - Simplified Cost Function and Gradient Descent
 - Advanced Optimization

* Muliclass Classification
 - Multiclass Classification: One-vs-all

* Solving the Problem of Overfitting
 - The Problem of Overfitting
 - Cost Function
 - Regularized Linear Regression
 - Regularized Logistic Regression

## Classification and Representation

### Classification
* Classification 예시
 - Email: Spam / Not Spam?
 - Online Trasactions: Fraudulent (Yes / No)?
 - Tumor: Malignant / Benign?

예시한 Classification들은 결과값 Y가 2가지를 가진다. 예시에서 암검진이라면
양성은 1을 음성은 0, 즉 Y를 수학으로 모델링하면 0 혹은 1을 같는다.
0은 Negative Class (ex. Benign Tumor) 이며 1은 Positive Class (ex. malignant tumor) 이다.
![classifcation01](https://github.com/hephaex/ML_class/blob/master/week3/week3_01_classification_01.png)

 - 0: Negative Class
   - 정상 값, 구분을 원하지 않은 값
   - ex) email: Not Spam
   - ex) tumor: benign tumor
   - ex) online transactions: No frudulent
 - 1: Positive  Class
   - 비정상 값, 우리가 구분하길 원하는 값
   - ex) email: Spam
   - ex) tumor: malignant tumor
   - ex) online transactions: frudulent

우선은 결과가 2가지 인것부터 공부하고 Y가 다양한 값을 가지는 경우도 공부하자.

암검진으로 예시를 들어 설명하면,
결과 htheta(x)가 0.5 라고 하면
 htheta(x) >= 0.5 라면 y = 1로 예측할 수 있다.
 htheta(x) <  0.5 라면 y = 0로 예측할 수 있다.

여기서 만약 tumor size가 아주 특이한 값(큰값을 가지다면)
Linear Regression에 의해서 htheta(x)는 0.5에서 더 큰 값으로 바뀔 것이다.
이렇게 되면 tumor size에 대한 일부는 양성이지만 음성으로 판정될 수 있다.
![classifcation02](https://github.com/hephaex/ML_class/blob/master/week3/week3_01_classification_02.png)

따라서 linear regression은 classifcation 문제에 잘 사용하지 않는다.

Classification: y = 0, y = 1의 이산값을 가지면 htheta(x)는 > 1 or < 0 이다.
logistic regression은 0 =< htheta(x) =< 1 이 되며,
역사적인 이유로 classification을 logistic regression 이라고 한다.
![classifcation03](https://github.com/hephaex/ML_class/blob/master/week3/week3_01_classification_03.png)


## Hypothesis Representation
* 예시로 살펴본 암 검진에서 종양을 악성과 양성으로 검증하기 위한 가설을 세웠다. 
 - 종양이 크기에 따라(ᆨx) 학습하려는 현상을 만족하는 가설을 위해서 여러가지 모델 중
 - 선형회기 모델(Linear regression model) 을 사용하기로 하였다.
 - 선형 회기 모델에 따른 가설 검증(예측 값)은 음성(0)과 양성(1)사이로 하였다. 
   - htheta(x) 는 0 =< htheta(x) =< 1 로
 - 이것을 htheta(x) = theta Transpose x로 나타낼 수 있다.
   - htheta(x) = thetaT * x
 - 가설 htheta(x)를 logisting regression 모델을 위해서 조금 수정하면.
   - g(htheta T * x) 로 변환하였다.
 - 여기서 G 변환함수는
   - g(z) : 1 / (1 + e^-z)
 - G 변환함수를 Sigmoid function 혹은 logistic function 이라고 한다.
 - G 변환함수를 다시 정리해서 쓰면
 - htheta(x)
   - = 1 / (1 + e^-z)
   - = 1 / (1 + e^-(theta T * x) )
 - G 함수에 대해서 그래프로 그려 보면
   - Z값이 음수 (z < 0) 일 때 0에 무한히 가까운 값에서 부터 증가하여 0.5까지 증가한다.
   - Z값이 양수 (z > 0) 일 때 0.5부터 증가하여 1에 무한히 가까운 값까지 증가한다. 
   - 왜냐하면 htheta(z) 는 0과 1 사이 값을 가지기 때문이다. 

* 가설의 결과 값을 해석해 봅시다.
* 가설 htheta(x)는 입력 X에 대해서 결과 Y가 1(양성)이때 확률이라고 하면.
 - ᆨx = [ x0, x1 ] = [ 1, tumorSize ] 가 된다.
 - 여기서 htheta(x)에 x(종양 크기, tumorSize) 를 0.7이라면 결과가 양성(1)인 확률이다.

* 이것을 주어진 x와 매개변수 theta가 있을 때 결과(y)가 1인 확률이라면
 - htheta(x) = P(y=1 | x, theta) 가 된다.

* 종양은 양성(1)과 음성(0)으로 구분되므로.
 - P(y = 0 | x, theta) + P(y = 1 | x, theta) = 1 이다.
 - 이것을 P(y = 0 | x, theta) 값을 기준으로 정리하면.
   - P(y = 0 | x, theta) = ᇂᇂ1 - P(y = 1 | x, theta)
   
## Decision Boundary

# Logistic Regression Model

## Cost Function

## Simplified Cost Function and Gradient Descent

## Advanced Optimization
 로지스틱 회기(Logistic regression)에서 theta에 대한 비용함수(cost)를 최소화하는
 방법인 경사하강법(Gradient Descent)에 대해서 이야기 했습니다.

 좀더 나아가서 최적화하는 알고리즘과 방법에 대해서 좀 더 이야기 하겠습니다.

이방법은 Gradient descent(경사하강법)을 사용한 것보다 좀더 빠르게 실행됩니다.
그리고 좀더 많은 학습대상에서도 유용합니다.

Optimization algorithm
Cost Function J(theta) 에서 theta에 대하여 cost function J(theta)가 최소화되는
값을 구하는 것이 목적입니다.

 최초값을 찾기 위해서는 theta에 대하여 편미분을 적용합니다.
 - J(theta)
 - Partial derivative theta (for j = 0, 1, ... , n)

이것을 gradient descent를 적용하면

로 정리할 수 있으며 cost function J(theta)와 theta에 대한 편미분 J(theta)로
나타낼 수 있습니다.

최적화 하는 알고리즘
- Gradient descent
  - 선형 회기 분석에서 사용했음.
- Conjugate Gradient
- BFGS
- L-BFGS

ᆱGradient Descent에 비하여 Conjugate gradient , BFGS, L-BFGS는 비교
- Learning Rate (ᆮAlpha)를 지정하지 않아도 된다.
- Gradient descent에 비교하여 대부분이 더 빠르다.
- 좀더 복잡하다는 단점이 있다.

구현이 복잡하지만 이것은 알고리즘을 개발하는 사람들이 라이브러리로
잘 만들었기 때문에 Matlab, octabe에서는 이 라이브러리를 잘 활용하면 된다.
다른 언어 Java, C/C++에서는 구현하는 방식이 차이가 있을 수 있기 때문에
이른 잘 선별해서 사용하는 것이 좋다.

예시

theta 는 theta1, theta2가 있다고 할때.
cost function J(theta) =  (theta1 - 5)^2 + (theta2 - 5)^2
 -> J를 theta1으로 편미분 하면 J(theta1) = 2 * (theta1 - 5)
 -> J를 theta2으로 편미분 하면 J(theta2) = 2 * (theta2 - 5)

octabe로 코드로 이것을 써보자.
function [jVal, gradient] = costFunction(theta)
  jVal = (theta(1) - 5)^2 + (theta(2) - 5)^2
  gradient = zeros(2,1);
  gradient(1) = 2 * (theta(1) - 5);
  gradient(2) = 2 * (theta(2) - 5);

좀더 나아가서 비용함수 J(theta)에 대한 최적화 함수 fminunc를 사용하면.

> options = optimset('GradObj', 'on', 'MaxIter', '100');
> initialTheta = zeros(2,1);
> [optTheta, functionVal, exitFlag] ...
>     = fminuc(@contFunction, initialTheta, options);

여기서 선택가능한 옵션을 지정할 수 있는데, 'gradObj', 'on'이란
gradient object기능을 활성화하는 의미이다.
'MaxIter' 최대 반복 수를 100회로 제한한다.

# Muliclass Classification

## Multiclass Classification: One-vs-all
multiclass(여러 종류)를 classification(구분)할때 어려움.


예싷1.) e-mail을 구분하거나 인식표를 붙일때 4가지로 다양하게 구분을 할 수 있다. 
 - work    | y = 1
 - friends | y = 2
 - family  | y = 3
 - hobby   | y = 4

예시2.) 의료검진에서 3가지로 구분할 수 있음.
 - 안 아품 | y = 1
 - 감기    | y = 2
 - 독감    | y = 3 

예시3.) 일기예보에서 날씨를 4가지로 구분할 수 있다.
 - 맑음    | y = 1
 - 구름 낌 | y = 2
 - 비      | y = 3
 - 눈      | y = 4

이것은 종양검진처럼 0, 1이 아닌 결과값이 다양한 값을 가진다.

# Solving the Problem of Overfitting

## The Problem of Overfitting
overfitting

- htheta(x) 가
 - too large
 - too variable
 - 제한된 자료만으로 만족할 만한 가설을 세우기 어렵다.
 
* Cost Function
Cost funtuon을 최적화하기 위한 regularization 방법

htheta(x)가 2차거나 다차 함수일 때.
3차, 4차에서 theta3, theta4가 아주 큰 값이라면(1000)
x3와 x4는 아주 작은 값을 가질 것이기 때문에,
x3 = 0 , x4 = 0 로 볼 수 있고
htheta(x)는 단순화할 수 있다.

Regularization
Small value for parameters
 - simpler hypothesis
 - less pron to overfitting

예시에서 보듯 theta3와 theta4를 0으로 본다면 가설함수는 단순해진다.

예시
- feature: x1, x2, ,,, , x100
- parametets: theta0, theta1, ,,, , theta100

여기서 어떤 것을 남기고 어떤 것을 지울지 알수 없기 때문에,
cost function J(theta)는 ᇂ 바꾸면,
 regulation 항을 추가한 형태로 바꿀 수 있다.

Regulazation
lamda: regulazation parameter,
 - 람다 앞에 항을 데이터를 잘 맞게 해주고,
 - 람다항은 작은 값으로 만든다.

예시
고차항으로 overfitting한 것을 regulazation을 통하여.
낮은 차수로 바꾸어 준다.

람다가 아주 큰 값이라면 theta1, theta2, ,,, thetaN이 0이되어야만 한다.
이때 theta0만 남게 되므로 가설함수는 일정한 값을 가지게 된다.
이것을 underfitting이라고 한다.

이것을 피하기 위해서 람다 값을 잘 선택해야만 한다. 


* Regularized Linear Regression
선형회기에서는 
* Regularized Logistic Regression

