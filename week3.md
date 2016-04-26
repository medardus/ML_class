# Machine Learing Note (week #3)
* Logistic Regression
* progress: #3 week
* date: 2016.04.26

* Note:
 - 1주와 2주는 supervised learing 에서 Linear Regression(선형 회귀)을 공부했다.
 - 3주는 classification(군집화)에 대하여 공부한다. 

## week #3 content
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
**ᆱClassification(군집화)**
- 입력 x에 대하여 이것이 어떤 곳(Y)가 될지 구분하는 방법이다.

* Classification 예시
 - E-mail: Spam / Not Spam?
   - 어떤 이메일이 스팸으로 구분할지? 아닌지(정상)?
 - Online Trasactions: Fraudulent (Yes / No)?
   - 온라인 쇼핑에서 사기가 의심되는지? (사기/정상)
 - Tumor: Malignant / Benign?
   - Tumor(종양)이 Malignant(악성)인지? Benign(음성)인지? 

위의 3가지 예시는 Classification들은 결과값 Y가 2가지를 가진다.
 - 이것을 binary class problems (2항 계층 문제)라고 한다. 
 - 우선은 결과 Y가 2가지 인것부터 공부하고
 - 이후 결과 Y가 다양한 값을 가지는 경우도 공부하자.

* Binary class problem에서 예측값(출력,Y)은 0 혹은 1이다.
 - 0: negative class (absence of something)
   - 정상 값, 구분을 원하지 않은 값
   - ex.) 이메일(email): 스팸이 아니다.(Not Spam)
   - ex.) 암검진에서 음성(Benign tumor)
   - ex.) 온라인 결재에서 (online transactions): 정상거래이다.(No frudulent)
 - 1: Positive class (presence of something)
   - 비정상 값, 우리가 구분하길 원하는 값
   - ex.) 이메일(email): 스팸이다.(Spam)
   - ex.) 암검진에서 음성(Malignant tumor)
   - ex.) 온라인 결재에서 (online transactions): 사기거래이다.(frudulent)

![classifcation01](https://github.com/hephaex/ML_class/blob/master/week3/week3_01_classification_01.png)

예시에서 암검진으로 가설 htheta(x)가 0.5 라고 하면
 - htheta(x) >= 0.5 라면 y = 1로 예측할 수 있다.
 - htheta(x) <  0.5 라면 y = 0로 예측할 수 있다.

여기서 만약 tumor size가 아주 특이한 값(큰값을 가지다면)
 - Linear Regression에 의해서 htheta(x)는 0.5에서 더 큰 값으로 바뀔 것이다.
 - 이렇게 되면 tumor size에 대한 일부는 양성이지만 음성으로 판정될 수 있다.
 - 따라서 linear regression은 classifcation 문제에 잘 사용하지 않는다.

![classifcation02](https://github.com/hephaex/ML_class/blob/master/week3/week3_01_classification_02.png)

**이항 계층 문제(binary class probelm)의 특징**
 - Classification:
   - y = 0 이나, y = 1의 이산값을 가질 때 htheta(x)는 > 1 혹은 < 0 이다.
   - Hypothesis can give values large than 1 or less than 0
 - Logistic regression:
   - 생성된 값 (Y)는 0 =< y =< 1 이다.
   - 역사적인 이유로 classification을 logistic regression 이라고 한다.
   - **Logistic regression is a classification algorithm**

![classifcation03](https://github.com/hephaex/ML_class/blob/master/week3/week3_01_classification_03.png)

## Hypothesis Representation (가설 표현)
* 예시로 살펴본 암 검진에서 종양을 악성과 양성으로 검증하기 위한 가설을 세웠다. 
 - 종양이 크기에 따라(ᆨx) 학습하려는 현상을 만족하는 가설을 위해서
 - 여러가지 모델 중선형회기 모델(Linear regression model) 을 사용하기로 하였다.
 - 선형 회기 모델에 따른 가설 검증(예측 값)은 음성(0)과 양성(1)사이로 하였다. 
   - hθ(x) 는 0 =< hθ(x) =< 1 로
   - ![hypothesis representation01](https://github.com/hephaex/ML_class/blob/master/week3/week3_02_hypothesis_representation_01.png)
 - 이것을 htheta(x) = theta Transpose x로 나타낼 수 있다.
   - hθ(x) = (θT x)
 - 가설 hθ(x))를 logisting regression 모델을 위해서 조금 수정하면.
   - g(θT * x) 로 변환하였다.
 - 여기서 G 변환함수는
   - g(z) : 1 / (1 + e^-z)
   - z: real number
   - ![hypothesis representation02](https://github.com/hephaex/ML_class/blob/master/week3/week3_02_hypothesis_representation_02.png)
 - 변환된 G 함수를 Sigmoid function 혹은 logistic function 이라고 한다.
 - 변환된 G 함수를 다시 정리해서 쓰면
 - htheta(x)
   - = 1 / (1 + e^-z)
   - = 1 / (1 + e^-(θT * x) )
   - ![hypothesis representation03](https://github.com/hephaex/ML_class/blob/master/week3/week3_02_hypothesis_representation_03.png)   
 - G 함수에 대해서 그래프로 그려 보면
   - ![hypothesis representation04](https://github.com/hephaex/ML_class/blob/master/week3/week3_02_hypothesis_representation_04.png)
   - Z값이 음수 (z < 0) 일 때 0에 무한히 가까운 값에서 부터 증가하여 0.5까지 증가한다.
   - Z값이 양수 (z > 0) 일 때 0.5부터 증가하여 1에 무한히 가까운 값까지 증가한다. 
   - 왜냐하면 htheta(z) 는 0과 1 사이 값을 가지기 때문이다. 

* 가설의 결과 값을 해석해 보자.
* 가설 htheta(x)는 입력 X에 대해서 결과 Y가 1(양성)이때 확률이라고 하면.
 - ᆨx = [ x0, x1 ] = [ 1, tumorSize ] 가 된다.
 - 여기서 htheta(x)에 x(종양 크기, tumorSize) 를 0.7이라면 결과가 양성(1)인 확률이다.
   - hθ(x) = 0.7 : 70%의 확률로 양성이다. 
   - ![hypothesis representation05](https://github.com/hephaex/ML_class/blob/master/week3/week3_02_hypothesis_representation_05.png)
* 이것을 주어진 x와 매개변수 theta가 있을 때 결과(y)가 1인 확률이라면
 - hθ(x) = P(y = 1 | x, θ) 가 된다.
 - ![hypothesis representation06](https://github.com/hephaex/ML_class/blob/master/week3/week3_02_hypothesis_representation_06.png)
* 종양은 양성(1)과 음성(0)으로 구분되므로.
 - P(y = 0 | x, θ) + P(y = 1 | x, θ) = 1 이다.
*이것을 P(y = 0 | x, θ) 값을 기준으로 정리하면.
 - P(y = 0 | x, θ) = ᇂᇂ1 - P(y = 1 | x, θ) 이다.

![hypothesis representation07](https://github.com/hephaex/ML_class/blob/master/week3/week3_02_hypothesis_representation_07.png)   

## Decision Boundary
hθ(x) = g(θ0 + θ1x1 + θ2x2) 라고 해보자.
![Decision Boundary 01](https://github.com/hephaex/ML_class/blob/master/week3/week3_03_DecisionBoundary_01.png)

여기서 θ0, θᇂᇂ1, θ2를 예를 들면.
 - θ0 = -3
 - θ1 =  1
 - θ2 =  1
라고 하면 이 θ 값에 대한 벡터는
> [-ᆸ3 ;
>  1 ;
>  1  ]
이 된다.

이것에 전치행렬 θT로 바꾸면
* θT= [-3,1,1] 이다.

* ᆷz = θT 라고 하면, y = 1 일때
> -3x0 + 1x1 + 1x2 >= 0
> -3 + x1 + x2 >= 0
> 가 된다.
>
> 이것을 바꾸서 풀어 쓰면,
>
> (x1 + x2 >= 3) 일때 y = 1 로 예상할 수 있고,
> x1 + x2 = 3 을 그래프로 그리면
> ![Decision Boundary 02](https://github.com/hephaex/ML_class/blob/master/week3/week3_03_DecisionBoundary_02.png)
> 이것을 **decision boundary** 라고 한다.

그림에서 파랑과 보라색으로 두개의 집합이 나뉘었다.
- 파랑: false 라고 정의하고
- 보라: true 라고 정의하면.
> 이것을 나누는 선을 글 수 있는데
> 이 선은 가설함수 hθ(x) = 0.5 로 나타낼 수 있다.
>
> 여기서 가설함수에 따라서 경계가 나누었고,
> 이것을 이용하여 다른 입력에 대해서도 구분을 할 수 있게 된다.
>
> 예를 들면
> 5 - x1 > 0
> 5 > x1 이라면
> 결과는 y = 1 이 된다.

## Non-linear decision boundaries
비 선형 데이터는 logistic regression 하기 어려움이 있다.
- 고차항이 있을 수 있기 때문이다.
- 예를 들면 hθ(x) = g(θ0 * ᆨx^0 + θ1 * x^1 + θ3 * x1^2  +  θ4 * x2^2)
  - 입력에 대하여 θ에 전치행렬 θT를 쓰면.
  - [-1,0,0,1,1] 이 된다.

- y = 1 일때
  - -1 + x12 + x22 >= 0
  - x1^2 + x2^2 >= 1
  - 0을 중심으로한 반경 1인 원이 된다.
  - ![Non-linear decision boundary 01](https://github.com/hephaex/ML_class/blob/master/week3/week3_04_Non-linearDecisionBoundary_01.png)

예시처럼 비선형이라도 복잡하긴 하지만 logistic regression에
따른 decision boundary를 구할 수 있다.
더 복잡한 decision boundary를 구한다면 보다 많은 고차항을 사용해서 구할 수는 있다. 
![Non-linear decision boundary 02](https://github.com/hephaex/ML_class/blob/master/week3/week3_04_Non-linearDecisionBoundary_02.png)

# Logistic Regression Model

## Cost Function
θ의 파라메터를 정하기 위해서 cost fucntion(비용함수)를 사용해서 최적의 값을 구할 수있다.
> 학습 feature의 수가 m개 있다면 각각의 대응하는 θ는 n+1인 벡터가 된다.
> m examples에 대하여
> ![Cost Function 01](https://github.com/hephaex/ML_class/blob/master/week3/week3_05_CostFunction4LogisticRegression_01.png)
>
> - ᆨx0 = 1이다.
> - y ∈ {0,1} : y는 0 혹은 1값을 가진다.

θ에 대한 cost function J(θ)는
> ![Cost Function 02](https://github.com/hephaex/ML_class/blob/master/week3/week3_05_CostFunction4LogisticRegression_02.png)
 비용함수 cost(hθ(xi), y) = 1 / 2 * { hθ(xi) - yi }^2

 선형회기(linear regression)처럼 학습자료(traing data)가 개별(individual)적이라면
 비용함수 cost(hθ(xi), y)를 다음 처럼 고쳐 쓸 수 있다. 
> ![Cost Function 05](https://github.com/hephaex/ML_class/blob/master/week3/week3_05_CostFunction4LogisticRegression_05.png)

이것을 다시 개별(individual) 비용의 합으로 근사화하면 
비용항수 J(θ)는
> ![Cost Function 03](https://github.com/hephaex/ML_class/blob/master/week3/week3_05_CostFunction4LogisticRegression_03.png)


* 

![Cost Function 04](https://github.com/hephaex/ML_class/blob/master/week3/week3_05_CostFunction4LogisticRegression_04.png)

![Cost Function 06](https://github.com/hephaex/ML_class/blob/master/week3/week3_05_CostFunction4LogisticRegression_06.png)

![Cost Function 07](https://github.com/hephaex/ML_class/blob/master/week3/week3_05_CostFunction4LogisticRegression_07.png)

## Simplified Cost Function and Gradient Descent
![ᆭSimplified Cost Function and Gradient Descent 01](https://github.com/hephaex/ML_class/blob/master/week3/week3_04_Non-linearDecisionBoundary_01.png)
![ᆭSimplified Cost Function and Gradient Descent 02](https://github.com/hephaex/ML_class/blob/master/week3/week3_04_Non-linearDecisionBoundary_02.png)
![ᆭSimplified Cost Function and Gradient Descent 03](https://github.com/hephaex/ML_class/blob/master/week3/week3_04_Non-linearDecisionBoundary_03.png)
![ᆭSimplified Cost Function and Gradient Descent 04](https://github.com/hephaex/ML_class/blob/master/week3/week3_04_Non-linearDecisionBoundary_04.png)
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
![Advanced Optimization 01](https://github.com/hephaex/ML_class/blob/master/week3/week3_07_AdvancedOptimization_01.png)
![Advanced Optimization 02](https://github.com/hephaex/ML_class/blob/master/week3/week3_07_AdvancedOptimization_02.png)
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

![Multiclass Classification 01](https://github.com/hephaex/ML_class/blob/master/week3/week3_08_MulticlassClassificationProblem_01.png)
![Multiclass Classification 02](https://github.com/hephaex/ML_class/blob/master/week3/week3_08_MulticlassClassificationProblem_02.png)

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

