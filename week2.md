# Machine Learing
* progress: week #2
* 
* date: 2016.04.12

# Multivariate Linear Regression
* Multiple features
* Gradient Descent for Multiple Variables
* Gradient Descent in Practice I - feature scaling
* Gradient Descent in Practice II - Learning Rate
* Features and Polynomial Regression

## Multiple features
* simple features: House size and price
 - x: size(feet^2)
 - y: price($1000)
 - htheta(X) = theta0 + theta1 * X

![simple feature](https://github.com/hephaex/ML_class/blob/master/week2/week2_MultipleFeature_%231.png)

* multiple feature:
 - features: size, number of bedrooms, number of floors, age of home
   - x1: size
   - x2: number of bedrooms
   - x3: number of floors
   - x4: age of home
   -  y: price
 - Notation
   - n: number of features (n=4)
   - x(i): input(features) of i^th training example
   - xj(i): value of featre j in i^th training example
   
![multiplee features](https://github.com/hephaex/ML_class/blob/master/week2/week2_MultipleFeature_%232.png)

![multiplee features](https://github.com/hephaex/ML_class/blob/master/week2/week2_MultipleFeature_05.png)
여기서 X1(4) 는 the size (in feet^2) of the 4th home in the training set 이 된다.

다변항에 대하여 가설을 세워보자.
* previously: htheta(x) = theta0 + theta1 * x
* multiple : htheta(x) = theta0 + theta1 * x1 + theta2 * x2 + theta3 * x3 + theta4 * x4
 - 여기서 x1, x2, x3, x4는
   - x1: size
   - x2: number of bedrooms
   - x3: number of floors
   - x4: age of home

![multiplee hypothesis](https://github.com/hephaex/ML_class/blob/master/week2/week2_MultipleFeature_%233.png)

그러면 다변항에 대하여 일반적인 가설을 세워보면.
![multiplee hypothesis](https://github.com/hephaex/ML_class/blob/master/week2/week2_MultipleFeature_%234.png)

* hypotesis : htheta(x) = theta0 * x0 + theta1 * x1 + theta2 * x2 + theta3 * x3 + theta4 * x4 + ... + thetaN * xN
  - x0: 1이라면 위에서 세운 다변항과 같아진다.
  - 이것을 행렬로 연산하기 위해서 식을 세워보면.
    - x = [x0 ; x1 ; x2 ; ... ; xN]
	- theta = [theta0 ; theta1; theta2; ... ; thetaN ]
	- htheta = thetaT * x 가 된다.

이것을 multivariate linear regression 이라고 한다. 

## Gradient Descent for Multiple Valiables 다변수에서 경사하강법

어떻게 가설에 사용한 매개변수(parameters)를 맞출것인가?
여기서는 경사하강법을 이용해 보자.

다변수 선형 회귀 (multivariable linear regression)에서 x0 = 1 이라고 하면.

* 가설 Hypothesis는

```
Htheta(x) = theta' X = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... thetaN * xN
```
![GD4MV_01](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescent4MutipleFeature_02.png)

* 매개변수 Parameters는 N개의 다변수가 있다면 다음처럼 정리할 수 있다.
 - theta0, theta1, theta2, ... ,thetaN
* 비용함수 Cost Function은 N개의 다변수이다.
 - ᆭSquare error 방식으로 구하면
 - J(theta0, theta1, theta2, ,,, , thetaN) = 1 / (2 * M)  *  sum(i=1, M){htheta( X(i)) - Y(I)) ^2

* 경사하강법 (gradient descent) 으로 추정하면
Gradient Descent도 theta0, theta1에서 단지 θ값이 많아졌지만 알고리즘이 바뀌지 않았고, 구해야 할 θ값이 더 많이 반복되어 구해야 한다. 
```
반복항 {
         theta(J) := theta(J) - alpha * partitial derivative(theta J) * J(theta0, theta1, ,,, , thetaN)
}
```

* n=1 이일 때 경사하강법을 적용해서 theta0 와 theta1 에 반복하서 구할수 있다. 
![GD4MV_0ᆸᆸ3](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescent4MutipleFeature_03.png)

* 새로운 알고리즘으로 n >= 1 일때로 해서 바꾸어 보면
 - 편미분을 통한 비용함수를 다른 표현으로 바뀌었다.
![GD4MV_04](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescent4MutipleFeature_04.png)
 - 이전 알고리즘에서 n=1 일때 theta0 를 정리하면 
![GD4MV_05](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescent4MutipleFeature_05.png)
 - 새로운 알고리즘에서 theta0 는
![GD4MV_06](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescent4MutipleFeature_06.png)
 - 이전 알고리즘에서 n=1 일때 theta1 를 정리하면 
![GD4MV_07](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescent4MutipleFeature_07.png)
 - 새로운 알고리즘에서 theta1 는
![GD4MV_08](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescent4MutipleFeature_08.png)
 - 새로운 알고리즘에서 theta2 는
![GD4MV_09](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescent4MutipleFeature_09.png)

새로운 알고리즘은 N을 수에 따라 thetaN을 구할 수 있다.

## Gradient Descent in Practice I: Feature Scaling
* feature scaling
 - 다변수에서 각 변수에 대하여 스케일이 다를 때 이를 조정하는 것.
 
ex) x1과 x2가 있을 때 x1은 2000의 범위, x2는 5의 범위라면. 
 - x1: 면적 size(0~2000)의 범위,
   - 면적의 대역 폭 (2000)으로 나누어 준다.
   - ᆫsize(feet^2) / 2000
   - 0 =< x1 =< 1
 - x2: 침실 수 
   - 침실 수 (number of bedrooms)는 (1~5)의 범위
   - 침실 수의 변수 폭 (5)으로 나누어 준다.
   - 0 =< x2 =< 1
   
![GDPI_01](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeI_01.png)

* feature scaling에서 음수 값의 폭을 가진다면.
  - -1 =< x(i) =< 1 로 표현할 수 있다.
  - x0 = 1이라면 x1은 -ᆸ3~3, -0.3 ~ 0.3 정도는 괜찮다.

![GDPI_02](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeI_02.png)

* Mean Normalization
  - ᆨxi를 xi - ui 로 바꾸면, 평균 값이 0에 가깝게 된다.
  - ᆨx1 <= (x1 - u1) / s1
    - u1: avg value of x1 in taining set
	- s1: range : (max - min)
  - x2 <= (x2 - u2) / s2
    - u2: avg value of x2 in taining set
	- s2: range : (max - min)
    
![GDPI_03](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeI_03.png)
  
## Gradient Descent in Practice II: Learning rate
* 경사 하강법을 사용했을 때 이것이 올바르게 동작하는 검증에 대하여
* Learning Rete Alpha 값을 선택에 대하여.
![GDPII_01](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_01.png)

* 경사하강법 (Gradient Descent)의 목표
 - 비용함수 J(theta)를 최소화 하는 값을 찾는 것이다. 
 - 경사 하강법을 반복(Interations)하면 할 수록 J(theta)는 줄어든다.
 - x는 theta가 아니다.
 - x: No. of iterations 반복 수이다.
 - 여기 예시에서는 100 반복에서 200반복일 때 J(theata)의 변화량과
 - 300에서 400 반복에서 J(theata)의 변화량이 다름을 알 수 있다.
 - 계속 반복하면 비용함수 J(theta)가 최소가 되는 값을 찾을 수 있다.
 - 하지만
 - 30번 반복해서 비용함수 J(theta)를 찾는 것과
 - 3000번 반복해서 비용함수 J(theta)를 찾는 것과
 - 3백만번 반복해서 비용함수 J(theta)를 찾는 것은 다르다.
 - 매번 이것을 고려하지 않기 위해서 자동 수렴 검증 (Automatic convergence test)을 사용하기도 한다.
 - 한번 반복에 10^-3보다 비용함수 J(theta)가 줄어든다면 수렴위치로 해서 멈춘다.
![GDPII_02](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_02.png)

* 겸사하강법이 올바르게 동작하는 지 검증해보자. 
 - 반복할 수록 J(theta)가 커진다면
   - 경사하강 법은 올바르게 동작하지 않는 것이다.
   - Alpha(Learning Rate) 값이 크기 때문에 수렴하지 않는 것이다. 
   - 이때 ᆮAlpha(Learning Rate) 를 작게한다.
 - 반복할수록 J(theat)값이 크거나 작거나 한다면.
   - 이때도 ᆮAlpha(Learning Rate) 를 작게한다.
 - 작은 ᆮAlpha(Learning Rate)라면 매 반복마다 J(theta)는 감소할 것이다.
   - 하지만 너무 작은 Alpha(Learning Rate)라면 많이 반복해야 원하는 J(theta)를 찾을 수 있다.
![GDPII_03](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_03.png)

* 예시
 - In Graph C: The cost function is increasing, the learning rate is set too high.
 - Both Graph A and B: converge to an optimum of the cost function
   - Graph B: so very slowly, so its learning rate is set too low.
   - Graph A: good choosen alpha
![GDPII_05](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_05.png)

* 요약
 - Alpha (Learning Rate): 작다면 천천히 수렴한다.
 - ᆮAlpha (Learning Rate): 너무 크다면
   - J(theta)는 매 반복마다 줄어들지 않는다.
   - 수렴하지 않을 수도 있다.
 - ᆮAlpha를 작은 값에서 (예시 0.001 -> 0.01 -> 0.1) 큰값으로 바꿔가면서
   - J(theta)가 크게 바뀌기 때문에 Alpha 를 반복시 적절히 바꾸어 가는 것이 좋다. 

![GDPII_04](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_04.png)

## Features and Polynomial Regression
* 학습 알고리즘이 때로는 매우 강력해서 대상에 잘 맞을 수도 있다.
* 학습 알고리즘이 선형성이 아닌 비 선형성일 수 있다.
 - 비선형 학습 함수는 선형에 비해서 복잡하다.
 
![FPR_01](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_FeaturesPolinomialRegression_01.png)

* 집값 예측
 - 집값의 가설함수는 앞마당 길이와 폭으로 나타낸다면. 
 - htheta(x) = theta0 + theta1 * frontage + theta2 * depth
   - frontage: x1
   - depth: x2
 - 여기서 면적(Area)는
   - X = frontage * depth 로 나타낼수 있다.
   - 이것으로 가설 함수를 다시 세우면.
   - htheta(x) = theta0 + theta1 * X 가 된다.
   - 여기서 X는 면적(Area) 이다.
 - 따라서 집값 문제도 어떻게 변수를 정하는 지에 따라서 가설 함수는 달라진다.

* Polynomial regression
 - 집값 (price): y
 - 면적 (size): x
 - 이것을 가지고 여기서는 두가지 모델로 세워보면.
   - ᇂ1) 2차함수: theta0 + theta1 * x + theta2 * x^2
   - ᆻ2) 3차함수: theta0 + theta1 * x + theta2 * x^2 + theta3 * x^3
 - 2차 함수는 크기가 증가함에 따라 감소할수도 있다.
 - 3차 함수는 크기가 증가하여도 감소하지 않는다.
 - 따라서 기계학습에서는 3차 함수를 사용하는 것이 좋다.
![FPR_02](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_FeaturesPolinomialRegression_02.png)

* ᆱ특징 선택 (choice of feature)
 - 2차 가설함수: htheta(x) = theta0 + theta1 * (size x) + theta2 * (size x)^2
   - 증가항도 있지만, 감소하는 부분도 있기 때문에 가설함수로 적합하지 않다. 
 - SQRT를 사용해서 변형한 가설함수: htheta(x) = theta0 + theta1 * (size x) + theta2 * SQRT(size x)
   - 계속 증가하므로 가설함수로 보다 적합하다.  
![FPR_03](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_FeaturesPolinomialRegression_03.png)

* quiz
![FPR_04](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_FeaturesPolinomialRegression_04.png)

# Computing Parameters Analyticaaly
* Normal Equation
* Normal Equation Noninvertibility

## Normal Equation

![CPA_NE_01](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquation_01.png)
![CPA_NE_02](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquation_02.png)
![CPA_NE_03](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquation_03.png)
![CPA_NE_04](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquation_04.png)
![CPA_NE_05](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquation_05.png)
![CPA_NE_06](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquation_06.png)

## Normal Equation Noninvertibility

![CPA_NEN_01](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquationNonInvertiability_01.png)
![CPA_NEN_02](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquationNonInvertiability_02.png)
