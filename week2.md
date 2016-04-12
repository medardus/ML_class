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

![GDPII_01](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_01.png)
![GDPII_02](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_02.png)
![GDPII_03](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_03.png)
![GDPII_04](https://github.com/hephaex/ML_class/blob/master/week2/week2_GradientDescentPracticeII_04.png)

## Features and Polynomial Regression

![FPR_01](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_FeaturesPolinomialRegression_01.png)
![FPR_02](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_FeaturesPolinomialRegression_02.png)
![FPR_03](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_FeaturesPolinomialRegression_03.png)

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
