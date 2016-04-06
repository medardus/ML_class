# Machine Learing
* progress: week #2
* 
* date: 2016.04.12

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


