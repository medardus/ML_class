# Machine Learing
* progress: week #3
* 
* date: 2016.04.??

## Classification and Representation
* Classification
* Hypothesis Representation
* Decision Boundary

## Logistic Regression Model
* Cost Function
* Simplified Cost Function and Gradient Descent
* Advanced Optimization

## Muliclass Classification
* Multiclass Classification: One-vs-all

## Solving the Problem of Overfitting
* The Problem of Overfitting
* Cost Function
* Regularized Linear Regression
* Regularized Logistic Regression

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
