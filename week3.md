# Machine Learing
* progress: week #3
* 
* date: 2016.04.12

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

![CPA_NEN_02](https://github.com/hephaex/ML_class/blob/master/week2/week2_LinearRegressionMultipleVariables_NormalEquationNonInvertiability_02.png)

## Classification and Representation

### Classification
* Classification 예시
 - Email: Spam / Not Spam?
 - Online Trasactions: Fraudulent (Yes / No)?
 - Tumor: Malignant / Benign?

예시한 Classification들은 결과값 Y가 2가지를 가진다. 예시에서 암검진이라면
양성은 1을 음성은 0, 즉 Y를 수학으로 모델링하면 0 혹은 1을 같는다.
0은 Negative Class (ex. Benign Tumor) 이며 1은 Positive Class (ex. malignant tumor) 이다.

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

따라서 linear regression은 classifcation 문제에 잘 사용하지 않는다.

Classification: y = 0, y = 1의 이산값을 가지면 htheta(x)는 > 1 or < 0 이다.
logistic regression은 0 =< htheta(x) =< 1 이 되며,
역사적인 이유로 classification을 logistic regression 이라고 한다.



* Hypothesis Representation
* Decision Boundary
