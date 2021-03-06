# Machine Learing
* week #1

## Welcome to Machine Learing!
* Machine Learning 이란?
  - 구글에서 machine learning 이라 검색할 때, 이 결과를 알려주는 것
  - 페이스 북에서 친구 사인을 추천하는 것
  - 애플에서 iPhoto등으로 친구 얼굴을 인식해 주는 것.
  - 이메일에서 스팸이나 자동으로 분류해 주는 것
  - the science of getting computers to learn, without being explicitly programmed.

* 기계학습의 예시
  - 집을 청소하는 로봇
  - 로봇이 물건을 집고 올리고 다른 곳에 놓은 모든 행동

* ᆰᆼDatabase mining
  - web cliccked data collected to use machine learing
  - 사용자의 행동을 읽어서 사용자에게 더 좋은 서비스를 제공하기 위함.

* Medical Records
  - 전자로 기록된 의료 기록을 통해서 질병을 좀더 잘 이해할 수 있음.

* Biology
  - ᆰDNA분석등 많은 데이터를 수집
  - gene sequence, DNA sequences ...
  - Human Genome을 좀 더 잘 이해하기 위함.

* Applications can't program by hand
  - [autonomous helicopters](https://www.youtube.com/watch?v=_qah8oIzCwk)
  - 손글씨 인식.
  - 자연어 인식.
  - 컴퓨터 비전.

* Self-customized program
  - 아마존, 넷플릭스에서 추천 시스템.

* 인공지능 (Artificial Intelligence)
  - 머신 러닝이 인공 지능으로 가는 길이라 믿는다.
  - machine learning to do it by itself.
  - Many Scientiest think the best way to make progress on this is through learning algorithms called neural networks which mimic now the homan brain works.

* 이 과정에서 배울 것
  - State of the art machine learning algorithms.


## Supervised Learning
* Housing price prediction

* ᆭSupervised learning
  - right answer given
  - housing price
     - what is right price?
  - what is the actual price ?
  - Breast cancer
     - 5개의 악성/양성
     - benign (양성): 칼슘 성분으로 된 미세한 석회질
	 - malignant (악성):
	 - maligant tumor (악성종양), carcinoma(암종), ᆫᆼsarcoma(육종)

* Regression (superviced learing)
  - predict continuous valued output

* Classification (superviced learing)
  - Discrete valued output (0, 1, ,,, )
  - 유방암의 경우 사이즈 만으로 구분을 두어서 악성/양성을 평가하기 어렵다.
  - 다양한 features로 되었다면 예측은 더 어려워진다. 오류또한 증가



- explicitly what is the so-called right answer.
  - whether it's benign or malignant

* Unsupervised Learning
  - don't have any label
  - find some structure in the data!
  - seperated cluster (ex. 암 2개 분류: 악성 vs 양성)
  - 예시: 구글 뉴스
    - 수많은 뉴스 중에서 자동으로 구분하여 묶어줌
  - unsuperviced learing or clustering 예시:
    - Organize computer cluster
	- Social network analysis
	- Market segmentation
	- Astronomical data analysis

* ᆱCocktail Party 문제
  - 두사람이 있고, 동시에 말을 할 때. 마이크로 두개가 각각 다른 소리가 저장.
  - Octave 코드: [ᇀᇀW, s, v] = svd( (repmat(sum(x.*x,1), size(x,1),1).*x) *x');
  - ᆭᆶᆭSVD: singular Value Decomposition
  - 처음 배우는 사람들은 Octave로 알고리즘을 익힐 것을 추천, 이후 Java나 C++로 확장...

## Model representation

* 주택 가격 예측
 - 집의 크기에 따라 집의 가격이 결정
 - 예측 값또한 real-valued output이다. 
 - 따라서 Supervised Learing 이며, regression 임.
 - 크기가 1250feet^2 인 집의 예상 가격을 구해보자.
 - 직선으로 함수를 구했다면 1250: 220k 가 예상 주택 가격이다. 

![Housing Prices](https://github.com/hephaex/ML_class/blob/master/week1/week1_model%20representation_hosing%20price.png)

* Training set of housing prices
 - m: Number of traing example (학습 수, m=47)
 - X's: "input" variable / feature
   - X(1): 2104
   - X(2): 1416
 - Y's: "output" variable / target variable
   - Y(1): 460
   - Y(2): 232
![training sets](https://github.com/hephaex/ML_class/blob/master/week1/week1_model%20representation_traing%20set.png)

* Model Representaion 과정
  - Training set
  - Learning Algorithm
  - H: Hypothesis 학습 알고리즘에 따른 모델 정립
  - 1변수 추정이라면 Htheta(X) = theta0 + theta1(X)
  - 다변수 추정이라면 더 복잡한 theta항이 나타난다. 
  - ex)
     - input X's: size of house
	 - hypothesis htheta(X): model representation
	 - output Y's: 예상 값 Estimated price 
![reprent h](https://github.com/hephaex/ML_class/blob/master/week1/week1_model%20representation_represent%20h.png)

## Cost Function

* Parameters of model: theta 0, theta 1, ,,,
![cost funtion](https://github.com/hephaex/ML_class/blob/master/week1/week1_cost%20funtion_training%20set%20and%20hypothesis.png)
* choose theta0 and theta1
![cost funtion hypothesis](https://github.com/hephaex/ML_class/blob/master/week1/week1_cost%20funtion_hypothesis.png)
* Hthets(X) 을 Y에 가깝게 만들어 주어야만 한다.
  - overal objective function for linear regression
  - ToDo: minimize theta0, theta1
  - Cost Function J(theta0, thera1): Square error function
  - Error funtion: ᆱCost Function과 실제 값과의 차이
    - 선형 가설 모델(Lineare hypothesis regression)에서는 Square error function을 주로 사용
	- 다변이나 비선형이라면 다른 오류 함수를 선택
![cost funtion choose theta0 theta1](https://github.com/hephaex/ML_class/blob/master/week1/week1_cost%20funtion_choose%20theta0%20theta1.png)

## Cost Function 

* Hypothesis function: htheta(x) is funciton of X
* Cost function: J(theta1) is function of theta1
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_cost%20function_summary.png)

* hypothesis function과 Cost Function의 관계
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_cost%20functinn%20intunition_intro.png)

* case: theta1: 1 일 때 Cost Function의 0
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_cost%20functinn%20intunition_htheta(x)%20Cost(theta1).png)

* case: theta1: 0.5 일 때 Cost Function은 0.5
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_cost%20functinn%20intunition_htheta(x)%20Cost(theta1)_%232.png)

* case: theta1: 1.5 일 때 Cost Function은 0.5
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_cost%20functinn%20intunition_htheta(x)%20Cost(theta1)_%233.png)

* case: theta1: ... 일 때 Cost Function을 구할 수 있다. 

## 등고선 선도 (conture plot)
* 등고선 선도(Conture plot) theta 값을 가지고 비용함수 ᆱ(cost function)을 3차원으로 그린 그림.
 - 예시는 theta0, theta1에 2개를 가지고 Cost function을 3차원으로 그림
 - theta0, theta1, cost function(theta0, theta1)
 - bow shaped function이 구해짐.

![cost function 3d](https://github.com/hephaex/ML_class/blob/master/week1/week1_gradient%20decent_intro.png)

* 효과적인 알고리즘은 자동으로 theta0, theta1으로 cost funcition J를 최적값을 구해준다.

## 경사하강법 (gradient decent)

* 경사하강법 (gradient decent) 목적
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_gradient%20decent_goal.png)

* 반복하여 최적의 비용함수 theta (J)를 찾음

* 등고선 선도 (conture plot) J(theta0, theta1)
  - 입력: 변수가 2개 (theta0, theta1)
  - 출력: 비용함수 (cost function) J(theta0, theta1)

![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_gradient%20decent_intro.png)


* ᆼAssignment 와 truth assertion 비교
 - ᆼAssignment
  - a := b
  - a := a +1
 - truth assertion
  - a = b
  - a = a + 1 (X)

* 경사하강법 (gradient decent) 알고리즘
 - 비용함수 (cost function)을 최소화
 - 모든 theta0, theta1에 대하여도 비용함수 J를 구할 수 있지만,
 - 조금씩 theta 값들을 바꿔가면서 지역 최저점 (local minimum)을 구한다.

* theta0를 gradient decent로 추정
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_gradient%20decent_theta0.png)

* theta1를 grdient decent로 추정
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_gradient%20decent_theta1.png)

여기서 theta0일때와 theta1일 때 각각 비용함수 최소값이 달라 질수 있다는 점에 유의하자.

* ᆼalpha(learning rate)
 - 크면 더 빨리 수렴하지만 정확도는 떨어진다.
 - 작으면 늦게 수렴하지만 정확도는 올라간다. 
 - 이것을 이용해서 비용함수 theta(J)가 감소한다면 alpha를 작게한다.

* hypothesis 함수에 따른 cost function 찾기
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_gradient%20decent_htheta.png)
 
* current: simultaneous update
![hypothesis cost](https://github.com/hephaex/ML_class/blob/master/week1/week1_gradient%20decent_algorithm.png)

* "batch" 경사하강법 (gradient decent) 
 - theta(J)가 최소가 될때 까지 반복한다.
 - each step of gradient descent uses all the training examples.
 - 이 과정 (gradient descent) 중 J가 감소한다면 alpha를 작게한다.

* 등고선 함수 (convex function, bowl shaped)의 지역 최소값, 전역 최적값
 - 지역 최소값 (local optimal minimum)
 - 전역 최적 값을 가진다. grobal optiomal
 - 지역 최적 값은 가지지 않는다. No local optimum

