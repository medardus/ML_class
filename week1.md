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
