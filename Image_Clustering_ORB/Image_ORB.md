## Image Feature Matching -  **ORB** 

구글에서 어떤 식당의 특정 메뉴 이미지를 검색해서 나온 사진들을 크롤링 해왔을 때, 내가 원하던 메뉴 사진이 아닌 각종 광고들, 셀카들 등등 심지어 전혀 관계 없는 사진들도 같이 따라온다.

내가 원하는 데이터 셋을 만들기 위해, 즉 해당 식당 해당 메뉴와 관련된 사진들만을 어떻게 걸러낼까를 고민하던 과정해서 시도해 본, open cv 라이브러리의 Feature Detection and Description 중, 가장 결과가 빠르게 되고 성능도 괜찮았던 방법이 바로 ORB였다.

----------
우선 내가 생각했던 가장 모범적인 '다운타우너 아보카도버거' 사진은 아래와 같았다.

![enter image description here](http://i65.tinypic.com/28ib3nd.jpg)

먼저 이미지들에 있는 코너와 엣지로부터 feature들을 뽑아내고(ORB), 가장 모범적인 이미지와 matching되는 거리가 작은 사진들을 한번 뽑아보았다.

![enter image description here](http://i66.tinypic.com/4n8uw.png)

확실히 설정했던 모범 사진과 비슷한 사진들이 뽑혔다.
가장 거리가 멀었던 사진 10개는 아래와 같았다.
![enter image description here](http://i65.tinypic.com/vypead.png)

두 사진이 어떻게 matching되는지를 한번 그려보기도 했다.

![enter image description here](http://i67.tinypic.com/2j5azra.png)

확실히 저기 찍힌 점들이 사진에 있는 엣지와 코너들에 몰려있다.


----------
위의 matching 은 가장 간단한 brute-force matcher로 매칭한 결과이다. 이 방법 말고도 좀 더 빠른 FLANN matcher로도 가능하다. 결과는 비슷비슷했다.

