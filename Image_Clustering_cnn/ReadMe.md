

# **Classifier1** 만들기

정제된 데이터 셋을 만들기 위한 방법으로, 어떻게 하면 크롤링한 특정식당 특정메뉴 이미지들을 우리가 분류하고자 하는 카테고리로 나눠서 레이블을 붙일까 고민했다.

생각해낸 방법은, Semi-Supervised Learning으로 agglomerative clustering과 이를 이용한 KNN이었다.

아래의 설명은 Classifier_demo 파일로 실행해보았다.

----------


### **1. Agglomerative Clustering**

앞서서 InceptionV4 모델로부터 뽑아낸 이미지들의 Feature Vector를 기준으로 해서 가장 거리가 가까운 이미지들부터 순서대로 묶는 방법이다. 위의 예시는 layer 중 'Logits' 레이어에서의 feature vector를 사용해보았다. 여기서 hyperparamer 설정을 통해, 최소한 몇개의 이미지 수를 가진 클러스터가 몇개 이상이 생겼을 때 클러스터링을 멈추는 것을 설정할 수 있다. 또한 클러스터들 간의 거리를 mean 으로 할 것인지 max로 할 것인지 등등도 hyperparameter로 설정해준다. 

![enter image description here](http://i68.tinypic.com/2096kag.png)

![enter image description here](http://i67.tinypic.com/35214w3.png)

### **2. KNN**

이 과정에서 Semi-Supervised Learning을 구현한다고 할 수 있다. 위의 clustering 과정에서 보여준 그룹핑을 통해서, 사람인 내가 직접 10개의 내가 분류하고자 했던 카테고리들의 대표 이미지 몇개를 지정해준다. 

![enter image description here](http://i64.tinypic.com/219s1g9.png)

그 후 knn을 돌려 clustering이 되게 하여 레이블 폴더별로 분류해서 저장해주었다.

![enter image description here](http://i66.tinypic.com/2v3ijqq.png)

### **3. TSNE**

TSNE를 통해서 나의 분류기가 어떻게 clustering을 했는지 시각적으로 확인해 보았다.

![enter image description here](http://i66.tinypic.com/311uxbt.png)

Tsne로 변환 이후, 분류해낸 데이터들의 분포를 색깔별로 찍어보았다. 가장 넓은 부위에 퍼져있는 파란색 색깔은 음식과는 관계 없는 사진들이다. 나머지 사진들은 나름대로 가까운 이미지들끼리 같은 클러스터로 묶인 것을 확인할 수 있다.
