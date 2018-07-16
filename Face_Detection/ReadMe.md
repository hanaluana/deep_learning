# **Dlib**을 이용한 **Face Detection**

식당들의 음식 사진을 구글에서 크롤링 해왔을 때, 제일 먼저 걸러야 할 사진들 1순위가 사람의 얼굴이 나와있는 사진이다. 가장 정확도가 높게 걸러지는 여러가지 face detection api들을 찾아 보았다.

 - open cv - Haar Classifier
 - open cv - LBP Classifier
 - dlib - get frontal face detection
 - dlib - cnn face detection

위 네가지 기법을 모두 사용해 본 결과, 가장 정확도가 높은 것은 역시 마지막 cnn face detection 이었다.


----------

###설치###
먼저 dlib 설치를 위해서는 많은 설치가 필요했다.

Brew install cmake, brew install boost, brew install boost-python ---with-python3, brew install gtk+3 boost, brew install dlib, pip3 install dlib, pip3 install scikit-image

그러고는 직접 git에서 클론해왔다.

    Git clone https://github.com/davisking/dlib

또한 나같은 경우에는 파이썬 버전 문제 때문에, /usr/local/opt/boost-python/lib/ 에 있는 파일들 중 앞에 libbost_python3 로 시작하는 파일들을 전부 libboost_python으로 시작할 수 있게 바꿔줬다.

그리고 나서 설치한 dlib 폴더에 들어가서 setup을 설치해준다.

       python setup.py install –yes USE_AVX_INSTRUCTIONS

설치 끝!!


----------

###cnn face detection###
 

> https://arxiv.org/pdf/1502.00046.pdf

다른 detection들과 가장 차별화된 점이라면 학습되는 windows들 차이이다. 보통은 training 시킬 때, training image에서부터 positive image window 들과 negative image window 들의 일부가 subsample되어 binary classifier로 학습된다. 그러나 이 MMOD 기법으로는 windows scoring function을 통해 overlap 되지 않는 범위에서 windows들 전체로 학습한다. 또한, 전자는 object detection 전체가 optimize 되는 것이 아닌 부분부분인 binary classifier를 optimize하지만, 후자는 detection 전체를 최적화하는 parameter 세트를 구한다는 점이다.


----------

워낙 무겁기 때문에, 많은 사진들을 detection 하고 싶다면 gpu로 돌려야 한다. 나는 일종의 toy project로 올리기 위해 10장 정도의 사진만을 돌렸기 때문에 cpu로 충분하게 돌릴 수 있었다.

얼굴 부위에 confidence score를 반환하는데, False Negative를 줄이기 위해서 낮은 confidence score를 설정했다.

확실히 get_frontal_face 에 비해서는 훨씬 성능이 좋았따. 사람 얼굴이라면 왠만하면 거의 다 detect 해내는 결과를 보여주었다.

![예시](http://i65.tinypic.com/2i700uv.png)
