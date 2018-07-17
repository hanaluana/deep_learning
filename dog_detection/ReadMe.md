
# 집나간 강아지 찾기

### **Object Detection** using **Tensorflow Api**

나의 애완견 뚜비를 detection해내는 모델을 만들어 보는 미니 프로젝트이다. 만약에 뚜비가 가출을 했을 경우를 대비해, cctv에 담긴 강아지가 나의 뚜비가 맞는지 아닌지를 detection해내는 모델을 만들고 싶었다.

### 1. **Data** 수집
뚜비의 사진을 많이 찍는 것 보다는 차라리 동영상을 하나 찍어서 데이터를 수집해 보았다. 다양한 각도가 나오게 찍었고, opencv를 사용하여 동영상 컷들을 캡쳐하여 140개 정도의 이미지로 저장하였다.

![enter image description here](http://i66.tinypic.com/38wag.gif)

### 2. **Annotation** 만들기
LabelImg라는 툴을 사용하여 annotation을 만들어 주어 xml 파일로 저장하였다.
![enter image description here](http://i63.tinypic.com/34pmh34.png)

### 3. **TFR Dataset** 만들기

Tensorflow Api로 돌리려면 데이터셋이 TFGRecord 파일 포멧이어야 한다. 따라서 기존에 있던 예시 중 하나인 create_kitti_tf_record.py를 참고하여 create_dog_tf_record.py를 만들었다. 

또한 레이블 이름을 숫자아이디로 변환해줄 pbtxt파일도 생성했다.

    item {
     id: 1
     name: ‘ddubi’
    }
그 후에, tf 레코드 파일을 만들어 준다.

    python create_dog_tf_record.py
    --label_map_path=./doglabel.pbtxt
    --data_dir=./images 
    --output_path=./output
   결과물로는 train.record, test.record 파일이 생긴다.

### 4. **Config** 파일 만들기

기존 cnn model을 가지고 학습을 시켜야 하므로, 어떤 모델을 쓸지 결정해야 한다. 가장 빠른 SSD모델을 많이 사용하기는 하지만, 나는 이번 경우에는 Faster RCNN resnet101을 가지고 학습시키기로 했다. 속도는 SSD에 비해서는 느리지만, 성능이 좀 더 좋다고 한다.
config 파일 또한 기존에 예시로 나온 kitti의 config파일을 참고하여 만들어보았다. 
![enter image description here](http://i67.tinypic.com/2vuf2gx.png)
google cloud에 돌리려고 하지만, 나는 아직 돈이 없어서 공짜 credit안에서 해결해야 하므로, step은 일단 2500정도로만 잡았다. data augmentation 옵션은 우선 horizontal flip으로만 추가해주었다.

### 5. **Bucket**만들고 추가해주기
![enter image description here](http://i66.tinypic.com/fkmfkw.png)

위의 7개 파일을 모두 google cloud 버킷에 data폴더 안에 넣어주었다.
또한 tensorflow 폴더에 들어가 여러가지 설정들을 바꿔준 후, 정말 본격적으로 training을 시키려고 하였다.

    ~/Documents/google-cloud-sdk/bin/gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
    --runtime-version 1.8 \
    --job-dir=gs://pekenese/model_dir \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region us-central1 \
    --config object_detection/samples/cloud/cloud.yml \
    --\
    --model_dir=gs://pekenese/model_dir \
    --pipeline_config_path=gs://pekenese/data/faster_rcnn_resnet101_ddubi.config

그런데 아직까지도 계속 에러가 너무 많이 뜨고 있다. 이게 각종 패키지들과 파이썬 버전 문제 때문인 것으로 추정된다. 조만간 성공하면 결과를 업데이트 해보겠다!

