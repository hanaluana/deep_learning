# Feature Extraction

데이터셋을 만들기 위해 하나하나 내가 직접 labeling하기 보다는, 어떻게 하면 좀 더 효율적으로 label을 할까 고민을 해보았다.

우선, Classification을 하고자 했던 category들은, 해당 사진이 음식의 근접사진, 초근접사진, 원접사진, 다른 메뉴와 함께 있는 사진, 메뉴판 사진, 내부 사진, 외부 사진, 나머지 상관없는 사진 등등이었다. 대략 8개의 레이블을 하나하나 사진마다 달아주기에는 너무 시간이 오래 걸리기에 다른 방법을 고민해 보았다.

agglomerative clustering을 통해서 좀 더 쉽게 label을 달아주는 방법 이었다. 여기서 사진들을 어떻게 clustering을 할까 고민을 하다가, 나중에 우리가 Inception V4를 transfer learning을 통해 분류기를 만들 것임을 감안하여, Inception V4 모델에서 사진들의 Feature Vector를 추출해낸 것으로 클러스터링을 해보기로 했다. 

총 4개의 layer에서 추출을 시도했다: Logits, PreLogitsFlatten, globalpool, AuxLogits

해당 폴더는 Feature Extraction을 돌려본 결과를 Demo에 담아보았다.
