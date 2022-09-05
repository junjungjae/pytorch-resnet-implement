# pytorch-resnet-implement

- Reference
    1. pytorch 공식 resnet 코드(https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html)
    2. resnet 논문(HE, Kaiming, et al. Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. p. 770-778.)

- 추후 개선점
    1. 이미지/라벨 입력 -> 전처리 -> dataloader 탑재 -> 학습 -> 모델 저장 및 학습 결과 시각화까지 일련의 과정 자동화
    2. 가중치 초기화 기능 추가

<br><br>
- 최종 목표
    1. classification, object detection, image segmentation 등 다 구현해서 나만의 프레임워크 만들어서 필요할때 가져다 쓰기
       - 이미 github 등에 구현되 있는 모델 중에 논문 검증용으로만 만들어져서 있는 모델들이 많아 써먹기가 생각보다 힘들다.
    <br><br>
    2. 기존 구조와 유사해도 좋으니까 나만의 모델 만들어보기
<br><br>
- 최최종 목표
    1. tensorflow, pytorch 손발처럼 다루기
    2. tensorflow2pytorch, pytorch2tensorflow 포팅 시스템 만들어보기
