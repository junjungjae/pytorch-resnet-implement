import torch.nn as nn


class plainblock(nn.Module):
    bottleneck_multiple = 1  # bottleneck block 구현 시 channel을 맞춰주기 위한 계수

    def __init__(self, in_channel, out_channel, stride=1):
        super(plainblock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # Convolution 구현부
        # plainblock의 경우 기존 모델들과 유사하게 layer를 쌓음
        # 다만, feature map resizing 과정에서 maxpooling 대신 stride를 2를 부여하여 resizing
        # resizing은 각 Convolution block에서 한번만 이루어짐
        # 연산량 -> maxpooling 승, 학습 측면 -> stride 승(kernel은 trainable 하기 때문에)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        # resizing으로 인해 기존 입력 데이터와 차원이 맞지 않거나
        # block의 출력부와 입력값의 차원이 맞지 않는 경우
        # 두 값을 더해주기 위해 shortcut의 차원을 맞추는 과정 수행
        if (stride != 1) or (in_channel != out_channel * plainblock.bottleneck_multiple):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * plainblock.bottleneck_multiple, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * plainblock.bottleneck_multiple)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class bottleneckblock(nn.Module):
    bottleneck_multiple = 4  # bottleneck block 구현 시 channel을 맞춰주기 위한 계수

    def __init__(self, in_channel, out_channel, stride=1):
        super(bottleneckblock, self).__init__()

        # bottleneck block의 특수한 구조
        # 3*3 kernel을 2개 사용하는 대신 1*1, 3*3, 1*1 kernel 사용
        # 동일한 작업을 수행하는 효과를 얻으면서 연산량 감소 & 활성화 함수 사용 증가로 인한 비선형성 증가의 이점을 가짐
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel * bottleneckblock.bottleneck_multiple, kernel_size=(1, 1),
                               stride=(1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * bottleneckblock.bottleneck_multiple)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        # plainblock과 동일하게 resizing이나 입력부와 출력부의 차원이 맞지 않을 경우 맞춰주는 과정 수행
        if (stride != 1) or (in_channel != out_channel * bottleneckblock.bottleneck_multiple):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * bottleneckblock.bottleneck_multiple, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * bottleneckblock.bottleneck_multiple)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out