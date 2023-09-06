### 환경을 위한 행동, 모두가 참여하는 서비스 (PICKTRE, Pick Trash)

![tensorflow-version](https://img.shields.io/badge/tensorflow-2.9.1-FF6F00?logo=tensorflow)
![python-version](https://img.shields.io/badge/python-3.10.9-3776AB?logo=python)
[![license](https://img.shields.io/badge/License-GPL-7F5AB6)](https://github.com/PICKTRE/PICKTRE_ai/blob/main/LICENSE)

## 소개
<img src="https://github.com/PICKTRE/PICKTRE_backend/assets/101933437/bb49dce2-a814-4722-8545-8b91e318f268" width="300" height="200">
<br>

> 환경을 위한 행동, 모두가 참여하는 서비스 (PICKTRE, Pick Trash)

현대 사회에서 쓰레기 문제는 점점 심각해지고 있으며, 쓰레기의 적절한 처리는 환경보호와 지속가능한 발전을 위해 중요한 문제로 인식되고 있습니다.

하지만, 여전히 공공장소에서 쓰레기를 적절하게 버리지 않는 경우가 많아 환경오염과 공공장소의 미관을 해치는문제가 발생하고 있습니다.

이러한 배경에서 PICKTRE는 쓰레기를 적절하게 처리하고, 시민들의 쓰레기 버리기 습관을 개선하여 환경보호와 재활용 문화를 확산 시키는데 목적을 두고 있습니다. 또한, 사용자들이 쓰레기를 버릴 때 보상을 제공하여 쓰레기 문제에 대한 인식과 참여 향상을 목표로 합니다.

[PICKTRE 홈페이지](https://picktre.netlify.app "PICKTRE") 

### 개발 환경
#### requirements.txt
---
https://github.com/PICKTRE/PICKTRE_ai/blob/8d565980f1e721da5b1a30908f707a05277bf6ab/requirements.txt

#### 사용 데이터셋
---
TrashBox

https://github.com/nikhilvenkatkumsetty/TrashBox

다음과 같은 쓰레기 종류에 대한 데이터셋을 사용하였습니다.

Medical waste : Syringes, Surgical Gloves, Surgical Masks, Medicines( Drugs and Pills) [Number of images: 2010]

E-Waste : Electronic chips, Laptops and Smartphones, Applicances, Electric wires, cords and cables [Number of images: 2883]

Plastic : Bags, Bottles, Containers, Cups, Cigarette Butts (which have a plastic filter) [Number of images: 2669]

Paper : Tetra Pak, News Papers, Paper Cups, Paper Tissues [Number of images: 2695]

Metal : Beverage Cans, Cnostruction Scrap, Spray Cans, Food Grade Cans, Other metal objects. [Number of images: 2586]

Glass : [Number of images: 2528]

Cardboard : [Number of images: 2414]

### 모델 아키텍쳐
![distillation](https://github.com/PICKTRE/PICKTRE_ai/blob/7c49407ce573ef79d31fd04605564b15c80d3420/src/diagram.png)

사용하게 될 모델의 구조는 위와 같습니다.

knowledge Distillation을 통해 ResNet152V2를 finetune한 모델의 가중치를 경량화된 ResNet50V2 커스텀 모델에 학습시켰으며, 이를 통해 적은 파라미터 개수로도 비슷한 성능을 보이는 모델을 구현하였습니다.

현재 서비스에 사용된 모델은 student model에 적용된 ResNet50V2 finetuned 모델이며, 이후 위 모델로 변경할 예정입니다. 이를 통해 같은 파라미터여도 더 좋은 성능을 기대할 수 있습니다.

![model](https://github.com/PICKTRE/PICKTRE_ai/blob/7c49407ce573ef79d31fd04605564b15c80d3420/src/result.PNG)
현재 서비스에 사용된 모델의 성능은 위와 같습니다.


### Reference

#### 1. Knowledge Distillation
---
https://arxiv.org/abs/1503.02531

#### 2. ResNet
---
https://arxiv.org/abs/1512.03385


### License
---
This project is licensed under the GPL License - see the [LICENSE](https://github.com/PICKTRE/PICKTRE_ai/blob/fb56d697c04e6f1a4d8257985c6813ec5dd6cb84/LICENSE) file for details

#### Licence overview of included 3rd party libraries

tensorflow
- Apache License 2.0
- https://github.com/tensorflow/tensorflow/blob/master/LICENSE

keras
- MIT License
- https://github.com/keras-team/keras/blob/master/LICENSE

numpy
- BSD 3-Clause License
- https://github.com/numpy/numpy/blob/main/LICENSE.txt

pandas
- BSD 3-Clause License
- https://github.com/pandas-dev/pandas/blob/main/LICENSE

sklearn
- BSD 3-Clause License
- https://github.com/scikit-learn/scikit-learn/blob/main/COPYING

PIL
- MIT License
- https://github.com/python-pillow/Pillow/blob/main/LICENSE

