# bbang-hyung-4
# 4강 논리회귀와 의사결정 나무
## 목차
- 논리 회귀 실습
- 분류 평가 지표
- 의사결정나무 실습
## 논리회귀
선형 회귀 방식을 이용한 이진 분류 알고리즘이다.(0이냐 1이냐)
### 선형회귀로 풀기 힘든 문제의 등장
시험 전 날 공부한 시간을 가지고 해당 과목의 이수 여부(pass or fail)를 예측하는 문제

- Fail(미이수): 0
- Pass(이수): 1

선형 회귀로 풀었을때, 0.5가 넘으면 통과로 분류해야되는데 떨어지는 것들도 있기 때문에 논리회귀를 씀.

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/ef6de9aa-6d34-4b9a-b7b8-e73e2781ce72)

논리 회귀로 풀었을때,

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/0619ef48-5b55-4d1e-97a1-a018a7cc8eab)
### Sigmoid 함수
![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/08d3537f-0a85-44dd-845d-cbb7d4394793)

x(입력)가 음수 방향으로 갈 수록 y(출력)가 0에 가까워지고,

x(입력)가 양수 방향으로 갈 수록 y(출력)가 1에 가까워진다.

즉, 시그모이드 함수를 통과하면 0 에서 1 사이 값이 나온다.

분류 문제를 풀때 효과적이다.
```
실제 많은 자연, 사회현상에서는 특정 변수에 대한 확률값이 선형이 아닌 S 커브 형태를 따르는 경우가 많다고 합니다. 이러한 S-커브를 함수로 표현해낸 것이 바로 로지스틱 함수(Logistic function)입니다. 딥러닝에서는 시그모이드 함수(Sigmoid function)라고 불립니다.
```

## 논리 회귀 실습
### 유방암 데이터셋을 이용한 실습
#### 데이터셋 로드
- 반지름 radius (mean of distances from center to points on the perimeter)
- 질감 texture (standard deviation of gray-scale values)
- 둘레 perimeter
- 면적 area
- 평탄성 smoothness (local variation in radius lengths)
- 밀도 compactness (perimeter^2 / area - 1.0)
- 오목함 concavity (severity of concave portions of the contour)
- 오목한 점의 개수 concave points (number of concave portions of the contour)
- 대칭 symmetry
- fractal dimension (“coastline approximation” - 1)
- Target: WDBC-Malignant(악성), WDBC-Benign(양성)

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
// pandas와 load_breast_cancer 데이터 셋을 가져온다.

data = load_breast_cancer()
// 데이터 넣어준다.

df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
// 판다스의 데이터프레임 함수로 데이터 셋 시켜준다.

df.head()
// 상위 5개 데이터를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/313fb13d-8a4b-4b0f-918d-fd6a97991684)
![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/de5776e3-1de4-4aac-a37c-5431a3a23e9e)
#### 데이터 시각화
```python
df.describe()
// df의 형태를 묘사한다. (describe : 묘사하는 함수)
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/070a00d5-0ff5-4f10-aad5-c6fa3ed680b7)
![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/8a2894e8-7367-4779-a516-03798cdbb407)

```python
import matplotlib.pyplot as plt
import seaborn as sns
// matplotlib.pyplot과 seaborn 패키지를 가져온다.

sns.countplot(x=df['target'])
// 0과 1의 대한 데이터를 세주는 그래프를 만든다.
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/3dd63d0a-5cdf-4adc-8f60-ff142cd0ddfb)
#### 데이터 전처리
```python
from sklearn.preprocessing import StandardScaler
// StandardScaler 표준화 패키지를 가져온다.

scaler = StandardScaler()
// scaler 변수에 StandardScaler() 객체를 생성한다.

scaled = scaler.fit_transform(df.drop(columns=['target']))
// scaler 데이터에서 taget 값은 제거하고 표준화 시켜준다.

scaled[0]
// scaled의 0번째 데이터를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/2b6cbe3d-5c2d-4d4f-90a3-6dfb4225ce78)

#### 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(scaled, df['target'], random_state=2020)
// scaled와 taget값으로 분할 시켜준다.

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
// 분할된 데이터의 형태를 알아본다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/de1439aa-7b22-4584-90aa-88062624ef41)

#### 모델 정의, 학습, 검증
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
// 논리회귀와 정확도 패키지를 가져온다.

model = LogisticRegression()
// 모델정의

model.fit(x_train, y_train)
// 모델 훈련

y_pred = model.predict(x_val)
// 정답값 예측

accuracy_score(y_val, y_pred)
// 정답값과 예측값 비교
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/c0bea902-d598-45a0-ab74-82f3ce3a0e13)
## 분류 평가 지표
내가 만든 분류기가 얼마나 잘 작동하는지 알아보자
### Confusion Matrix
![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/2b57726d-e30e-4958-8d05-9bd31f7b9c75)

모델이 Positive/Negative 예측했고, 실제로 True/False 인 경우

1. True Positive (TP): 모델이 임신했다고 예측했고, 실제로 임신한 경우 (참)
2. True Negative (TN): 모델이 임신하지 않았다고 예측했고, 실제로 임신하지 않은 경우 (참)
3. False Positive (FP): 모델이 임신했다고 예측했지만, 실제로 임신하지 않은 경우 (거짓)
4. False Negative (FN): 모델이 임신하지 않았다고 예측했지만, 실제로 임신한 경우 (거짓)

```python
from sklearn.metrics import confusion_matrix
// confusion_matrix 패키지를 가져온다.

cm = confusion_matrix(y_val, y_pred)
// 예측한 값과 정답 값을 표로 나타낸다.

cm
```
![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/0fca2755-6bce-4537-a77e-ec86cf8bace3)

```python
sns.heatmap(cm, annot=True, cmap='PiYG')
// 악성과 약성에 대한 히트맵을 그린다.

plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
// x축은 예측한 값이고 y는 정답 값이다.

plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/546c6cdd-f711-4459-bd2c-6aaf8750810c)

```python
TN, FP, FN, TP = cm.flatten()
// 2차원이었던 데이터값을 1차원으로 바꿔준다.

print(TP, TN, FP, FN)
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/a7151265-7188-4d49-86fe-74b4c1a7db49)

### 정확도 Accuracy
- 얼마나 정답을 잘 예측했는가
- Accruacy = (TP + TN) / Total

```python
(TP + TN) / (TP + TN + FP + FN)
// 정확도는 모든 값을 더하고 맞은 값으로 나누면 된다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/56262c2c-ce9f-4abb-885f-7d6e782afb01)

```python
from sklearn.metrics import accuracy_score
// accuracy_score 패키지를 가져온다.

accuracy_score(y_val, y_pred)
// 정답 값과 예측 값을 비교하여 정확도를 알아낸다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/e1bb8bf6-8451-48b4-9f8b-2e30e85e5e3a)
### 오류율 Error Rate
- 얼마나 틀렸는가
- ER = (FP + FN) / Total

```python
(FP + FN) / (TP + TN + FP + FN)
// 오류율은 모든 값을 더하고 틀린 값으로 나누면 된다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/8b4687a6-4b4d-4241-87e9-ea1b11924884)
### 정밀도 Precision
모델이 임신했다고 예측한 것 중에서, 진짜 임신한 경우는 얼마나 되는가

모델이 암이라고 예측한 것 중에서, 진짜 암인 경우는 얼마나 되는가

모델이 스팸 메일이라고 분류한 것 중에서, 진짜 스팸 메일인 경우는 얼마나 되는가
- 중요: 스팸 메일로 분류했는데 중요 메일인 경우

Precision = TP / Predicted yes

```python
TP / (TP + FP)
// Positive 값들을 다 더하고 맞은 Positive 값을 나눈다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/1d462da7-5add-4761-b52b-a2ed679beb08)

```python
from sklearn.metrics import precision_score
// precision_score 패키지를 사용하여 정밀도를 확인할 수 있다.

precision_score(y_val, y_pred)
// 정답 값과 예측 값을 비교하여 정밀도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/7c454770-6223-4dc4-8109-aee90fd73d16)
### 재현율 Recall
실제 임신했는데, 모델이 임신이라고 예측한 경우는 얼마나 되는가

실제 암인데, 모델이 암이라고 예측한 경우는 얼마나 되는가

- 중요: 암에 걸렸는데 암이 아니라고 예측해버렸을 경우
실제 비가 오는데, 모델이 비가 온다고 예측한 경우는 얼마나 되는가

- 중요: 비가 오는데 비가 안온다고 예측해버렸을 경우 
Recall = TP / Actual yes

```python
TP / (FN + TP)
// 맞는 Positive와 틀린 Negative 값을 더하고 맞은 Positive 값으로 나눈다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/3b34e68b-3e4e-4f02-8fbb-9ff977189c04)

```python
from sklearn.metrics import recall_score
// recall_score 패키지를 가져온다.

recall_score(y_val, y_pred)
// 정답값과 예측값을 비교하여 재현율을 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/0cad149d-ffc1-4d78-92ef-dd4ec5c58720)
### F1 Score
- 정밀도와 재현율의 균형도 (가중 평균)
- F1 = 2 * (precision * recall) / (precision + recall)
- 정밀도와 재현율을 잘 표현했다고 보면된다.

```python
from sklearn.metrics import f1_score
// f1_score를 가져온다

f1_score(y_val, y_pred)
// 정답 값과 예측 값을 비교하여 f1 score를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/e672edb1-8206-470f-b49e-2a57898565ec)

## 의사 결정 나무
Decision Tree 스무 고개

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/300623cd-27c7-4a76-80d6-926bee0ddc21)
### 유방암 데이터셋을 이용한 실습
#### 모델 정의, 학습, 검증
```python
from sklearn.tree import DecisionTreeClassifier
// DecisionTreeClassifier 모델을 가져온다.

model = DecisionTreeClassifier()
// 모델정의

model.fit(x_train, y_train)
// 모델 훈련

y_pred = model.predict(x_val)
// 정답값 예측

accuracy_score(y_val, y_pred)S
// 정답값과 예측값을 비교하여 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/c58906ab-ccc8-4b9e-824c-68f91ebc4443)
#### 의사결정나무 시각화
```python
!pip install -q dtreeviz
// dtreeviz 패키지를 설치한다.
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/28614b64-ffd6-46e7-8111-1770acfcc924)

```python
from dtreeviz.trees import dtreeviz
// dtreeviz 패키지를 가져온다.

viz = dtreeviz(model, 
  x_train,
  y_train,
  target_name='target',
  feature_names=data['feature_names'],
  class_names=['Malignant', 'Benign'])
// 모델과 train데이터, 타겟, 퓨쳐네임, 악성과 양성을 넣어서 dtreeviz를 만든다.
viz
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/3e5edbe9-90d2-4498-ab7d-e44224b5b8da)

#### Feature Importance
의사결정나무가 학습할 때 중요하게 생각한 특징들을 중요도 순으로 나열할 수 있습니다.

```python
features = pd.DataFrame(
    model.feature_importances_,
    index=data['feature_names'],
    columns=['importance']
).sort_values('importance', ascending=False)
// 악성과 약성을 분류하는데 가장 중요한것들을 중요한 순서대로 데이터 프레임으로 나타낸다.

features
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/7a84c4bd-c317-4e20-9911-fbe3a39d95f1)

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/eb27ecc9-c4eb-4783-af38-791ce8266a68)

```python
features.plot.bar(figsize=(16, 6))
// 퓨쳐의 중요도를 가로16, 세로 6인 막대차트를 통해 알 수 있다.
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung-4/assets/104752580/4031a9b4-d31b-435b-8acf-b2cbf30aa56f)
