<img src="https://user-images.githubusercontent.com/72016560/179430633-6104dc77-9758-43bf-b1eb-ffca42ba9bdd.png" width="700">

위 그림에서 3위인 텍사스(TX)와 2위인 플로리다(FL)를 비교해보자. 텍사스의 인구는 2.8억명, 플로리다의 인구는 2.1억명으로 텍사스가 플로리다보다 인구가 더 많다. 데이터에서 발견한 이런 편향이 모델의 성능을 해칠 수 있다면, 다시 돌아가서 더 많은 데이터를 수집하거나 오버/언더샘플링하여 정확한 분포를 얻어야하는 것이다.  

### 4.3.5 데이터 슬라이싱으로 편향 확인하기
TFDV를 사용하여 선택한 변수(피처)에서 데이터셋을 슬라이싱하여 편향을 확인할 수 있다.  
(예) 데이터 누락시 편향 발생 → 데이터를 **임의로 누락**시키면 편향 해결

다음 예제에서는 미국의 여러 주 중에서 캘리포니아만을 통계내도록 슬라이싱한다. 

```python
from tensorflow_data_validation.utils import slicing_util
# 피처값은 이진수로 제공해야 한다. 
slice_fn1 = slicing_util.get_feature_value_slicer(
    features={'state': [b'CA']})
slice_options = tfdv.StatsOptions(slice_functions=[slice_fn1])
slice_stats = tfdv.generate_statistics_from_csv(
    data_location='data/consumer_complaints.csv',
    stats_options=slice_options)
```

```python
# 슬라이싱한 통계를 시각화하기 위한 함수 정의
from tensorflow_metadata.proto.v0 import statistics_pb2

def display_slice_keys(stats):
   print(list(map(lambda x: x.name, slice_stats.datasets)))
   
def get_sliced_stats(stats, slice_key):
   for sliced_stats in stats.datasets:
      if sliced_stats.name == slice_key:
         result = statistics_pb2.DatasetFeatureStatisticsList()
         result.datasets.add().CopyFrom(sliced_stats)
         return result
      print('Invalid Slice key')
      
def compare_slices(stats, slice_key1, slice_key2):
  lhs_stats = get_sliced_stats(stats, slice_key1)
  rhs_stats = get_sliced_stats(stats, slice_key2)
  tfdv.visualize_statistics(lhs_stats, rhs_stats)
  
# 시각화
tfdv.visualize_statistics(get_sliced_stats(slice_stats, 'state_CA'))

# 전체와 캘리포니아 통계 비교
compare_slices(slice_stats, 'state_CA', 'All Examples')
```

변수(피처)값으로 캘리포니아 주만을 슬라이싱한 데이터를 시각화한 결과는 다음과 같다.  

<img src="https://user-images.githubusercontent.com/72016560/179643636-917d732d-85fc-4eec-8443-6d38741e76fa.png" width="700">


## 4.4 GCP를 사용한 대용량 데이터셋 처리

데이터가 많을수록 검증 단계에 시간이 더 많이 소요된다. 하지만 클라우드 환경을 활용하면 검증 시간을 단축할 수 있다. (클라우드 서비스를 사용하면, 데이터셋이 클라우드 환경에 적재되기 때문에 노트북이나 사내 리소스의 컴퓨팅 능력에 제한을 받지 않는다.)

TFDV는 GCP 환경에서도 동작이 가능하도록 구현되어있다. 

```python
# 터미널 shell에 설정된 GOOGLE_APPLICATION_CREDENTIALS 환경변수가 있다고 가정

### 파이프라인 객체 설정 (구글 클라우드 옵션 구성)
from apache_beam.options.pipeline_options import (
    PipelineOptions, GoogleCloudOptions, StandardOptions)
# 여기서 pipeline_options는 GCP에서 데이터 검증을 실행할 수 있는 
# GCP 세부정보를 모두 포함하는 객체

options = PipelineOptions()
google_cloud_options = options.view_as(GoogleCloudOptions)
# 프로젝트 식별자 지정
google_cloud_options.project = '<YOUR_GCP_PROJECT_ID>'
# 작업명 지정
google_cloud_options.job_name = '<YOUR_JOB_NAME>'
# 스테이징 및 임시 파일의 저장소 버킷을 가리킵니다. 
google_cloud_options.staging_location = 'gs://<YOUR_GCP_BUCKET>/staging'
google_cloud_options.temp_location = 'gs://<YOUR_GCP_BUCKET>/tmp'
options.view_as(StandardOptions).runner = 'DataflowRunner'

### Dataflow worker 설정
from apache_beam.options.pipeline_options import SetupOptions

setup_options = options.view_as(SetupOptions)
# 최신 TFDV 패키지를 로컬시스템에 다운로드
setup_options.extra_packages = [
    '/path/to/tensorflow_data_validation'
    '-0.22.0-cp37-cp37m-manylinux2010_x86_64.whl'] 

### 로컬 시스템에서 데이터 검증 시작
data_set_path = 'gs://<YOUR_GCP_BUCKET>/train_reviews.tfrecord'
output_path = 'gs://<YOUR_GCP_BUCKET>/' # 데이터 검증 결과를 기록하는 GCP 버킷 지점
tfdv.generate_statistics_from_tfrecord(data_set_path,
                                       output_path=output_path,
                                       pipeline_options=options)
```

GCP 서비스 중 Dataflow라는 서비스를 사용해 데이터 검증을 시작한 후,  
구글 클라우드 Jobs 콘솔로 다시 전환할 수 있다.  

<img src="https://user-images.githubusercontent.com/72016560/180149390-0d279c24-e04b-4d26-8e25-348528a32314.png" width="700">

또한 Dataflow에 TFDV 라이브러리를 설치하여 동일하게 시각화 자료를 추출하거나 통계치를 생성해볼 수 있다.

<img src="https://user-images.githubusercontent.com/72016560/180150193-73dd7360-659d-4e73-8f64-b1bb2b8e941b.png" width="700">


## 4.5 TFDV를 머신러닝 파이프라인에 통합하기

StatisticsGen은 이전 ExampleGen 컴포넌트의 출력을 입력받은 다음 통계를 생성한다. 

```python
# StatisticsGen 함수로 통계 생성
from tfx.components import StatisticsGen

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
context.run(statistics_gen)

context.show(statistics_gen.outputs['statistics'])
```

```python
# SchemaGen 함수로 스키마 생성
from tfx.components import SchemaGen

schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                       infer_feature_shape=True)
context.run(schema_gen)
```

```python
# 통계와 스키마를 사용하여 새로운 데이터셋 검증 - ExampleValidator 함수
# 이 함수는 기본적으로 우리가 앞서 살펴본 skew나 drift를 탐색하고, anomaly를 찾아주겠지만, 
# 내가 지정한 anomaly 조건으로 데이터를 검증하고 싶을 경우 custom component를 만들어 검증할 수 있다.
from tfx.components import ExampleValidator

example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
                                     schema=schema_gen.outputs['schema'])
context.run(example_validator)
```

여기서 이상치가 감지되지 않으면 파이프라인은 다음 단계인 '데이터 전처리 단계'로 넘어간다.  

---
# Chap.5 데이터 전처리

이전에는 데이터의 수집과 수집된 데이터의 검증을 다루었다면,  
이번에는 데이터를 주입하여 모델에 input하기 전까지의 과정을 다룰 것이다. 

## 5.1 데이터 전처리의 필요성

우리가 흔히 수집하는 데이터는 모델이 인식할 수 있는 포맷으로 수집되지 않는다.  
따라서 원활한 학습을 위해서는 모델이 인식할 수 있도록 데이터 포맷을 바꾸는 작업이 필요하다.  
(예를 들어, 모델의 정답지로 사용하는 label 데이터는 'Yes' 혹은 'No'로 수집되지만  
모델이 인식할 수 있게 하려면 1 또는 0로 변경해 주어야 한다.)  
그리고 파이프라인에서 데이터 전처리 과정을 표준화한다면 데이터를 효율적으로 다룰 수 있고, 배포할 수 있으며(with 아파치빔), 잠재적인 학습-서빙 왜곡을 방지할 수 있다.  

### Serving
Serving이란 **학습한 ML 모델을 실제 환경**(Production)**에서 사용할 수 있도록 배포하는 것**을 말한다.  
(Online Serving은 API로 만드는 것을 의미하며, Offline Serving은 배치로 처리하는 것을 의미한다.)

<img src="https://user-images.githubusercontent.com/72016560/180369147-8cb03374-72b7-4ccb-a6e3-8bc73562d84e.png" width="700">

### 학습-서빙 왜곡(training-serving skew)이란?
ML model training outputs와 ML model serving outputs 사이에 차이가 있는 것으로, **Training 성능과 Serving 성능 간의 차이**를 말한다.  

#### 이러한 왜곡이 나타나는 이유 :
 - Train 파이프라인과 Serving 파이프라인에서 데이터를 처리하는 방법의 차이
 - 학습시 데이터와 제공 시 데이터 간의 변화
 - 모델과 알고리즘 간의 피드백 루프

가장 좋은 해결방법은 시스템과 데이터의 변화로 인해 예기치 않은 격차가 생기지 않도록 직접 모니터링하는 것이다.


## 5.2 TFT를 사용한 데이터 전처리
TFT는 Tensorflow Transform의 약자로, TFDV와 같은 TFX 프로젝트의 일부이다.  
TFT는 이전에 생성한 데이터셋 스키마에 따라 수집된 데이터를 처리하고, 아래와 같은 두 종류의 output(아티팩트)을 만든다.

- TFRecord 포맷으로 학습/평가용 데이터셋 두 가지를 생성 (이후 단계인 Trainer 컴포넌트에 사용됨)
- 전처리 그래프 생성(머신러닝 모델을 export할 때 사용)

<img src="https://user-images.githubusercontent.com/72016560/180339111-2f102d37-edf8-4ced-bc5f-7085b558d45a.png" width="700">

그림에서도 볼 수 있듯이 TFT에서 가장 핵심적인 역할을 하는 함수는 `preprocessing_fn()`이다.  
이 함수는 **raw 데이터에 적용할 모든 변환을 정의**한다. (여기서 변환은 모두 텐서플로 작업이어야 함)  
Transform 컴포넌트를 실행하면 `preprocessing_fn()` 함수가 raw 데이터를 수신하고 변환을 적용하며 처리된 데이터를 반환한다.  
(데이터는 변수에 따라 TensorflowTensor 또는 SparseTensor로 제공됨) 

이제 TFT를 직접 사용해보자. 

### 5.2.1 TFT 설치
TFX 패키지를 설치할 때, 종속성으로 TFT도 설치되기 때문에 따로 설치할 필요는 없다.  
하지만 별도 설치를 원한다면 다음 명령어로 설치할 수 있다.  

```
$ pip install tensorflow-transform
```

설치 이후에는 전처리 단계를 통합할 수 있다. 

### 5.2.2 전처리 전략 (주의할 점과 노하우)

#### 1. 데이터 자료형을 고려해야 한다.  
TFT는 `tf.string`, `tf.float32`, `tf.int64` 중 하나로 데이터를 처리하게 되는데, 만약 머신러닝 모델이 위 데이터 타입들을 받아들이지 못한다면 (예를 들어 BERT 모델은 tf.int32 로 input값을 받음) 자료형을 변환하여 머신러닝 모델에게 주어야 할 것이다.
#### 2. 전처리는 배치 단위로 해야 한다.  
전처리 함수를 직접 프로그래밍할 경우, 전처리는 한 번에 한 row만 수행하는 것이 아니라, 배치형으로 수행된다는 것을 인지해야한다. 이로 인해 `preprocessing_fn()` 함수 결과를 Tensor나 SparseTensor로 다시 처리해야될 수도 있다.
#### 3. TF 명령어로 작성해야 한다.  
`preprocessing_fn()` 함수의 내부 함수는 모두 TensorFlow 명령어여야 한다. 즉, Tensorflow의 내장 함수로 수행되어야 한다. (예를 들어, 문자열의 대문자를 소문자로 변경하고 싶을 경우 `lower()`이 아닌 `tf.strings.lower()` 함수를 사용해야 한다.)

### 5.2.3 TFT 함수
다음은 TFT에서 제공하는 대표적인 함수들이다. 

- `tft.scale_to_z_score()`  
   평균이 0, 표준편차가 1인 변수를 정규화할 때 사용한다.
- `tft.bucketize()`  
   변수를 bin으로 버킷화하기 위해 사용한다. bin 또는 bucket index를 반환한다.  
   num_buckets로 버킷 수를 지정하면 TFT가 버킷을 동일한 크기로 나눈다.
- `tft.pca()`  
   주성분 분석을 할 때 사용한다. output_dim 인수로 차원을 설정한다.
- `tft.compute_and_apply_vocabulary()`  
   변수 열의 모든 고유값을 조사한 뒤, 가장 빈번한 값들을 index에 매핑한다. 그런 다음 인덱스 매핑을 사용하여 변수를 숫자로 표현한다. 이 index 매핑값은 feature를 숫자형으로 표현하려고 할 때 사용될 수 있다. 최빈값은 (1)n개의 최상위 고유 항목을 top_k로 정의하거나 (2)각 요소에 frequency_threshold를 사용하여 어휘를 고려하여 검출할 수 있다. 
- `tft.apply_saved_model()`  
   전체 텐서플로 모델을 변수에 적용하는 함수이다. 저장된 모델을 지정 태그와 signature_name으로 로드하면 입력 내용이 모델에 전달된다. 그 다음 모델 실행의 예측이 반환된다. 

#### 텍스트 데이터 처리에 사용되는 함수 (for 자연어 문제)
- `tft.ngrams()`  
   n그램을 생성한다. 문자열 값의 SparseTensor를 입력으로 사용한다. ngram_range 인수로 n그램의 범위를 설정할 수 있고, separator 인수로 조인 문자열이나 문자를 설정할 수 있다. 
- `tft.bag_of_words()`  
   tft.ngram을 사용하고 고유한 각 n그램에 대해 행이 있는 bag-of-words 벡터를 생성한다.  
   (+) **단어가방 (bag-of-words)** : 순서를 고려하지 않고 단어의 집합으로 취급하는 것.  
                            TFIDF가 부여된 단어 집합들을 순서 없이 쓰면 단어가방이 된다.
- `tft.tfidf()`  
   TFIDF(Term Frequency Inverse Document Frequency)는 간단히 말해 '다른 문서에서는 잘 안 나오지만 그 문서에서 많이 나오는 term에 대해 가중치를 주는 방안'이다. Term Frequency은 '1개의 문서 내에서 term의 출현횟수'를 의미하고, Document Frequency는 '전체 문서에서 중복을 제거한 term의 출현횟수(term이 몇 개의 문서에서 나오는 지)를 의미한다. TFIDF는 TF를 DF로 나누는 방식으로 그 문서를 잘 대표하는 term을 골라낸다. 이때 많은 여러 문서에서 공통으로 출현하는 term은 흔하디 흔한 term, 덜 중요한 것이라고 본다. 즉, DF의 값이 큰 term 일수록 특정 문서의 관점에서 볼 때는 가치가 떨어지는 term인 셈이다.  
   [https://euriion.com/?p=411929 참고](https://euriion.com/?p=411929, "https://euriion.com/?p=411929 참고")
   
#### 이미지 데이터 처리에 사용되는 함수 (for 컴퓨터 비전 문제)
TensorFlow는 이미 이미지 전처리 명령어들을 제공하고 있는데, `tf.dll`과 `tf.io` API가 그에 해당한다.  
- `tf.dll`  
   이미지를 자르거나, 크기를 조정하거나, 색 구성표를 변경하거나, 영상을 조정하거나, 이미지를 뒤집거나, 전치하는 등 이미지 변환을 수행하는 변수가 있다. 
- `tf.io`  
   이미지를 모델 그래프의 일부로 여는 데 유용한 변수(예: tf.io.dll_jpeg, tf.io.dlls_png)를 제공한다. 
   
### 5.2.5 TFT 단독으로 실행하기
앞에서는 preprocessing_fn() 함수를 정의하는 방법에 대해 다루었다면,  
이제는 Transform을 어떻게 수행(execute)하느냐에 대해 다룰 것이다.  
전처리 단계만 단독으로 수행할 수도 있고, TFX 컴포넌트 중 하나로 수행해볼 수도 있다.  
양쪽의 경우 모두 Apache Beam이나 Google Cloud의 Dataflow 서비스를 이용하여 수행할 수 있다.  

우선 단독으로 수행하는 경우에 대해 살펴보자.

```python
## preprocessing_fn() 함수 정의
def preprocessing_fn(inputs):
    """입력 열을 반환된 열로 전처리합니다."""
    print(inputs)
    x = inputs['x']
    x_xf = tft.scale_to_0_1(x)
    return {
        'x_xf': x_xf,
    }

## x라는 이름의 컬럼 하나로만 구성된 작은 raw_data 생성
raw_data = [
    {'x': 1.20},
    {'x': 2.99},
    {'x': 100.00}
]

## raw_data를 tf.float32 타입으로 데이터 스키마 정의후 메타데이터 생성
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

raw_data_metadata = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
    }))
    
## 데이터를 TFT를 이용해 변환 후, TFRecord 데이터로 output을 만들어 저장
import tempfile
import tensorflow_transform.beam.impl as tft_beam

with beam.Pipeline() as pipeline: # 파이썬의 컨텍스트 문 with
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()): 
         # 원하는 타이밍에 정확하게 리소스를 할당하고 제공하는 컨텍스트 매니저

        tfrecord_file = "/your/tf_records_file.tfrecord"
        raw_data = (
            pipeline | beam.io.ReadFromTFRecord(tfrecord_file))

        transformed_dataset, transform_fn = (
            (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset( # 데이터셋을 분석 및 변환하는 함수
                preprocessing_fn))
                
## 변환 결과 출력
transformed_data, transformed_metadata = transformed_dataset
pprint.pprint(transformed_data)
```

```python
# 결과: 작게 처리된 데이터셋을 볼 수 있다. 
[
    {'x_xf': 0.0},
    {'x_xf': 0.018117407},
    {'x_xf': 1.0}
]
```

이처럼 preprocessing_fn 함수만 잘 정의해주면 손쉽게 데이터 전처리가 가능하다.

### 5.2.6 TFT를 파이프라인에 통합하기

다음 코드에서는 변수를 정의하고, 
```python
## Feature 예시 및 정의

import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "consumer_disputed"

# Feature 이름, feature 차원
ONE_HOT_FEATURES = {
    "product": 11,
    "sub_product": 45,
    "company_response": 5,
    "state": 60,
    "issue": 90
}

# Feature 이름, bucket count
BUCKET_FEATURES = {
    "zip_code": 10
}

# Feature 이름, 값은 정의되지 않음
TEXT_FEATURES = {
    "consumer_complaint_narrative": None
}

## 변형된 feature를 담을 변수명 정의하기
def transformed_name(key):
    return key + '_xf'

## Sparse(희소)한 변수들은 결측값 채우기
def fill_in_missing(x):
    default_value = '' if x.dtype == tf.string or to_string else 0
    if type(x) == tf.SparseTensor:
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
                            default_value)
    return tf.squeeze(x, axis=1)

## 각 변수별로 전처리할 함수 정의
# one-hot encoding
def convert_num_to_one_hot(label_tensor, num_labels=2):
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])
# bucketize
def convert_zip_code(zip_code):
    if zip_code == '':
        zip_code = "00000"
    zip_code = tf.strings.regex_replace(zip_code, r'X{0,5}', "0")
    zip_code = tf.strings.to_number(zip_code, out_type=tf.float32)
    return zip_code

# preprocessing_fn 정의
## 1. One-hot encoding : 카테고리 이름을 compute_and_apply_vocabulary 함수를 이용하여 인덱스화
##                       카테고리 값을 인덱스화된 카테고리 이름별로 one-hot 인코딩
## 2. Bucketize : zip code 개별 값은 너무 sparse하기 때문에 10개의 bucket(bin)으로 만든 뒤, bucket index는 one-hot encoding
## 3. Text : 문자열은 따로 변환할 필요가 없어 sparse할 경우를 대비하여 missing value 처리한 뒤 dense feature로 가공
def preprocessing_fn(inputs):
    outputs = {}
    for key in ONE_HOT_FEATURES.keys():
        dim = ONE_HOT_FEATURES[key]
        index = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k=dim + 1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            index, num_labels=dim + 1)
    for key, bucket_count in BUCKET_FEATURES.items():
        temp_feature = tft.bucketize(
                convert_zip_code(fill_in_missing(inputs[key])),
                bucket_count,
                always_return_num_quantiles=False)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
                temp_feature,
                num_labels=bucket_count + 1)
    for key in TEXT_FEATURES.keys():
        outputs[transformed_name(key)] = \
            fill_in_missing(inputs[key])
    outputs[transformed_name(LABEL_KEY)] = fill_in_missing(inputs[LABEL_KEY])

    return outputs
```
