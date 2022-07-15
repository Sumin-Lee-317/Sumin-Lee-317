
#### 사용자 지정 데이터를 TFRecord 데이터 구조로 변환하기

```python
### 1. 데이터 다운로드 및 전처리
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# 웹 상에서 원시 데이터를 data 폴더로 다운받는다.
filepath = tf.keras.utils.get_file(
    "complaints.csv.zip",
    "http://files.consumerfinance.gov/ccdb/complaints.csv.zip")

dir_path = Path(__file__).parent.absolute()
data_dir = os.path.join(dir_path, "..", "..", "data")
processed_dir = os.path.join(dir_path, "..", "..", "data", "processed")
Path(processed_dir).mkdir(parents=True, exist_ok=True)

# 압축을 해제한다.
shutil.unpack_archive(filepath, data_dir)
# pandas로 csv 파일을 읽어온다.
df = pd.read_csv(os.path.join(data_dir, "complaints.csv"))

# df의 필드명(열이름)을 변경한다.
df.columns = [
    "date_received", "product", "sub_product", "issue", "sub_issue",
    "consumer_complaint_narrative", "company_public_response",
    "company", "state", "zip_code", "tags",
    "consumer_consent_provided", "submitted_via",
    "date_sent_to_company", "company_response",
    "timely_response", "consumer_disputed", "complaint_id"]

# consumer_disputed 필드값(row)이 공백인 경우, 그 자리에 NaN을 넣는다. 
df.loc[df["consumer_disputed"] == "", "consumer_disputed"] = np.nan

# 주요한 필드가 비어있는 경우(NA) 해당 레코드를 삭제한다.
df = df.dropna(subset=["consumer_complaint_narrative", "consumer_disputed"])

# Label 필드인 consumer_disputed를 Yes, No에서 1, 0 으로 변경한다.
df.loc[df["consumer_disputed"] == "Yes", "consumer_disputed"] = 1
df.loc[df["consumer_disputed"] == "No", "consumer_disputed"] = 0

# zip_code 필드값(row)이 공백이거나 NA인 경우, 그 자리에 000000을 넣는다.
df.loc[df["zip_code"] == "", "zip_code"] = "000000"
df.loc[pd.isna(df["zip_code"]), "zip_code"] = "000000"

# zip_code가 5글자인 레코드만 뽑아 df로 다시 정의한다.
df = df[df['zip_code'].str.len() == 5]
# zip_code 필드값(row)에 있는 XX를 00으로 바꾼다. 
df["zip_code"] = df['zip_code'].str.replace('XX', '00')
# df의 인덱스를 리셋한다. 
df = df.reset_index(drop=True)
# zip_code 자료형을 숫자로 변경한다. (에러 발생시 결측값으로 처리)
df["zip_code"] = pd.to_numeric(df["zip_code"], errors='coerce')

# 판다스 DataFrame을 csv 파일로 다시 저장한다.
df.to_csv(os.path.join(processed_dir, "processed-complaints.csv"), index=False)
```

이제 전처리한 데이터를 TFRecord 데이터 구조로 변환할 것이다.  
하지만 그 전에, 데이터 레코드를 올바른 데이터 type으로 변환하는데 도움이 되도록 함수를 정의하자. 
```python
import os
import re

import tensorflow as tf
import pandas as pd
from pathlib import Path

### 2. 데이터 type별 feature 생성 코드 (함수)

# string, byte는 BytesList로 리턴
def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode()])
    )

# float, double은 FloatList로 리턴
def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])
    )

# bool, enum, int, uint는 Int64List로 리턴
def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])
    )
    
### 그외에 추가할 코드
import re
import pandas as pd

def clean_rows(row):
    if pd.isna(row["zip_code"]):
        row["zip_code"] = "99999"
    return row

def convert_zipcode_to_int(zipcode):
    nums = re.findall(r'\d+', zipcode)
    if len(nums) > 0:
        int_zipcode = int(nums[0])
    else:
        int_zipcode = 99999
    return int_zipcode


### 3. 데이터를 TFRecord 데이터 구조로 변환

dir_path = Path(__file__).parent.absolute() # 현재 파일의 전체 경로
data_dir = os.path.join(dir_path, "..", "..", "data") 
tfrecord_dir = os.path.join(dir_path, "..", "..", "data", "tfrecord")
df = pd.read_csv(os.path.join(data_dir, "processed-complaints.csv"))

tfrecord_filename = "consumer-complaints.tfrecord"
tfrecord_filepath = os.path.join(tfrecord_dir, tfrecord_filename)
# tfrecord_filename에 지정된 경로에 저장하는 TFRecordWriter 객체 생성
tf_record_writer = tf.io.TFRecordWriter(tfrecord_filepath)

for index, row in df.iterrows():
    row = clean_rows(row)
    # 모든 데이터 레코드를 tf.train.Example로 변환
    example = tf.train.Example(     # feature 이용해 Example 객체 생성
        features=tf.train.Features(
            feature={    # 기록할 데이터 레코드들을 딕셔너리 형태로 묶어 feature 객체 생성
                "product": _bytes_feature(str(row["product"])),
                "sub_product": _bytes_feature(str(row["sub_product"])),
                "issue": _bytes_feature(str(row["issue"])),
                "sub_issue": _bytes_feature(str(row["sub_issue"])),
                "state": _bytes_feature(str(row["state"])),
                "zip_code": _int64_feature(convert_zipcode_to_int(row["zip_code"])),
                "company": _bytes_feature(str(row["company"])),
                "company_response": _bytes_feature(str(row["company_response"])),
                "timely_response": _bytes_feature(str(row["timely_response"])),
                "consumer_disputed": _float_feature(row["consumer_disputed"]),
            } # 여기서 str()는 문자열 변환 함수
        )
    )
    # 데이터 구조를 직렬화
    tf_record_writer.write(example.SerializeToString())
    # 위에서 생성한 example 객체를 tf.io.TFRecordWriter를 사용해 consumer-complaints.tfrecord에 써준다. 
tf_record_writer.close()
```

이제 이렇게 생성된 TFRecord 파일 consumer-complaints.tfrecord를 ImportExampleGen 구성요소로 가져올 수 있다.


### 3.1.2 원격 데이터 파일 수집
ExampleGen 컴포넌트는 원격 클라우드 저장소 버킷(구글 클라우드 스토리지, AWS S3 등)에서 파일을 읽을 수 있다.  
TFX 사용자는 다음과 같이 external_input 함수에 대한 버킷 경로를 제공할 수 있다.   
(* 버킷 bucket: 객체가 파일이라면, 버킷은 연관된 객체들을 그룹핑한 최상위 디렉터리라고 할 수 있다.)
```python 
from tfx.components import CsvExampleGen
example_gen = CsvExampleGen(input_base="gs://example_compliance_data/")
```

### 3.1.3 데이터베이스에서 직접 데이터 수집
DB에서 직접 데이터를 수집할 때에는 2가지 컴포넌트를 사용할 수 있다.

#### 1. BigQueryExampleGen 
구글 클라우드 빅쿼리 테이블에서 데이터를 수집하는 컴포넌트이다.  
GCP 생태계에서 머신러닝 파이프라인을 실행할 때 정형 데이터를 매우 효율적으로 수집하는 방법이다. 

다음은 빅쿼리 테이블을 쿼리하는 가장 간단한 방법이다. 

```python
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen

query = """
    SELECT * FROM `<project_id>.<database>.<table_name>`
"""
example_gen = BigQueryExampleGen(query=query)
```

#### 2. PrestoExampleGen
프레스토 데이터베이스에서 데이터를 수집하는 컴포넌트이다.  
이 컴포넌트를 사용하려면 DB의 연결 세부 정보를 지정하는 추가 구성이 필요하다.

```
 $ pip install presto-python-client
```

```python
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

query = """
    SELECT * FROM `<project_id>.<database>.<table_name>`
"""
presto_config = presto_config_pb2.PrestoConnConfig(
    host='localhost',
    port=8080)
example_gen = PrestoExampleGen(presto_config, query=query)
```


## 3.2 데이터 준비
각 ExampleGen 컴포넌트를 사용하여 데이터셋의 입력 설정(input_config)과 출력 설정(output_config)을 구성할 수 있다. 

### 3.2.1 데이터셋 분할
파이프라인 후반부에는 학습 중에 머신러닝 모델을 평가하고 모델 분석 단게에서 테스트하려고 한다.  
따라서 데이터셋을 필요한 하위 집합으로 분할해두면 좋다. 

#### 1. 단일 데이터셋을 하위 집합으로 분할
다음은 학습, 평가, 테스트 데이터셋을 각각 6:2:2 비율로 데이터셋을 분할하는 코드이다.  
비율은 hash_buckets 파라미터로 정의한다.  
(+) 출력 구성을 지정하지 않으면 ExampleGen 컴포넌트는 학습 및 평가 데이터셋(2:1 비율)으로 분할한다.  

```python
import os
from pathlib import Path

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2

context = InteractiveContext()

dir_path = Path().parent.absolute()
data_dir = os.path.join(dir_path, "..", "..", "data", "processed")
output = example_gen_pb2.Output(
    # 선호하는 분할을 정의한다.
    split_config=example_gen_pb2.SplitConfig(splits=[
        # 비율을 지정합니다. 
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=6),
        example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
        example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=2)
    ]))

# output_config 인수를 추가한다.
example_gen = CsvExampleGen(input_base=data_dir, output_config=output)
context.run(example_gen)

# example_gen 객체 실행 후 아티팩트 목록을 출력하여 생성된 아티팩트를 검사한다. 
for artifact in example_gen.outputs['examples'].get():
    print(artifact)
```

#### 2. 기존 분할 보존
입력 구성을 정의하여 기존 분할을 그대로 가져올 수 있다.  

다음 구성에서는 데이터셋을 외부에서 분할해 하위 디렉터리에 저장했다고 가정하자.

```
┗ data
   ┣ train
   ┃   ┗ 20k-consumer-complaints-training.csv
   ┣ eval
   ┃   ┗ 4k-consumer-complaints-eval.csv
   ┗ test
       ┗ 2k-consumer-complaints-test.csv
```

이 입력 구성을 정의하여 기존 입력 분할을 유지할 수 있다. 

```python
import os
from pathlib import Path

from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

context = InteractiveContext()

dir_path = Path().parent.absolute()
data_dir = os.path.join(dir_path, "..", "..", "data", "tfrecord")

tfrecord_filename = "consumer-complaints.tfrecord"
tfrecord_filepath = os.path.join(data_dir, tfrecord_filename)

# 기존 하위 디렉터리를 설정한다.
input = example_gen_pb2.Input(splits=[
    example_gen_pb2.Input.Split(name='train', pattern='train/*'),
    example_gen_pb2.Input.Split(name='eval', pattern='eval/*'),
    example_gen_pb2.Input.Split(name='test', pattern='test/*')
])

# input_config 인수를 정의 후, ExampleGen 컴포넌트에 설정을 전달한다. 
example_gen = CsvExampleGen(input_base=data_dir, input_config=input)
context.run(example_gen)
```

### 3.2.2 데이터셋 스패닝

#### ☆ 스팬(span) 
- 데이터의 점진적인 수집에 사용되는 것
- 데이터의 스냅샷으로 간주 (기존 데이터 레코드를 복제 가능)  
  밑의 그림에서, export-1에는 이전 export-0의 데이터와 export-0 이후 새로 생성된 레코드가 포함된다. 

- ExampleGen 컴포넌트를 사용하여 스팬을 사용 가능
- 시, 일, 주마다 배치 추출, 변환, 로드 (ETL) 프로세스가 이런 데이터 스냅샷을 만들고 새 스팬을 생성 가능

```
┗ data
   ┣ export-0
   ┃   ┗ 20k-consumer-complaints.csv
   ┣ export-1
   ┃   ┗ 24k-consumer-complaints.csv
   ┗ export-2
       ┗ 26k-consumer-complaints.csv
```

이제 스팬의 패턴을 지정해보자.

```python
import os

from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

context = InteractiveContext()

base_dir = os.getcwd()
data_dir = os.path.join(os.pardir, "data")

input = example_gen_pb2.Input(splits=[
    example_gen_pb2.Input.Split(pattern='export-{SPAN}/*') 
    # 여기서 {SPAN} 자리표시자는 폴더 구조에 표시된 숫자(0,1,2)를 나타냄
])

# 입력 구성을 사용해 ExampleGen 컴포넌트가 '최신' 스팬을 선택한다. (여기서는 export-2) 
example_gen = CsvExampleGen(input_base=os.path.join(base_dir, data_dir), input_config=input)
context.run(example_gen)
```

이미 분할된 데이터라면 입력 정의에서 하위 디렉터리를 정의할 수도 있다.  

```python
input = example_gen_pb2.Input(spilts=[
    example_gen_pb2.Input.Split(name='train',
                                pattern='export-{SPAN}/train/*'),
    example_gen_pb2.Input.Split(name='eval',
                                pattern='export-{SPAN}/eval/*')
])
```

### 3.2.3 데이터셋 버전 관리
머신러닝 파이프라인에서 우리는 머신러닝 모델을 학습하는 데 사용한 데이터셋과 함께 생산된 모델을 추적하려고 한다.  
이때 데이터셋을 버전화하면, 수집한 데이터를 더 자세히 추적할 수 있다.  
이런 버전 추적을 통해 학습 중에 사용한 데이터셋이 학습 이후 시점의 데이터셋과 동일한 지 확인할 수 있다.  
(이런 피처는 엔드투엔드 ML 재현성에 매우 중요함)

그러나 TFX ExampleGen 컴포넌트에서는 데이터 버전화를 지원하지 않는다.  
그래서 다음과 같은 도구를 사용할 수 있다. 

#### 데이터 버전 제어 (DVC)
DVC(https://dvc.org/)는 머신러닝 프로젝트용 오픈소스 버전 제어 시스템이다. 전체 데이터셋 자체 대신 데이터셋 해시를 커밋할 수 있다. DVC는 머신러닝 프로젝트를 위한 오픈소스 버전 제어 시스템이다. 전체 데이터셋 자체 대신 데이터셋의 해시를 커밋할 수 있다. 따라서 데이터셋의 상태는 git을 통해 추적되지만 리포지토리는 전체 데이터셋 단위로 적재되진 않는다. 

#### 파키덤 (Pachyderm)
파키덤(https://www.pachyderm.com/)은 쿠버네티스에서 운영하는 오픈소스 머신러닝 플랫폼이다. 이는 데이터 버전 관리(데이터용 깃)라는 개념에서 시작했지만, 이제는 데이터 버전을 기반으로 하는 파이프라인 조정을 포함한 전체 데이터 플랫폼으로 확장되었다. 


## 3.3 수집 전략

### 3.3.1 정형 데이터 (표 형식 데이터 지원, DB나 디스크에 파일 형식으로 저장)
DB에 있는 데이터를 CSV로 내보내거나 PrestoExampleGen 또는 BigQueryExampleGen 컴포넌트로 데이터 직접 사용할 수 있다.  

표 형식 데이터를 지원하는 파일 형식으로 디스크에 저장된 데이터는 CSV로 변환하고 CsvExampleGen 컴포넌트를 사용하여 파이프라인으로 수집해야 한다.  
데이터양이 수백 메가바이트 이상으로 증가할 때는 데이터를 TFRecord 파일로 변환하거나 아파치 파케이로 저장해야 한다. 

### 3.3.2 텍스트 데이터 (for 자연어 처리)
텍스트 말뭉치는 눈덩이처럼 상당한 크리로 불어날 수 있다.  
따라서 데이터셋을 TFRecord나 아파치 파케이 포맷으로 변환하는 것이 좋다. 
또한 DB에서 말뭉치를 수집할 수도 있는데, 이때는 네트워크 트래픽 비용과 병목 현상을 고려해야 한다. 

### 3.3.3 이미지 데이터 (for 컴퓨터 비전)
이미지 데이터셋을 TFRecord 파일로 변환하는 것은 좋지만, 이미지 디코딩은 권장하지 않는다.  
고도로 압축된 이미지를 디코딩하면 중간 tf.Example 레코드를 저장하는 데 필요한 디스크 공간만 증가한다.  
압축한 이미지는 tf.Example 레코드에 바이트 문자열로 저장할 수 있다. 

```python
import os
import tensorflow as tf

base_path = "/path/to/images"
filenames = os.listdir(base_path)

def generate_label_from_path(image_path):
   pass
   # ...
   # return label

def _bytes_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
   
def _int64_feature(value):
   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
   
tfrecord_filename = 'data/image_dataset.tfrecord'

with tf.io.TFRecordWriter(tfrecord_filename) as writer:
   for img_path in filenames:
   img_path = os.path.join(base_path, img_path)
   try:
      raw_file = tf.io.read_file(image_path)
   except FileNotFoundError:
      print("File {} could not be found".format(image_path))
      continue
   example = tf.train.Example(features=tf.train.Features(feature={
      'image_raw': _bytes_feature(raw_file.numpy()),
      'label': _int64_feature(generate_label_from_path(image_path))
   }))
   writer.write(example.SerializeToString())
```
제공된 경로 /path/to/images에서 이미지를 읽어 tf.Example에 바이트 문자열로 저장한다.  
파이프라인의 이 시점에서는 이미지를 전처리하지 않겠다. (잠재적인 학습-서빙 왜곡이 발생할 수 있음)

tf.Examples에 레이블과 함께 원시 이미지를 저장한다. 여기서는 파일 이름에서 generate_label_from_path 함수를 사용하여 각 이미지의 레이블을 추출한다. 레이블 생성은 데이터셋에 따라 다르므로 이 예제에는 포함하지 않았다. 

이미지를 TFRecord 파일로 변환한 후에는 ImportExampleGen 컴포넌트를 사용하여 데이터셋을 효율적을 소비하고, '기존 TFRecord 파일 가져오기'에서 살펴본 전략을 적용할 수 있다. 



---
# Chap.4 데이터 검증

3장에서는 다양한 소스에서 파이프라인으로 데이터를 수집하는 방법을 알아보았다.  
이번 4장에서는 수집한 데이터를 검증하는 방법에 대해서 다룰 것이다.  
먼저 데이터 검증의 필요성에 대해 다루고, TFX에서 제공하는 파이썬 패키지인 TFDV를 소개한다. 

## 4.1 데이터 검증의 필요성

머신러닝에서는 데이터셋의 패턴에서 학습하고 이를 일반화하려고 한다.  
따라서 데이터는 머신러닝 워크플로에서 가장 중요하며, 데이터의 퓸질은 머신러닝 프로젝트 성공의 핵심 요소이다.  

파이프라인의 데이터 검증 단계는 다음과 같은 역할을 한다.  

- 검증(검사) 수행 
  + 데이터 이상치 확인
  + 스키마 비교를 통해 데이터 구조가 변했는지 확인 
  + 새 데이터셋의 통게와 이전 학습 데이터셋의 통계가 일치하는지 확인 (데이터 드리프트 발생 여부 확인)  
  (+) 데이터 드리프트: 새 피처를 선택하거나 데이터 전처리 단계를 업데이트해야 함을 의미
- 모든 오류 강조 표시
  + 데이터 피처에 관한 통계를 생성하며,  
   피처에 높은 비율의 결측값이 포함되는지 혹은 피처와 높은 상관관계가 있는지를 강조 표시

오류를 감지하면 워크플로를 중지하고 데이터 문제를 직접 해결 가능하다.  


## 4.2 TFDV - 텐서플로 데이터 검증 도구
- TFRecord와 CSV 파일 형식을 허용하여 데이터 검증 시작
- 아파치빔을 통해 분석을 배포
- 구글 PAIR 프로젝트 패싯에 기반한 시각화 제공 (스크린샷 참고)
<img src="https://user-images.githubusercontent.com/72016560/178866721-93ec0993-d83a-4999-b926-d4830af5c172.png" width="600">

### 4.2.1 설치
TFX를 설치하면 TFDV도 같이 설치된다.  
TFDV만 사용하려면 다음 명령을 사용하자.  
```
$ pip install tensorflow-data-validation
```
tfx나 tensorflow-data-validation를 설치하면, 데이터 검증을 워크플로에 통합하거나 주피터 노트북에서 시각적으로 데이터를 분석할 수 있다. 

### 4.2.2 데이터에서 통계 생성
데이터 검증 프로세스의 첫 번째 단계는 데이터 요약 통계를 생성하는 것이다. 
```python
import tensorflow_data_validation as tfdv

# CSV 파일에서 피처에 대한 통계 생성
stats = tfdv.generate_statistics_from_csv(
     data_location='/data/comsumer_complaints.csv',
     delimiter',')

# TFRecord 파일에서 피처 통계 생성
stats = tfdv.generate_statistics_from_tfrecord(
    data_location='/data/consumer_complaints.tfrecord')
```

위에서 본 2가지 TFDV 방법은 모두 최솟값, 최댓값, 평균값을 포함하여 각 피처에 관한 요약 통계를 저장하는 데이터 구조를 생성한다.  
그 데이터 구조는 다음과 같다. 
```
datasets {
   num_examples: 66799
   features {
      type: STRING
      string_stats {
         common_stats {
         num_non_missing: 66799
         min_num_values: 1
         max_num_values: 1
         avg_num_values: 1.0
         num_values_histogram {
            buckets {
               low_value: 1.0
               high_value: 1.0
               sample_count: 6679.9
...
}}}}}}
```

TFDV는 각 피처에 대해 다음을 계산한다.  
숫자형 피처 | 범주형 피처
:--|:--
전체 데이터 레코드 개수 | 전체 데이터 레코드 개수
누락 데이터 레코드 개수 | 누락 데이터 레코드의 백분율
데이터 레코드 전체에서 피처의 평균과 표준편차 | 고유 레코드 개수
데이터 레코드에서 피처의 최솟값과 최댓값 | 피처의 모든 레코드의 평균 문자열 길이
데이터 레코드에서 피처의 0값의 비율 | 각 범주에서 각 레이블의 샘플 수와 순위를 결정
(+) 각 피처의 히스토그램 생성 | ..

### 4.2.3 데이터에서 스키마 생성
요약 통계를 생성한 다음에는 데이터셋 스키마를 생성해야 한다.  
스키마는 데이터셋에 필요한 피처와 각 피처의 기반이 되는 데이터 타입과 데이터 범위(피처에 허용된 최댓값/최솟값, 누락 레코드의 임곗값 개요)를 정의한다.  
이렇게 정의를 하고나면, 데이터셋의 스키마 정의를 사용하여 향후 데이터셋이 이전 학습 데이터셋과 일치하는지 확인할 수 있다.  

다음과 같이 단일 함수 호출로 생성한 통계에서 스키마 정보를 생성할 수 있다.  

```python
schema = tfdv.infer_schema(stats)  # 피처통계파일로 스키마 프로토콜 버퍼 생성

tfdv.display_schema(schema)  # 스키마 표시
```


다음은 피처들중 "product"에 대한 구조를 나타낸다. 
```
feature {
   name: "product"
   type: BYTES
   domain: "product"
   presence [
      min_fraction: 1.0
      min_count: 1
      
   }
   shape {
     dim {
        size: 1
     }
   }
}
```
스키마 표시 결과는 다음과 같다. 

<img src="https://user-images.githubusercontent.com/72016560/178867538-c94f082f-3fe2-4159-a1ed-7abe0b05e930.png" width="600">

- Type: 데이터 형식
- Presence: 피처가 데이터에 꼭 있어야(required) 하는지, 선택사항(optional)인지
- Valency: 학습 데이터당 필요한 값의 수 (single이면, 학습 예제마다 해당 피처에 정확히 하나의 범주가 있어야한다는 뜻)


## 4.3 데이터 인식
이전 절에서는 데이터 요약 통계와 스키마를 생성하는 방법을 배웠다.  
이는 설명해주지만, 잠재적인 문제를 발견하지는 못한다.  
여기서는 TFDV가 데이터에서 문제를 발견하는데 어떻게 도움이 되는지 살펴보겠다.  

### 4.3.1 데이터셋 비교

#### 1. 학습 데이터셋과 검증 데이터셋 비교 (데이터셋 로드 후 시각화)

```python
train_stats = tfdv.generate_statistics_from_tfrecord(
    data_location = train_tfrecord_filename)
val_stats = tfdv.generate_statistics_from_tfrecord(
    data_location = val_tfrecord_filename)

tfdv.visualize_statistics(lhs_statistics=val_stats, rhs_statistics=train_stats,
                          lhs_name='VAL_DATASET', rhs_name='TRAIN_DATASET')
```

결과는 다음과 같다. 
<img src="https://user-images.githubusercontent.com/72016560/178927062-8f9a2630-8975-4dd2-93d9-1232d58dfada.png" width="600">

예를 들어, 검증 데이터셋(레코드 4,998개)의 누락(missing)된 sub_issue 값 비율이 더 낮다.  
이는 피처가 검증 집합에서 분포를 변경하고 있음을 의미할 수 있다.  
그리고 시각화에서 모든 레코드의 절반 이상이 sub_issue 정보를 포함하고 있지 않음을 강조하고 있다.  
sub_issue가 모델 학습에 중요한 피처라면, 데이터 캡처방법을 수정해서 올바른 문제 식별자로 새 데이터를 수집하도록 해야한다. 

#### 2. 이상치 탐지

```python
anomalies = tfdv.validate_statistics(statistics=val_stats, schema=schema)

tfdv.display_anomalies(anomalies)
```

주피터 노트북에서 시각화된 이상치 보고서 결과는 다음과 같다.  

<img src="https://user-images.githubusercontent.com/72016560/178931515-0e7f8c09-03b8-4cbc-a5d7-4456e2cd4417.png" width="600">

다음 코드는 default로 설정된 이상치 프로토콜을 보여준다. 여기에는 머신러닝 워크플로를 자동화하는 데 유용한 정보가 포함된다. 

```
anomaly_info {
   key: "company"
   value: {
      description: "The feature was present in fewer examples than expected."
      severity: ERROR
      short_description: "Column dropped"
      reason {
         type: FEATURE_TYPE_LOW_FRACTION_PRESENT
         short_description: "Column dropped"
         description: "The feature was present in fewer examples than expected."
      }
      path {
         step: "company"
      }
   }
}
```
이런 식으로 이상치를 데이터셋에 적합하도록 조정할 수 있다. 

### 4.3.2 스키마 업데이트
데이터에 관한 도메인 정보에 따라 스키마를 수동으로 설정할 수 있다. 

앞에서 설명한 sub_issue 피처를 사용하여 이 피처가 학습 예제의 90% 이상에 포함되도록 요구해야 한다고 판단되면 스키마를 업데이트하여 이를 반영할 수 있다. 

#### 
```python 
# 스키마를 직렬화된 위치에서 로드
schema = tfdv.load_schema_text(schema_location)

# 이 특정 피처의 min_fraction 값을 90%로 설정
sub_issue_feature = tfdv.get_feature(schema, 'sub_issue')
sub_issue_feature.presence.min_fraction = 0.9

# 미국 주 목록을 업데이트하여 알래스카(AK)를 제거
state_domain = tfdv.get_domain(schema, 'state')
state_domain.value.remove('AK')

# 스키마가 검증되면 다음과 같이 스키마 파일을 직렬화하여 다음 위치에 생성
tfdv.write_schema_text(schema, schema_location)

# 통계를 다시 확인하여 업데이트된 이상치 확인
updated_anomalies = tfdv.validate_statistics(eval_stats, schema)
tfdv.display_anomalies(updated_anomalies)
```

### 4.3.3 데이터 스큐 및 드리프트
TFDV는 두 데이터셋의 통계 간의 큰 차이를 감지하는 내장 '스큐 비교기'를 제공한다.  
이는 스큐(평균 주위에 비대칭적으로 분포된 데이터셋)의 통계적 정의가 아니다.  
TFDV에서는 두 데이터셋의 service_statistics 간의 차이에 대한 L-infinity Norm으로 정의된다.  
두 데이터셋 간의 차이가 특정 피처에 대한 L-infinity Norm의 임곗값을 초과한다면,  
TFDV는 이 장 앞부분에서 정의한 이상치 감지를 사용하여 이상치로 강조한다. 

다음은 데이터셋 간의 왜곡을 비교하는 방법이다.  
```python
tfdv.get_feature(schema,
                 'company').skew_comparator.infinity_norm.threshold = 0.01
skew_anomalies = tfdv.validate_statistics(statistics=train_stats,
                                          schema=schema,
                                          serving_statistics=serving_stats)
```

결과는 다음과 같다. 
<img src="https://user-images.githubusercontent.com/72016560/179162612-baaeeda7-2c73-4877-9d49-ea36f604720b.png" width="700">

