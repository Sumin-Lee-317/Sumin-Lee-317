
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

