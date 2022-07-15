
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

학습 데이터셋과 서빙 데이터셋 간의 데이터 드리프트 시각화의 결과는 다음과 같다.

<img src="https://user-images.githubusercontent.com/72016560/179162612-baaeeda7-2c73-4877-9d49-ea36f604720b.png" width="700">

skew_comparator와 drift_comparator의 L-infinity Norm은 데이터 입력 파이프라인에 문제가 있음을 알려주는 데이터셋 간의 큰 차이를 보여주는 데 유용하다. L-infinity Norm은 단일 숫자만 반환하므로 스키마가 데이터셋 간의 변동을 탐지하는데 더 유용하기도 하다. 

### 4.3.4 편향된 데이터셋
여기서 편향은 '현실 세계와 동떨어진 데이터' 이다.  

실제 세계를 표본으로 추출하는 방법은 어떤 식으로든 항상 편향된다.  
데이터셋은 항상 실제 환경의 부분 집합이며, 우리는 모든 세부 정보를 캡처할 수 없기 때문이다. 

#### 선택 편향
- 데이터셋의 분포가 실제 데이터 분포와 같지 않은 상황
- TFDV의 시각화를 통해 확인 가능
- 예) 데이터셋에 범주형 피처로 Gender가 포함되면, 값이 남성 범주에 치우치지 않았는지 확인 가능
- 만약, 편향이 확인되었고 그 편향이 모델의 성능을 해칠 수 있는 경우에는  
  다시 돌아가서 더 많은 데이터를 수집하거나 오버/언더샘플링하여 정확한 분포를 얻어야 한다. 
- 
