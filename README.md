# Diary

# 10. 실전 데이터 분석 프로젝트
## 10.1 foreign 패키지
- SPSS, SAS, STATA 등 다양한 통계분석 소프트웨어의 파일을 불러올 수 있다.
- read.spss(file = "파일경로/파일명.sav", to.data.frame = T) 사용: SPSS 파일을 데이터프레임으로 변환하여 불러오는 기능
```R
# foreign 패키지의 read.spss()
data <- read.spss(file = "파일경로/파일명.sav", 
                  to.data.frame = T)  # SPSS파일을 데이터프레임 형태로 변환, 이게 없으면 리스트로 불러옴
```

## 10.2 분석 프로젝트 절차
### 10.2.1 패키지 설치 및 로드
### 10.2.2 변수 검토
 - class() : 변수 타입 파악
 - 범주변수이면, table()로 빈도 파악
 - 연속변수이면, summary()로 요약통계량 확인
### 10.2.3 전처리
 - 이상치 확인
 - 이상치 결측 처리
 - 결측치 확인
 - 변수명 바꾸기
### 10.2.4 변수간 관계 분석하기
 - 평균표 만들기
 - 파생변수 만들기
 - 그래프 만들기
 
## 10.3 그래프 
### 10.3.1 범주 순서 지정하기
- scale_x_discrete(limits = c("범주1', "범주2, ...))
- 이때 
```R
# 막대를 young, middle, old 순으로 정렬
ggplot(data = ageg_income, aes(x=ageg, y=mean_income)) +
     geom_col() +
     scale_x_discrete(limits = c("young", "middle", "old"))
```

### 10.3.2 막대가 특정 범주에 따라 다른 색으로 표현되게끔 설정
- aes()에 fill=특정범주 를 추가하면 됨
```R
# 막대가 성별에 따라 다른 색으로 표현
ggplot(data = ageg_income, aes(x=ageg, y=mean_income, fill=sex)) +
     geom_col() +
     scale_x_discrete(limits = c("young", "middle", "old"))
```

### 10.3.3 막대 색깔에 지정할 변수의 범주 순서 지정
- 이때, 변수가 factor 타입이어야함
- factor타입이 아니라면? factor()로 변환해야함
- 변수 <- level = c("범주1", "범주2", ...)
```R
# 위에서는 young-, ageg의 범주 순서를 바꾸고 싶은 경우
ageg <- factor(ageg,
               level = c("old","middle","young"))
```

### 10.3.4 막대 분리하기
- geom_col()에 position="dodge" 를 추가하면 됨
```R
# 막대가 성별에 따라 다른 색으로 표현
ggplot(data = ageg_income, aes(x=ageg, y=mean_income, fill=sex)) +
     geom_col(position="dodge") +
     scale_x_discrete(limits = c("young", "middle", "old"))
```

### 10.3.5 그래프 회전하기
- coord_flip() : 그래프를 오른쪽으로 90도 회전
- x축의 범주이름이 길면 가로로 겹치기때문에 회전해서 겹치지않게끔 한다.
```R
# 막대가 성별에 따라 다른 색으로 표현
ggplot(data = top10, aes(x=reorder(job,-mean_income), y=mean_income)) +
     geom_col() +
     coord_flip()
```

### 10.3.6 빈도 막대그래프에서 빈도 표시하기
```R
ggplot(data = top10, aes(x=reorder(job,-mean_income), y=mean_income)) +
     geom_col() +
     scale_x_discrete(limit = order) +
     geom_text(aes(label = freq), hjust = -0.3)
```

# 11. 텍스트 마이닝

## 11.1 텍스트 마이닝 준비하기

### 11.1.1 패키지 로드
```R
library(KoNLP)  # 한글 자연어 분석 패키지
library(dplyr)
useNIADic()
```
### 11.1.2 데이터 불러오기
```R
txt <- readLines("파일명")
```
```R
txt <- read.csv("파일명",
                header = T, # 변수명 다음에 데이터 시작
                stringsAsFactors = F, # 문자 들어있는 파일
                fileEncoding = "UTF-8")
```
### 11.1.3 특수문자 제거하기
```R
install.packages("stringr")
library(stringr)

# txt파일의 특수문자(\\W)를 공백으로 바꾸기
txt <- str_replace_all(txt, "\\W", " ")
```

## 11.2 가장 많이 사용된 단어 알아보기

### 11.2.1 txt에서 명사 추출
- extractNoun() 사용: 출력결과를 리스트로 반환
```R
nouns <- extractNoun(txt)
```
### 11.2.2 추출한 명사 list를 문자열 벡터로 변환하고, 단어별 빈도표 생성
- 단어별 빈도표? 각 단어가 몇 번씩 사용됐는지 나타냄
- table(unlist()) 사용: 리스트 해제 후 빈도표 생성
```R
wordcount <- table(unlist(nouns))
```
### 11.2.3 단어별 빈도표를 데이터프레임으로 변환
```R
df_word <- as.data.frame(wordcount, stringsAsFactors = F)
```
### 11.2.4 (데이터프레임)변수명 수정
```R
df_word <- rename(df_word,
                  word = Var1,
                  freq = Freq)
```
### 11.2.5 두 글자 이상의 단어 빈도표 만들기
- 한 글자 단어는 의미없는 경우(조사)가 많으므로 두 글자 이상만 추출해야한다.
- nchar() 사용: 변수값의 길이를 반환
```R
# 두 글자 이상의 단어 추출
df_word <- filter(df_word, nchar(word) >= 2)
```
### 11.2.6 빈도순으로 정렬 후 상위 20개 단어 추출
```R
top20 <- df_word %>%
    arrange(desc(freq)) %>%
    head(20) 
```

## 11.3 워드 클라우드 만들기

### 11.3.1 패키지 준비하기
```R
install.packages("wordcloud")
library(wordcloud)
library(RColorBrewer)
```
### 11.3.2 단어 색상 목록 만들기 (팔레트)
- pal <- brewer.pal(필요한 색상 수, "팔레트명")
```R
# Dark2에서 8가지 색상 추출
pal <- brewer.pal(8, Dark2)
```
### 11.3.3 난수 고정하기
- wordcloud()는 함수를 실행할 때마다 난수를 이용해 매번 다른 모양의 워드 클라우드를 생성한다.
- 항상 동일한 워드 클라우드를 만드려면, wordcloud() 실행 전에 난수를 고정해야한다. 
```R
set.seed(1234)
```
### 11.3.4 워드 클라우드 만들기
```R
wordcloud(words = df_word$word, # 단어
          freq = df_word$freq,  # 빈도
          min.freq = 2,         # 최소 단어 빈도
          max.words = 200,      # 표현 단어 수
          random.order = F,     # 고빈도 단어 중앙 배치
          rot.per = .1,         # 회전 단어 비율
          scale = c(4, 0.3),    # 단어 크기 범위
          colors = pal)         # 색상 목록
```
많이 사용된 단어일수록 글자가 크고 가운데에 배치된다.

### 11.3.5 단어 색상 바꾸기
```R
pal <- brewer.pal(9, "Blues")[5:9]
set.seed(1234)
wordcloud( ... )
```

# 12. 지도 시각화
- 단계 구분도(Choropleth Map) : 지역별 통계치를 색깔의 차이로 표현한 지도
## 12.1 미국 주별 강력 밤죄율 단계 구분도 만들기

### 12.1.1 패키지 준비하기
- 단계 구분도 생성시 필요한 패키지
```R
install.packages("mapproj")
install.packages("ggiraphExtra")
library(ggiraphExtra)
```
### 12.1.2 데이터 준비하기
- USArrests 데이터 (R에 내장)
- 지역명 변수가 따로 없고, 대신 행(row) 이름이 지역명으로 되어있다. 
- tibble 패키지의 rownames_to_column() 사용: 행이름을 state 변수로 바꿔 새 데이터 프레임 생성
- tolower() 사용: 지도 데이터의 지역명 변수는 모두 소문자이므로, state도 소문자로 수정
```R
library(tibble)

crime <- rownames_to_column(USArrests, var = "state")
crime$state <- tolower(crime$state)
```
### 12.1.2 데이터 준비하기
- 단계 구분도를 만드려면, 지역별 위도/경도 정보가 있는 지도 데이터가 필요하다. 
- 미국 주별 위경도 데이터가 들어있는 map 패키지 필요
- ggplot2의 map_data() 사용: 데이터 프레임 형태로 불러오기
```R
install.packages("maps")
library(ggplot2)
states_map <- map_data("state")
```

### 12.1.3 단계 구분도 만들기
- ggiraphExtra 패키지의 ggChoropleth() 사용
- 살인범죄건수를 색깔로 표현
```R
ggChoropleth(data = crime,        # 지도에 표현할 데이터
             aes(fill = Murder,   # 색깔로 표현할 변수
                 map_id = state), # 지역 기준(구분) 변수
             map = states_map)    # 지도 데이터
```

### 12.1.4 인터랙티브 단계 구분도 만들기
- interactive = True : 마우스 움직임에 반응
```R
ggChoropleth(data = crime,        # 지도에 표현할 데이터
             aes(fill = Murder,   # 색깔로 표현할 변수
                 map_id = state), # 지역 기준(구분) 변수
             map = states_map,    # 지도 데이터
             interactive = True)  # 인터랙티브
```

## 12.2 한국 시도별 인구 단계구분도 만들기
### 12.2.1 패키지 준비하기
```R
install.packages("stringi")
install.packages("devtools")
devtools::install_github("cardiomoon/kormaps2014")
library(kormaps2014)
```
### 12.2.2 변수명 영어로 바꾸기
```R
str(changeCode(korpop1))

library(dplyr)
korpop1 <- rename(korpop1, pop = 총인구_명, name = 행정구역별_읍면동)
# 지역명이 꺠지지않도록 name의 인코딩을 CP949로 변경
korpop1$name <- iconv(korpop1$name, "UTF-8", "CP949")
```

### 12.2.3 단계 구분도 만들기
```R
ggChoropleth(data = korpop1,      # 지도에 표현할 데이터
             aes(fill = pop,      # 색깔로 표현할 변수
                 map_id = code,   # 지역 기준 변수
                 tooltip = name), # 지도 위에 표시할 지역명
             map = kormap1,       # 지도데이터
             interactive = T)     # 인터랙티브
```

# 13. 인터랙티브 그래프
- 인터랙티브 그래프(Interactive Graph): 마우스 움직임에 반응하며 실시간으로 형태가 변하는 그래프
- ggplot2로 만든 그래프를 **ggplotly()** 에 넣어주면 된다.
## 13.1 인터랙티브 그래프 만들기
### 13.1.1 패키지 준비하기
```R
install.packages("plotly")
library(plotly)
```
### 13.1.2 ggplot2로 그래프 만들기
```R
library(ggplot2)
p <- ggplot(data = mpg, aes(x = displ, y = hwy, col = drv)) + geomplot()
```
### 13.1.3 인터랙티브 그래프 생성
```R
ggplotly(p)
```
## 13.2 인터랙티브 시계열 그래프 만들기
### 13.2.1 패키지 설치하기
```R
install.packages("dygraphs")
library(dygraphs)
```
### 13.2.2 데이터 준비하기
- 인터랙티브 시계열 그래프를 만드려면, 데이터가 시간순서속성을 지니는 xts타입이어야함
- xts() 사용
```R
economics <- ggplot2::economics
library(xts)
# unemploy를 xts타입으로 바꿔주기
eco <- xts(economics$unemploy, order.by = economics$date)
```
### 13.2.3 인터랙티브 시계열 그래프 만들기
- dygraph() 사용
```R
dygraph(eco)
```
### 13.2.4 날짜 범위 선택 기능 추가
```R
dygraph(eco) %>% dyRangeSelector()
```
### 13.2.5 여러 값 표현하기
그래프에 마우스를 갖다대면 저축률과 실업자수를 동시에 나타내도록 만들어보자. 
```R
# 저축률
eco_a <- xts(economics$psavert, order.by = economics$date)
# 실업자수 (단위맞추기위해 /1000)
eco_b <- xts(economics$unemploy/1000, order.by = economics$date)

# 저축률과 실업자수를 가로로 결합
cbind(eco_a, eco_b)

# 변수명 변경
colnames(eco2) <- c("psavert","unemploy")

# 그래프 생성
dygraph(eco2) %>% byRangeSelector()
```

# 14. 통계 분석 기법을 이용한 가설 검정
## 14.1 통계적 가설 검정이란?
### 14.1.1 기술통계 vs 추론통계
통계분석은 기술통계와 추론통계로 나눌 수 있다. 
- 기술통계분석: 데이터를 요약해 설명하는 통계 기법
  
  ex) 사람들이 받는 월급을 집계해 전체 월급 평균을 구하는 것
- 추론통계분석: 단순히 숫자를 요약하는 것을 넘어 어떤 값이 발생할 확률을 계산하는 통계 기법

  ex) 수집된 데이터에서 성별에 따라 월급에 차이가 있는 것으로 나타났을 때, 
  
      이런 차이가 우연히 발생할 확률을 계산하는 것
 - 이런 차이가 나타날 확률이 작다면, 성별에 따른 월급 차이가 통계적으로 유의하다.
 - 이런 차이가 나타날 확률이 크다면, 성별에 따른 월급 차이가 통계적으로 유의하지 않다.
 
 ### 14.1.2 통계적 가설 검정
 - 통계절 가설 검정: 유의확률을 이용해 가설을 검정하는 방법
 - **유의확률(p-value)**: 실제로는 집단간의 차이가 없는데 우연히 차이가 있는 데이터가 추출될 확률
> 유의확률이 크다면, '집단간 차이가 통계적으로 유의하지않다.'
>(=실제로 차이가 없더라도 우연에 의해 이 정도의 차이가 관찰될 가능성이 크다.)

> 유의확률이 작다면, '집단간 차이가 통계적으로 유의하다.'
>(=실제로 차이가 없는데 우연히 이 정도의 차이가 관찰될 가능성이 작다. 우연이라고 보기 힘들다.)
 - t 검정: 두 집단의 평균 비교
 - 상관분석: 두 변수의 관계성 분석

### 14.2 t 검정 - 두 집단의 평균 비교
- **t 검정**: 두 집단의 평균에 통계적으로 유의한 차이가 있는지 알아보기 위해 사용하는 통계분석기법
- t.test() 사용 (R에 내장)
### *compact 자동차와 suv 자동차의 도시연비 t검정*
#### 1. 데이터 불러오고, 필요한 데이터만 추출
```R
mpg <- as.data.frame(ggplot2::mpg)
library(dplyr)
mpg_diff <- mpg %>%
  select(class, cty) %>% # class, cty 변수만 남기기
  filter(class %in% c("compact","suv")) # class가 compact, suv인 자동차 추출
head(mpg_diff)
table(mpg_diff$class)

##
## compact    suv
##      47     62
```
#### 2. t 검정
- **t.test(data=데이터프레임명, 비교할값 ~ 비교할집단, var.equal = T) 사용**
```R
t.test(data= mpg_diff, cty ~ class, var.equal = T) # 집단간 분산이 같다고 가정

## Two Sample t-test
## 
## data:  cty by class
## t = 11.917, df = 107, p-value < 2.2e-16
## alternative hypothesis: true difference in means is not equal to 0
## 95 percent confidence interval:
##   5.525180 7.730139
## sample estimates:
## mean in group compact     mean in group suv 
##              20.12766              13.50000 
```
(1) p-value < 0.05 이므로, compact와 suv 간 평균도시연비 차이가 통계적으로 유의하다.

(2) compact 집단의 평균은 20, suv 집단의 평균은 13이므로, suv보다 compact의 도시연비가 더 높다.

### 14.3 상관분석 - 두 변수의 관계성 분석
### *실업자 수와 개인 소비 지출의 상관관계*
#### 1. 데이터 불러오고, 필요한 데이터만 추출
```R
economics <- as.data.frame(ggplot2::economics)
cor.test(economics$unemploy, economics$pce)

##  Pearson's product-moment correlation

## data:  economics$unemploy and economics$pce
## t = 18.63, df = 572, p-value < 2.2e-16
## alternative hypothesis: true correlation is not equal to 0
## 95 percent confidence interval:
##  0.5608868 0.6630124
## sample estimates:
##       cor 
## 0.6145176 
```
(1) p-value < 0.05 이므로, 실업자수와 개인소비지출의 상관이 통계적으로 유의하다.

(2) 상관계수가 0.61(양수)이므로 양의상관관계를 가진다.

    즉, 실업자수와 개인소비지출은 정비례관계이다.
    
### 14.4 상관분석 - 여러 변수의 관계성 분석 (상관행렬 히트맵 만들기)
- 상관행렬: 여러 변수 간 상관계수를 행렬로 나타낸 표, 어떤 변수끼리 관련이 크고 적은지 파악 가능
- 히트맵 : 값의 크기를 색깔로 표현한 그래프
### *실업자 수와 개인 소비 지출의 상관관계*
#### 1. 데이터 불러오고, 필요한 데이터만 추출
```R
head(mtcars)
car_cor <- cor(mtcars)
round(car_cor, 2)
```
#### 2. 상관행렬 히트맵 만들기
```R
install.packages("corrplot")
library(corrplot)
corrplot(car_cor) # 원으로 표시
corrplot(car_cor, method="number") # 원대신 상관계수로 표시

col <- colorsRampPalette("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA")

corrplot(car_cor,
         method = "color", # 색깔로 표현
         col = col(200),   # 색상 200 개 선정
         type = "lower",   # 왼쪽 아래 행렬만 표시
         order = "hclust", # 유사한 상관계수끼리 군집화
         addCoef.col = "black", # 상관계수 색깔
         tl.col = "black", # 변수명 색깔
         tl.srt = 45,      # 변수명 45 도 기울임
         diag = F)         # 대각 행렬 제외
```
