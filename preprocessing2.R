if(!require("tidyverse")) install.packages("tidyverse")
if(!require("glmnet")) install.packages("glmnet")
if(!require("writexl")) install.packages("writexl")

require(tidyverse)
require(glmnet)
require(writexl)
select <- dplyr::select

# 작업 디렉토리 설정
setwd("C:/Users/LGPC/Desktop/project/KCYPS2018m1w2[CSV]/")

# 자녀 조사 데이터 불러오기
student_raw <- read_csv("KCYPS2018m1Yw2.csv")
# 부모 조사 데이터 불러오기
parent_raw <- read_csv("KCYPS2018m1Pw2.csv")

# 식별ID로 데이터 병합
data_raw <- inner_join(parent_raw, student_raw, by=c("PID", "HID"))

# 청소년, 부모 모두 조사에 참여한 데이터만 추출
data_raw2 <- data_raw %>% filter(SURVEY1w2==1, SURVEY2w2==1)

# 변수명 생성
feature_names <- colnames(data_raw2)

# 결측치가 10% 넘는 컬럼 삭제
na_idx <- which(apply(is.na(data_raw2), 2, sum) > nrow(data_raw2)*0.1) # 결측치 존재하는 컬럼 인덱스 찾기
data <- data_raw2 %>% select(-feature_names[na_idx])

# 정제된 data의 변수명 재생성
feature_name <- colnames(data)

#########################
# 반응변수(비행) 중영역별로 묶기 (비행변수 코드 "YDLQ1")
idx_y <- str_detect(feature_name, "YDLQ1")
y_raw <- data %>% select(feature_name[idx_y])

# 한번이라도 비행 경험이 있으면 1, 없으면 0
y_sum <- apply(y_raw, 1, sum)
y <- ifelse(y_sum==15, 0, 1)

#########################

## train, test set 나누기

n_obs <- length(y)
set.seed(42)
train_idx <- sample(1:n_obs, n_obs*0.8)
test_idx <- setdiff(1:n_obs, train_idx)

train_y <- y[train_idx]
test_y <- y[test_idx]

train_data <- data %>% slice(train_idx)
test_data <- data %>% slice(test_idx)

##### 결측치 median으로 대처

median_value <- apply(train_data, 2, function(x)
  median(as.numeric(x), na.rm=T))

median_input <- function(x){
  x <- as.numeric(x)
  x[is.na(x)] <- median(x, na.rm=T)
  return(x)
}

train_data <- sapply(train_data, median_input)

for(i in 1:nrow(test_data)){
  for(j in 1:ncol(test_data)){
    if(is.na(test_data[i,j])){
      test_data[i,j] <- median_value[j]
    }
  }
}

train_data <- as.data.frame(train_data)
test_data <- as.data.frame(test_data)

############################## LASSO를 활용한 변수선택

# 5-fold CV를 위한 fold index 설정
set.seed(42)
foldid <- sample(1:5, nrow(train_data), replace=T)

# Hyperparameter
lam <- 10^seq(5, -3, -0.1)

library(glmnet)

select_feature_lasso <- function(code, train_data, test_data){
  
  feature <- colnames(train_data)
  # 변수코드와 일치하는 feature뽑기 
  idx <- str_detect(feature, code)
  train_x_raw <- train_data %>% select(feature[idx]) %>% as.matrix()
  # LASSO는 scaling에 민감하므로 scaling 후, 변수 선택
  train_x <- scale(train_x_raw, scale=T)
  # LASSO
  cv_m <- cv.glmnet(train_x, factor(train_y), alpha=1, 
                    foldid=foldid, lambda=lam, family="binomial")
  
  # coefficient 뽑기
  coefs <- coef(cv_m, s="lambda.1se") # one-standard-error-rule
  selected_idx <- which(coefs != 0)
  selected_feature <- rownames(coefs)[selected_idx]
  
  # 최종적으로 선택된 변수로 데이터 구성
  train_result <- train_data %>% select(selected_feature[-1])
  test_result <- test_data %>% select(selected_feature[-1])
  
  result <- list(tr=train_result, te=test_result)
  return(result)
}

## 자녀 생활 시간 변수(코드 "YTIM")
train_YTIM <- select_feature_lasso("YTIM", train_data, test_data)$tr
test_YTIM <- select_feature_lasso("YTIM", train_data, test_data)$te

## 자녀 지적 발달 변수(코드 "YINT")
train_YINT <- select_feature_lasso("YINT", train_data, test_data)$tr
test_YINT <- select_feature_lasso("YINT", train_data, test_data)$te

## 자녀 진로 변수(코드 "YFUR")
train_YFUR <- select_feature_lasso("YFUR", train_data, test_data)$tr
test_YFUR <- select_feature_lasso("YFUR", train_data, test_data)$te

## 자녀 사회/정서/역량 발달 변수(코드 "YPSY")
train_YPSY <- select_feature_lasso("YPSY", train_data, test_data)$tr
test_YPSY <- select_feature_lasso("YPSY", train_data, test_data)$te
cor(data.frame(train_YPSY, train_y))

## 자녀 온라인비행 유무(코드 "YDLQ2")
train_YDLQ <- select_feature_lasso("YDLQ2", train_data, test_data)$tr
test_YDLQ <- select_feature_lasso("YDLQ2", train_data, test_data)$te

## 자녀 신체 발달 변수(코드 "YPHY")
train_YPHY <- select_feature_lasso("YPHY", train_data, test_data)$tr
test_YPHY <- select_feature_lasso("YPHY", train_data, test_data)$te

## 자녀 매체 변수(코드 "YMDA")
train_YMDA <- select_feature_lasso("YMDA", train_data, test_data)$tr
test_YMDA <- select_feature_lasso("YMDA", train_data, test_data)$te

## 자녀 활동/문화 환경 변수(코드 "YACT")
train_YACT <- select_feature_lasso("YACT", train_data, test_data)$tr
test_YACT <- select_feature_lasso("YACT", train_data, test_data)$te
cor(data.frame(train_YACT, train_y))

## 자녀 학교 변수(코드 "YEDU")
train_YEDU <- select_feature_lasso("YEDU", train_data, test_data)$tr
test_YEDU <- select_feature_lasso("YEDU", train_data, test_data)$te
cor(data.frame(train_YEDU, train_y))

## 자녀 가정 변수(코드 "YFAM")
train_YFAM <- select_feature_lasso("YFAM", train_data, test_data)$tr
test_YFAM <- select_feature_lasso("YFAM", train_data, test_data)$te
cor(data.frame(train_YFAM, train_y))

## 부모 사회/정서/역량 변수(코드 PPSY)
train_PPSY <- select_feature_lasso("PPSY", train_data, test_data)$tr
test_PPSY <- select_feature_lasso("PPSY", train_data, test_data)$te
cor(data.frame(train_PPSY, train_y))

## 부모 신체 변수(코드 "PPHY")
train_PPHY <- select_feature_lasso("PPHY", train_data, test_data)$tr
test_PPHY <- select_feature_lasso("PPHY", train_data, test_data)$te
cor(data.frame(train_PPHY, train_y))


train_x <- data.frame(train_YTIM, train_YINT, train_YFUR, 
                      train_YPSY, train_YDLQ, train_YPHY, 
                      train_YMDA, train_YACT, train_YEDU, 
                      train_YFAM, train_PPSY, train_PPHY)

test_x <- data.frame(test_YTIM, test_YINT, test_YFUR, 
                      test_YPSY, test_YDLQ, test_YPHY, 
                      test_YMDA, test_YACT, test_YEDU, 
                      test_YFAM, test_PPSY, test_PPHY)


tr_data <- data.frame(train_x, y=as.factor(train_y))
te_data <- data.frame(test_x, y=as.factor(test_y))

writexl::write_xlsx(tr_data, path = "train_data.xlsx")
writexl::write_xlsx(te_data, path = "test_data.xlsx")

