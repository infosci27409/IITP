library(tidyverse)
setwd("C:/SDI")


dtex <- function(i){
    return (dt[(i-3):(i+3),1:10])
}

dt <- read_csv('a.csv')
dt <- dt  %>%
    mutate(EVENTTIME = ymd_hms(EVENTTIME))
para_dt <- dt[,10:72]

# 모든 value가 NA인 column 제거
# dt <- read_csv('C:/SDI/CA6PRE01.csv')
# 
# dt <- dt %>%
#     select(which(colSums(is.na(dt))!=nrow(dt)))
# 
# write.csv(dt, 'a.csv', row.names = FALSE)



# 모든 column에 대한 시계열 value 변화 그래프

# for(i in 10:72){
#     print(colnames(dt)[i])
#     t.dt <- dt[,i]
#     t.dt[is.na(t.dt)] <- mean(t.dt)
# 
#     # print(table(t.dt))
#     filepath <- sprintf('%s.png',colnames(dt)[i])
#     png(filepath)
#     plot(t.dt, type='l')
#     dev.off()
# }

# column별 평균 구하기(NA값은 평균으로 대체)
# for(i in 1:ncol(para_dt)){
#     avg <- mean(para_dt[,i][!(is.na(para_dt[,i]))])
#     para_dt[,i][is.na(para_dt[,i])] <- avg
# }
# colmean <- para_dt %>% colMeans()
# plot(colmean, type='o', col='blue', lwd='2')

# EVENT, LOTEVENT, LOTID 가 NA값에서 탈출한 row들의 인덱스
# start_EVENT_diff <- c()
# for(i in 1:nrow(dt)){
#     if(dt$EVENT[i]=="" & dt$EVENT[i+1]!=""){
#         start_EVENT_diff<- c(start_EVENT_diff, i)
#     }
# }
# start_LOTEVENT_diff <- c()
# for(i in 1:nrow(dt)){
#     if(dt$LOTEVENT[i]=="" & dt$LOTEVENT[i+1]!=""){
#         start_LOTEVENT_diff<- c(start_LOTEVENT_diff, i)
#     }
# }
# start_LOTID_diff <- c()
# for(i in 1:nrow(dt)){
#     if(dt$LOTID[i]=="" & dt$LOTID[i+1]!=""){
#         start_LOTID_diff<- c(start_LOTID_diff, i)
#     }
# }
# start_STATE_diff <- c()
# for(i in 1:nrow(dt)){
#     if(dt$STATE[i]=="" & dt$STATE[i+1]!=""){
#         start_STATE_diff<- c(start_STATE_diff, i)
#     }
# }
# start_MODE_diff <- c()
# for(i in 1:nrow(dt)){
#     if(dt$MODE[i]=="" & dt$MODE[i+1]!=""){
#         start_MODE_diff<- c(start_MODE_diff, i)
#     }
# }
# start_MODESTATE_diff <- c()
# for(i in 1:nrow(dt)){
#     if(dt$MODESTATE[i]=="" & dt$MODESTATE[i+1]!=""){
#         start_MODESTATE_diff<- c(start_MODESTATE_diff, i)
#     }
# }
# start_RECIPEID_diff <- c()
# for(i in 1:nrow(dt)){
#     if(dt$RECIPEID[i]=="" & dt$RECIPEID[i+1]!=""){
#         start_RECIPEID_diff<- c(start_RECIPEID_diff, i)
#     }
# }
# start_LOTEVENT_diff
# start_EVENT_diff
# start_STATE_diff
# start_MODE_diff
# start_MODESTATE_diff
# start_LOTID_diff
# start_RECIPEID_diff
# 
# 
# # LOTEVENT, EVENT, LOTID 가 NA값의 반복이 시작된 row의 인덱스
# stop_EVENT_diff <- c()
# for(i in 1:(nrow(dt)-1)){
#     if(dt$EVENT[i]!="" & dt$EVENT[i+1]==""){
#         stop_EVENT_diff<- c(stop_EVENT_diff, i)
#     }
# }
# stop_LOTEVENT_diff <- c()
# for(i in 1:(nrow(dt)-1)){
#     if(dt$LOTEVENT[i]!="" & dt$LOTEVENT[i+1]==""){
#         stop_LOTEVENT_diff<- c(stop_LOTEVENT_diff, i)
#     }
# }
# stop_LOTID_diff <- c()
# for(i in 1:(nrow(dt)-1)){
#     if(dt$LOTID[i]!="" & dt$LOTID[i+1]==""){
#         stop_LOTID_diff<- c(stop_LOTID_diff, i)
#     }
# }
# stop_STATE_diff <- c()
# for(i in 1:(nrow(dt)-1)){
#     if(dt$STATE[i]!="" & dt$STATE[i+1]==""){
#         stop_STATE_diff<- c(stop_STATE_diff, i)
#     }
# }
# stop_MODE_diff <- c()
# for(i in 1:(nrow(dt)-1)){
#     if(dt$MODE[i]!="" & dt$MODE[i+1]==""){
#         stop_MODE_diff<- c(stop_MODE_diff, i)
#     }
# }
# stop_MODESTATE_diff <- c()
# for(i in 1:(nrow(dt)-1)){
#     if(dt$MODESTATE[i]!="" & dt$MODESTATE[i+1]==""){
#         stop_MODESTATE_diff<- c(stop_MODESTATE_diff, i)
#     }
# }
# stop_RECIPEID_diff <- c()
# for(i in 1:(nrow(dt)-1)){
#     if(dt$RECIPEID[i]!="" & dt$RECIPEID[i+1]==""){
#         stop_RECIPEID_diff<- c(stop_RECIPEID_diff, i)
#     }
# }
# 
# 
# stop_LOTEVENT_diff
# stop_EVENT_diff
# stop_LOTID_diff
# stop_STATE_diff
# stop_MODE_diff
# stop_MODESTATE_diff
# stop_RECIPEID_diff

# LOTEVENT PLOT(NA;0, START_LOT;1, END_LOT;2)
# tmp <- vector(length=1296000)
# tmp[which(is.na(dt$LOTEVENT))] <- 0 
# tmp[which(dt$LOTEVENT=="START_LOT")] <- 1
# tmp[which(dt$LOTEVENT=="END_LOT")] <- 2
# 
# png('LOTEVENT.png')
# plot(tmp, type='p')
# dev.off()



ttmp <- c()
for(i in 1:(1296000-1)){
    print(i)
    if(!is.na(dt$LOTEVENT[i]) & !is.na(dt$LOTEVENT[i+1])){
        if(dt$LOTEVENT[i]=="END_LOT" & dt$LOTEVENT[i+1]=="START_LOT"){
        ttmp <- c(ttmp, i)}
    }
}

