library(readxl)
library(tidyverse)
library(lubridate)
library(stringr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(fpc)
library(dlookr)

dt <- read_excel('C:/tmp/COMMAND_LOG.xlsx')
dt %>% arrange(COMPLETED_TIME) # 기존 엑셀처럼 정렬
# 필요 feature 추가
dt1 <- dt %>% mutate(TRANSFERING_TIME = as.numeric(COMPLETED_TIME - LOADED_TIME),STATION_TYPE = paste(FROM_STATION_TYPE, TO_STATION_TYPE, sep='-')) %>% mutate(TRANSFERING_VELOCITY = TRANSFERING_DISTANCE/as.numeric(TRANSFERING_TIME)) 
dt1 <- dt1 %>% mutate(STATION = as.factor(paste(FROM_STATION, TO_STATION, sep='-')))
dt1 <- dt1 %>% mutate(CLUSTER = as.integer(dt$TRANSFERING_VELOCITY>1530)) # Main cluster와 sub cluster로 clustering
dt1 <- dt1[,-c(1,2,7)] # useless feature 제외


dt2 <- as.matrix(read_csv('C:/tmp/ttmp2.csv',col_names=FALSE))



write.csv(dt2, "C:/tmp/ttmp2.csv")
ccc <- c()
for (i in 1:nrow(dt2)){
    for(j in 2:ncol(dt2)){
        if(dt2[i,j]!=0)
        ccc[i] <- paste(ccc[i],dt2[i,j],sep='-')    
    }
}
c1 <- grepl('InterBay_1',ccc)
c2 <- grepl('InterBay_2',ccc)
c3 <- grepl('InterBay_3',ccc)


a <- dt %>% group_by(STATION) %>% summarise(mm.ratio=max(TRANSFERING_TIME)/min(TRANSFERING_TIME))

## station type에 따른 선형회귀, PLOT
station.type <- unique(dt$STATION_TYPE)
for (i in station.type){
    station.dt <- dt[dt$STATION_TYPE == i,c(8,10)]
    lm1 <- lm(TRANSFERING_TIME ~ TRANSFERING_DISTANCE, station.dt)
    cat(i, lm1$coefficients)
    g <- ggplot(station.dt, aes(x=TRANSFERING_DISTANCE, y=TRANSFERING_TIME)) + geom_point() + stat_smooth(method=lm, colour='black')
    plot(g)
}

## OHT name에 따른 선형회귀, PLOT
oht.name <- unique(dt$OHT_NAME) %>% sort()
for (i in oht.name){
    oht.dt <- dt[dt$OHT_NAME == i,c(8,10)]
    lm1 <- lm(TRANSFERING_TIME ~ TRANSFERING_DISTANCE, oht.dt)
    print(i, lm1$coefficients,'\n')
    gg <- ggplot(oht.dt, aes(x=TRANSFERING_DISTANCE, y= TRANSFERING_TIME)) + geom_point() +stat_summary(method=lm)
    plot(gg)
}
gg <- ggplot(dt[dt$OHT_NAME %in% c('VMS_oht_249','VMS_oht_208','VMS_oht_226','VMS_oht_236'),], aes(x=TRANSFERING_DISTANCE, y= TRANSFERING_TIME, colour=OHT_NAME)) + geom_point() +stat_smooth(method=lm)
plot(gg)

# 출발지, 도착지가 같은 66개의 instance 관찰
same.route.dt <- dt[dt$STATION == 'PHT0310_ProcessPort_32567-PHT0409_ProcessPort_31982',]

a <- hist(same.route.dt$TRANSFERING_DISTANCE)
b <- hist(as.numeric(same.route.dt$TRANSFERING_TIME))
c <- hist(same.route.dt$TRANSFERING_VELOCITY)
gg <- ggplot(same.route.dt, aes(x=TRANSFERING_DISTANCE, y= TRANSFERING_TIME, colour=CLUSTER)) + geom_point() +stat_smooth(method=lm)
plot(gg)


# cluster -  scatter plot
g <- ggplot(dt, aes(x=TRANSFERING_DISTANCE, y=TRANSFERING_TIME, colour=CLUSTER)) + geom_point() + stat_smooth(method=lm, colour='red')
plot(g)


#선형회귀
lm1 <- lm(TRANSFERING_TIME ~ TRANSFERING_DISTANCE, dt1)
y.hat <- predict(lm1, newdata=dt1)
percentage.error <- 100 * (y.hat-dt1$TRANSFERING_TIME)/abs(dt1$TRANSFERING_TIME)



# distance에 따라 binning 후 각 bin에 따른 선형회귀오차(traffic?)
sub_dt <-sub_dt %>% 
    mutate(distance_bin = extract(binning(sub_dt$TRANSFERING_DISTANCE, type='equal', nbins=8)))
bins <- unique(sub_dt$distance_bin)

par(mfrow=c(2,4))
for (i in 1:8){
    sub_dt2 <- sub_dt[sub_dt$distance_bin==bins[i],]
    plot(ggplot(sub_dt2, aes(x=percentage.error)) + geom_histogram())
}

dt %>% group_by(STATION_TYPE) %>% summarise(mean.time = mean(TRANSFERING_TIME), mean.distance = mean(TRANSFERING_DISTANCE), n = n(), sub.cluster =sum(CLUSTER), sub.ratio = sum(CLUSTER)/n())