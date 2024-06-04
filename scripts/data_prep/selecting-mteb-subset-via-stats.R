#!/bin/env Rscript
library(glmnet)

csv<-commandArgs(trailingOnly=T)[1]
dir<-dirname(csv)
df<-read.csv(csv, header=T)
adf<-df$Average
df$Model<-NULL
df$Average<-rowMeans(df)
correlation<-cor(df)
correlation<-correlation[nrow(correlation),]
correlation<-sort(correlation,decreasing=T)
data.frame(correlation)

y<-df$Average
df$Average<-NULL
x<-data.matrix(df)
cv_fit <- cv.glmnet(x, y, alpha = 1)
coef(cv_fit)
