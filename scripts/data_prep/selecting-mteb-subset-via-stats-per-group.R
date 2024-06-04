#!/bin/env Rscript

csv<-commandArgs(trailingOnly=T)[1]
tasks<-commandArgs(trailingOnly=T)[2]
df<-read.csv(csv, header=T)
tasks<-read.csv(tasks, header=F)
df$Model<-NULL
df$Average<-NULL
tasks<-tasks[,1]
tasks

for (t in tasks){
    print(t)
    df$t
}

df<-df[, tasks]
df$Average<-rowMeans(df)

correlation<-cor(df)
correlation<-correlation[nrow(correlation),]
correlation<-sort(correlation,decreasing=T)
data.frame(correlation)
