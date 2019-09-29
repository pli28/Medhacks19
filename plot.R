# remember your input format: Rscript cell_type.R [input file]
library(ggplot2)

svd_files <- c('dn25.csv','dn30.csv','dn35.csv','dn40.csv','dn60.csv','dn80.csv','dn100.csv')
for (svd in svd_files){
  dat <- read.csv(svd) 
  p <- ggplot(dat, aes(x=DiseaseName, y=Frequency))
  p <- p + geom_bar(stat="identity", width=0.75)
  p <- p + ggtitle(paste('Disease Diagnosis Variability with SVD',substring(svd,3,4),sep=''))
  p <- p + xlab('Disease Diagnosed')
  p <- p + scale_y_continuous(limits=c(0,500))
  p <- p + theme(plot.title = element_text(size=14, face="bold"), 
  axis.text.x = element_text(size= 6,angle = 60, hjust = 1))

  ggsave(paste('svd', substring(svd,3,4),'.pdf',sep=''), p, width=18, height=4)
}