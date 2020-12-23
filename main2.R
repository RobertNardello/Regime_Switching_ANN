
rm(list = ls())

library(forecast)  #for forecast function
library(dplyr)
library(depmixS4)
library(ggplot2)

# Load Data
data <- read.csv("C:/Users/Hyeoncheol/Dropbox/1 Research Project/Robert Nardello/R Codes/Real_GDP20.csv")


# Set parameters
n.train <- 15        # Number of time periods in a training set
min.n.train <- 5     # Minimum number of time periods in a training set
n.lag <- 4           # Number of lagged variables (Number of input nodes in a neural network)
h <- 1               # Forecasting horizon
n.hidden <- 5        # Number of hidden nodes in a neural network


# Common variables
set.seed(1000)
n.fcast <- dim(data)[1]-h-n.train+1


## Proposed approach: Neural network with a refined training set by using regime switching
out.nn.hmm <- matrix(NA, nrow = n.fcast)
for (i in 1:n.fcast) {
  train <- data[i:(i+n.train-1),]
  train <- dplyr::select(train, GDPC1)
  test <- data[(i+n.train):(i+n.train-1+h),]
  # Check regime switching 
  hmm <- depmix(GDPC1 ~ 1, family = gaussian(), nstates = 2, data = train)
  hmmfit <- fit(hmm, verbose = FALSE)
  state <- hmmfit@posterior[["state"]]
  for (j in n.train:1) {
    if (state[j] != state[j-1]) {
    break
    }
  }
  n.train2 <- n.train - j + 1
  # If the number of time periods within the same regime state is less than the minimum number of
  # time periods, then we use the minimum number of time periods for a training set.
  if (n.train2 > min.n.train) {
    train <- as.data.frame(train[j:n.train,])
  } else {
    train <- as.data.frame(train[(n.train-min.n.train+1):n.train,])
  }
  names(train) <- c("GDPC1")
  # Fitting
  fit <- nnetar(train$GDPC1, p = n.lag, size = n.hidden)
  # Forecasting
  fcast <- forecast(fit, PI = T, h = h)
  # Mean squared prediction error (MSPE)
  out.nn.hmm[i] <- sum((test$GDPC1 - fcast$mean)^2) / h
}


## Benchmarking: Neural network (https://otexts.com/fpp2/nnetar.html)
out.nn <- matrix(NA, nrow = n.fcast)
for (i in 1:n.fcast) {
  train <- data[i:(i+n.train-1),]
  test <- data[(i+n.train):(i+n.train-1+h),]
  # Fitting
  fit <- nnetar(train$GDPC1, p = n.lag, size = n.hidden)
  # Forecasting
  fcast <- forecast(fit, PI = T, h = h)
  # Mean squared prediction error (MSPE)
  out.nn[i] <- sum((test$GDPC1 - fcast$mean)^2) / h
}


## Result
t <- c(1:n.fcast)
model1 <- rep(c("NN with HMM"), n.fcast)
model2 <- rep(c("NN"), n.fcast)
out1 <- data.frame(t, out.nn.hmm, model1)
names(out1) <- c("Time","MSPE", "Model")
out2 <- data.frame(t, out.nn, model2)
names(out2) <- c("Time", "MSPE", "Model")
out <- rbind(out1, out2)

p <- ggplot(out, aes(x=Time, y=MSPE, group=Model)) +
  geom_line(aes(linetype=Model, color=Model, alpha=Model)) +
  geom_point(aes(shape=Model, color=Model, size=Model, alpha=Model)) +
  scale_linetype_manual(values=c("solid","dotted")) +
  scale_color_manual(values=c("black", "blue")) +
  scale_size_manual(values=c(2.5, 2.5)) +
  scale_shape_manual(values=c(15, 16)) +
  scale_alpha_manual(values = c(0.7, 0.7)) +
  xlab('Time') + ylab('MSPE') +
  scale_x_continuous(breaks = c(2,10,18,26,34,42,50,58,66), 
                     labels = c("2004 1Q","2006 1Q","2008 1Q","2010 1Q","2012 1Q","2014 1Q","2016 1Q","2018 1Q","2020 1Q"))
p <- p + theme(legend.title = element_text(size=10, face="bold"))
p <- p + theme(legend.text = element_text(size=10, face="bold"))
p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), plot.margin = margin(1, 1, 1, 1, "cm"),
               panel.background = element_blank(), axis.line = element_line(colour = "black"), panel.border = element_rect(linetype = "solid", fill = NA))
p

mean(out1$MSPE)     # Our approach
mean(out2$MSPE)     # Benchmarking






