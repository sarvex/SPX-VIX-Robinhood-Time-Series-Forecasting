library(ggplot2)
library(ggfortify)
library(forecast)
library(lmtest)
library(fUnitRoots)
library(tseries)
library(fpp2)
library(xts)
library(dygraphs)
library(tsbox)
library(fGarch)
library(astsa)
library(dynlm)
source('eacf.R')
source('backtest.R')
source("https://gist.githubusercontent.com/ellisp/4002241def4e2b360189e58c3f461b4a/raw/9ab547bff18f73e783aaf30a7e4851c9a2f95b80/dualplot.R")     


raw_data = read.csv('../data/combined_with_popularity_stats.csv')
ts_data = xts(raw_data[,-1], order.by = as.Date(raw_data[,1]))
ts_data = ts_data["2014/"]
spx_close_ln = diff(log(ts_data$spx_Close))
spx_close_ln = na.omit(spx_close_ln)

# Distribution of SPX Log returns
ggplot(data=as.data.frame(spx_close_ln), aes(x=spx_close_ln)) + 
  geom_histogram(aes(y=..density..), color="black", fill="blue") + 
  stat_function(fun = dnorm, color="red", size=2, args = list(mean = mean(spx_close_ln), sd = sd(spx_close_ln)))
qqnorm(spx_close_ln)
qqline(spx_close_ln, col="red", lw=2)

# Normality Check
skewness(spx_close_ln)
kurtosis(spx_close_ln)
jarque.bera.test(spx_close_ln)

# Autocorrelation in the returns
Acf(spx_close_ln)
Box.test(spx_close_ln, lag=10, type="Ljung")

eacf(spx_close_ln) # 2 order MA, 4 order? Any AR?

# Fit an auto arima model to the returns
auto_fit = auto.arima(spx_close_ln)
summary(auto_fit)
coeftest(auto_fit)
acf(auto_fit$residuals)
Box.test(auto_fit$residuals, lag=10, type="Ljung")
# Arima model not giving white noise residuals

# GARCH analysis
gFit = garchFit( ~ arma(0, 2) + garch(1, 1), data=spx_close_ln, trace=F)
summary(gFit)

gRes = xts(residuals(gFit, standardize=T), order.by = time(spx_close_ln))
# Standardize to get garch residuals

autoplot(gRes)
# Normality check for GARCH Model
jarque.bera.test(gRes) 
skewness(gRes)
kurtosis(gRes)
qqnorm(gRes)
qqline(gRes, col="red", lw=2)

# Reject autocorrelation for GARCH model of SPX Log Returns
Acf(gRes)
Box.test(gRes, lag=506, type="Ljung")

# Plot the volatility over the residuals
plot(residuals(gFit), type="l", ylim=c(-.1, .1))
lines(gFit@sigma.t, col="green", lw=2)
lines(-gFit@sigma.t, col="green", lw=2)

# create new series from the volatility
vol_ts = xts(gFit@sigma.t, order.by = time(ts_data[-nrow(ts_data)]))
combined_xts = merge.xts(vol_ts, ts_data, join='inner')

# compare SPX Log Return volatility w/ VIX charts; will use VIX Cur (30 day) moving forward
vol_comps = cbind(combined_xts$vol_ts, combined_xts$vix_cur_Close, combined_xts$vix_9d_Close, combined_xts$vix_3m_Close, combined_xts$vix_6m_Close)
autoplot(vol_comps)

vol_xts = combined_xts$vol_ts
vix_cur_xts = combined_xts$vix_cur_Close
# Correlation
cor(vol_xts, vix_cur_xts) 
lag2.plot(vol_xts, vix_cur_xts, 8)  
ccf(drop(vol_xts), drop(vix_cur_xts), lag.max = 5)
# Large correlation between SPX Log Return Volatility from GARCH Model and VIX.  They are highest correlated at lag 0, which makes sense. Neither one predicts the other, move in parallel.

# New XTS object with spx ln return vol and vix
spx_vol_vix_xts = cbind(vol_xts, vix_cur_xts)
autoplot(spx_vol_vix_xts)

# Graph shows how closesly related spx ln return vol and VIX are
dygraph(cbind(vol_xts, vix_cur_xts)) %>%
  dyAxis("y", label = "SPX Vol") %>%
  dyAxis("y2", label = "VIX Current", independentTicks = TRUE) %>%
  dySeries("vix_cur_Close", axis = 'y2')

# Add in Robinhood data (starts May 2018)
spx_vol_vix_xts_robinhood = spx_vol_vix_xts['2018-05/']
robinhood_xts = merge.xts(spx_vol_vix_xts_robinhood, abs(ts_data$avg_users_holding_ln_ret), join='inner')

robinhood_xts = na.omit(robinhood_xts)
autoplot(robinhood_xts)

# Only use 2020 Data -- When users started to really pick up
autoplot(na.omit(ts_data$total_users_holding))
robinhood_xts2020 = robinhood_xts['2020/']
autoplot(robinhood_xts2020)

# Correlation between VIX and users holding ln return is significant
cor(robinhood_xts2020$vix_cur_Close, robinhood_xts2020$avg_users_holding_ln_ret) 
lag2.plot(robinhood_xts2020$vix_cur_Close, robinhood_xts2020$avg_users_holding_ln_ret, 8) 
ccf(drop(robinhood_xts2020$vix_cur_Close), drop(robinhood_xts2020$avg_users_holding_ln_ret), lag.max = 10)

# Graph shows how VIX and users holding ln return moves similarily.
dygraph(cbind(robinhood_xts2020$vix_cur_Close, robinhood_xts2020$avg_users_holding_ln_ret)) %>%
  dyAxis("y", label = "AVG Users Holding Ln Ret") %>%
  dyAxis("y2", label = "VIX Current", independentTicks = TRUE) %>%
  dySeries("vix_cur_Close", axis = 'y2') %>%
  dySeries("avg_users_holding_ln_ret", axis = 'y')

# Fitting a lm for spx ln return vol and VIX, want to analyze how residuals are correlating to the avg users holding ln return.
ggplot(robinhood_xts2020, aes(x = vix_cur_Close, y = vol_ts)) + geom_point() + stat_smooth(method=lm, formula = y ~ poly(x, 2, raw = FALSE))

# Model fits best with a 2nd order polynomial
lm_fit = lm(robinhood_xts2020$vol_ts ~ poly(robinhood_xts2020$vix_cur_Close, 2, raw = FALSE))
summary(lm_fit)
plot(lm_fit)
# Model doesn't create white noise residuals
acf(lm_fit$residuals)
Box.test(lm_fit$residuals, lag=10, type="Ljung")

robinhood_xts2020$predicted = predict(lm_fit)
robinhood_xts2020$residuals = lm_fit$residuals

# Plot of predicted vs actual for the polynomial model fit
original_plot = ggplot(robinhood_xts2020, aes(x = vix_cur_Close, y = vol_ts)) +
  geom_point() +
  geom_point(aes(y = predicted), shape = 2)
original_plot

# Get some of the outliers as dates that are influential
cooksd = cooks.distance(lm_fit)
influential_dates = (time(cooksd[(cooksd > 8*mean(cooksd, na.rm=T))]))
influential_dates

# Want to analyze the residuals as squared residuals. This is because the relationship between avg users holding and the residuals should be based off the magnitude of the residual and not the direction.
lm_resid_sq = lm_fit$residuals^2

# Not much autocorrelation of the squared residuals, but definitely not white noise
acf(lm_resid_sq)
Box.test(lm_resid_sq, lag=10, type="Ljung")
autoplot(lm_resid_sq)
autoplot(cbind(robinhood_xts2020$avg_users_holding_ln_ret, lm_resid_sq))

# Check correlation between avg users holding ln returns and residuals squared
cor(lm_resid_sq, robinhood_xts2020$avg_users_holding_ln_ret)
lag2.plot(lm_resid_sq, robinhood_xts2020$avg_users_holding_ln_ret, 5) 
ccf(drop(lm_resid_sq), drop(robinhood_xts2020$avg_users_holding_ln_ret), lag.max = 5)
# Hypothesis is proven correct, there is a lag -1 correlation between Robinhood avg. users holding ln return and the squared residuals of the univariate fit between SPX vol and VIX.

# Fit a new LM with the residuals sq and the lag -1 of users holding
lm_fit2 = dynlm(as.zoo(lm_resid_sq) ~ lag(as.zoo(robinhood_xts2020$avg_users_holding_ln_ret), -1))
summary(lm_fit2)
plot(lm_fit2)

# Check residuals
dynlm_resid = xts(lm_fit2$residuals, order.by = time(robinhood_xts2020['2020-01-03/']))
autoplot(dynlm_resid)
acf(dynlm_resid)
Box.test(dynlm_resid, lag=10, type="Ljung")
# Residuals are close to white noise, good sign that users holding are capturing the noise

# Create new xts object with the lag -1 of users holding, use only dates when this is applicable
zoo_ts = as.zoo(robinhood_xts2020)
zoo_ts$avg_users_holding_ln_ret_lag = lag(zoo_ts$avg_users_holding_ln_ret, -1)
combined_fit_xts = xts(zoo_ts, order.by = time(robinhood_xts2020))
combined_fit_xts = combined_fit_xts['2020-01-03/']

# Fit a LM model now including the users holding lag -1, and the same 2nd order polynomial fit of VIX to predict the SPX ln return vol. 
lm_fit_combined = lm(combined_fit_xts$vol_ts ~ combined_fit_xts$avg_users_holding_ln_ret_lag + poly(combined_fit_xts$vix_cur_Close, 2, raw = FALSE))
summary(lm_fit_combined)
plot(lm_fit_combined)

# Check residuals of this model
lm_combined_resid = lm_fit_combined$residuals
autoplot(lm_combined_resid)
acf(lm_combined_resid)
pacf(lm_combined_resid)
Box.test(lm_combined_resid, lag=10, type="Ljung")
# Not white noise but residuals show some autocorrelation which can be improved upon with a ARMA model.

# Predict vol values using new model. 
combined_fit_xts$predicted_from_combined = predict(lm_fit_combined)
combined_fit_xts$predicted_from_original = robinhood_xts2020$predicted['2020-01-03/']
combined_fit_xts$residuals = lm_fit_combined$residuals

# Create a plot showing how the avg users holding ln return improves the fit between vix and spx ln return vol.
final_plot = ggplot(combined_fit_xts, aes(x = vix_cur_Close, y = vol_ts)) +
  geom_point() +
  geom_point(aes(y = predicted_from_original), shape = 17, color='gray') +
  geom_point(aes(y = predicted_from_combined, color=avg_users_holding_ln_ret_lag), shape = 3) +
  theme(legend.position = c(.05, .95),
      legend.justification = c("left", "top"),
      legend.box.just = "left",
      legend.margin = margin(6, 6, 6, 6),
      legend.title = element_text(size=8)) +
  scale_color_gradient(low="blue", high="red") + labs(col='Lag 1 of users holding')
final_plot

# Now fit an arima model on the VIX using the SPX ln return vol values as regressor values
vix_arima_orignal = auto.arima(combined_fit_xts$vix_cur_Close, xreg = combined_fit_xts$vol_ts)
summary(vix_arima_orignal)
coeftest(vix_arima_orignal)
acf(vix_arima_orignal$residuals)
Box.test(vix_arima_orignal$residuals, type='Ljung')
autoplot(vix_arima_orignal$residuals)
combined_fit_xts$fitted_vals_original = xts(vix_arima_orignal$fitted, order.by = time(combined_fit_xts))

# Fit another arima model on the vix using the predicted values from the lm combining the users holding and the SPX ln return vol as the regressor
vix_arima_combined = auto.arima(combined_fit_xts$vix_cur_Close, xreg = combined_fit_xts$predicted_from_combined)
# Model has better AIC, BIC, simga^2
summary(vix_arima_combined)
coeftest(vix_arima_combined)
# ACF has less autocorrelation, box test is white noise
acf(vix_arima_combined$residuals)
Box.test(vix_arima_combined$residuals, type='Ljung')
autoplot(vix_arima_combined$residuals)
combined_fit_xts$fitted_vals_combined = xts(vix_arima_combined$fitted, order.by = time(combined_fit_xts))

# Plot over actual values of the VIX with both arima model fitted values, showing the improvement of using the lag -1 values of the users holding from the linear model.
ggplot(combined_fit_xts, aes(x = Index, y = vix_cur_Close)) +
  geom_line(aes(y = vix_cur_Close), color='black', size=1.1) +
  geom_line(aes(y = fitted_vals_original, color='red')) +
  geom_line(aes(y = fitted_vals_combined, color='blue')) +
  scale_x_date(date_labels = '%b', date_breaks = '1 month') +
  scale_color_discrete(name = "ARIMA Model", labels = c("Only SPX Vol", "SPX + Robinhood Users")) +
  theme(legend.position = c(.98, .98),
        legend.justification = c("right", "top"),
        legend.box.just = "left",
        legend.title = element_text(size=8))

# Compare backtest of both models
backtest(vix_arima_orignal, combined_fit_xts$vix_cur_Close, xre=combined_fit_xts$vol_ts, h=1, orig = .9*length(combined_fit_xts$vix_cur_Close))

backtest(vix_arima_combined, combined_fit_xts$vix_cur_Close, xre=combined_fit_xts$predicted_from_combined, h=1, orig = .9*length(combined_fit_xts$vix_cur_Close))

