
#install.packages("ISLR")

library(MASS)
library(ISLR)


# 3.6.2 Simple Linear Regression

fix(Boston)
names(Boston)

attach(Boston)

lm.fit <- lm(medv ~ lstat, data = Boston)
lm.fit <- lm(medv ~ lstat)

lm.fit
summary(lm.fit)

names(lm.fit)
coef(lm.fit)

# 계수 추정치에 대한 신뢰구간 확인
confint(lm.fit)


predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "confidence")
predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "prediction")

plot(lstat,medv)
abline(lm.fit)

abline(lm.fit, lwd = 3)
abline(lm.fit, lwd = 3, col = "red")
plot(lstat, medv, col = "red")
plot(lstat, medv, pch = 20)
plot(lstat, medv, pch = "+")
plot(1:20, 1:20, pch = 1:20)

par(mfrow = c(2, 2))
plot(lm.fit)

plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))

plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))


# 3.6.3 Multiple Linear Regression

lm.fit <- lm(medv ~ lstat + age, data = Boston)
summary(lm.fit)

lm.fit <- lm(medv ~ ., data = Boston)
summary(lm.fit)

#install.packages("car")
library(car)
vif(lm.fit)

lm.fit1 <- lm(medv ~ . -age, data = Boston)
summary(lm.fit1)
lm.fit1 <- update(lm.fit, ~ . -age)



# 3.6.4 Interaction Terms

summary(lm(medv ~ lstat*age, data = Boston))


# 3.6.5 Non-linear Transformations of the Predictors

lm.fit2 <- lm(medv ~ lstat + I(lstat^2))
summary(lm.fit2)

lm.fit <- lm(medv ~ lstat)
anova(lm.fit, lm.fit2)

par(mfrow = c(2, 2))
plot(lm.fit2)

lm.fit5 <- lm(medv ~ poly(lstat, 5))
summary(lm.fit5)

summary(lm(medv ~ log(rm), data = Boston))


# 3.6.6 Qualitative Predictors

fix(Carseats)
names(Carseats)

lm.fit <- lm(Sales ~ . + Income:Advertising + Price:Age, data = Carseats)
summary(lm.fit)

attach(Carseats)
contrasts(ShelveLoc)

