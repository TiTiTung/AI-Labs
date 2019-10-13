library(tree)
library(ISLR)
attach(Carseats)
# High本身沒有意義，為衡量sales高低所建立
High = ifelse(Sales <=8,"No","Yes")
Carseats = data.frame(Carseats ,High)
Carseats %<>% as.tibble()


# use the tree() to fit a classification tree
tree.carseats = tree(High ~. -Sales ,Carseats)
summary(tree.carseats)
plot(tree.carseats )
text(tree.carseats ,pretty =0)

set.seed (2)
train=sample(1:nrow(Carseats), 200)
Carseats.test=Carseats [-train ,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred ,High.test)
(86+57) /200

set.seed (3)
cv.carseats =cv.tree(tree.carseats ,FUN=prune.misclass )
names(cv.carseats )

par(mfrow=c(1,2))
plot(cv.carseats$size ,cv.carseats$dev ,type="b")
plot(cv.carseats$k ,cv.carseats$dev ,type="b")

prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats )
text(prune.carseats,pretty=0)

tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred ,High.test)
(94+60) /200










