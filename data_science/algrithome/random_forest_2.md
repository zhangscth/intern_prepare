
#机器学习算法之随机森林（Random Forest）
2015-04-23 | tags: pythonrandom forestmachine leaning
机器学习算法之随机森林（Random Forest）

转载请注明出处：BackNode

随机森林作为两大ensemble methods之一，近年来非常火热，本文试图探讨一下其背后原理，欢迎指正！
##Bagging

Bagging方法是ensemble methods中获得用于训练base estimator的数据的重要一环。 正如其名，Bagging方法就是将所有training data放进一个黑色的bag中，黑色意味着我们看不到里面的数据的详细情况，只知道里面有我们的数据集。然后从这个bag中随机抽一部分数据出来用于训练一个base estimator。抽到的数据用完之后我们有两种选择，放回或不放回。

既然样本本身可以bagging，那么feature是不是也可以bagging呢？当然可以！bagging完数据本身之后我们可以再bagging features，即从所有特征维度里面随机选取部分特征用于训练。在后面我们会看到，这两个‘随机’就是随机森林的精髓所在。从随机性来看，bagging技术可以有效的减小方差，即减小过拟合程度。

在scikit-learn中，我们可以很方便的将bagging技术应用于一个分类器/回归器，提高性能：

>>> from sklearn.ensemble import BaggingClassifier
>>> from sklearn.neighbors import KNeighborsClassifier
>>> bagging = BaggingClassifier(n_estimators=50, bootstrap=True,
                                KNeighborsClassifier(), bootstrap_features=True,
                                max_samples=0.5, max_features=0.5)

##Decision Tree

关于决策树，在这里不展开详细探讨，有机会的话另开一篇博客细说。先简单地举一个例子，以下是一棵分类树，决定下班后是否要观看机器学习公开课。

![决策树](http://keepcodingblog.qiniudn.com/DT-1.jpg)

###DT

我们可以看到从根节点开始往下会有分支，最终会走向叶子节点，得到分类结果。每一个非叶子节点都是一个特征，上图中共有三维特征。但是决策树的一个劣势就是容易过拟合，下面我们要结合上文提到的bagging技术来中和一下。
Random Forest

bagging + decision trees，我们得到了随机森林。将决策树作为base estimator，然后采用bagging技术训练一大堆小决策树，最后将这些小决策树组合起来，这样就得到了一片森林(随机森林)。
###OOB

我们看一下详细过程：
![random forest](http://keepcodingblog.qiniudn.com/randomForest-t1.jpg)

###RF

(X[1],Y[1])....(X[n],Y[n])是数据集，我们要训练T棵决策树g[1]....g[t]...g[T]。 每次从数据中有放回地随机抽取size-N'的子数据集D[t]用于训练第t棵决策树g[t]。上图右边的表格中，每一列的*数据是没有被选中用于训练决策树g[t]的数据，我们称之为决策树g[t]的out-of-bag(OOB)样本。为什么要引入这个概念？因为在实际中数据通常是异常宝贵的，按照传统流程我们要将从数据集中分出一部分作为验证集，进而用验证集来调参。在随机森林中既然每棵树都有OOB样本，那我们能不能把它们充分利用起来作为验证集呢？

![OOB](http://keepcodingblog.qiniudn.com/OOB.jpg)

上图中Eoob(G)是整个随机森林的OOB error，G-(X[n])中只含有对于X[n]是OOB样本的树。用OOB error代替验证集错误，在实践中效果非常好，更大的一点好处是节省了验证集数据开销。

###feature importance

在实践中，数据中会有很多多余甚至无关特征，这些特征会严重影响模型的分类/回归效果。在随机森林中我们可以根据它自带的feature importance筛选特征。 如果你了解决策树的话，会知道决策树会根据信息熵逐一选取重要的特征。那么在随机森林中如何计算feature importance呢？核心idea：如果特征i对模型是有利的，那么将第i维特征置换为随机值，将会降低模型的性能。

![feature importance](http://keepcodingblog.qiniudn.com/feature-importance.jpg)

将完整模型的性能减去置换第i维特征后的模型，就得到了第i维特征的重要性。那么问题来了，~~挖掘技术哪家强~~，要评估置换第i维特征后的模型性能，我们岂不是要重新训练并用验证集来评估性能？当然不用，忘了我们的OOB error吗？在随机森林中我们可以用OOB error来衡量模型性能，此处同样可以引入OOB error。

![feature importance with OOB](http://keepcodingblog.qiniudn.com/feature-importance-2.jpg)
##经验之谈：调参

理论上来说，随机森林中树的数目越多，模型的效果就越好，但是计算量也就越大，增加树的数目带来的效果提升程度是递减的。所以选择一个合适的参数就可以了，没有必要为了提升一丁点效果徒增计算量。

在scikit-learn中提供了一个函数GridSearchCV用于各类模型的调参，非常方便：

from sklearn.grid_search import GridSearchCV

最后，感谢coursera上林老师的机器学习算法公开课，本文图片均来自林老师的PPT。

转载请注明出处：BackNode

My zhiFuBao
