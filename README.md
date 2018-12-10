# Arvato_customer_segmentation
Create a customer segmentation report for Bertelsmann Arvato Analytics
 (see also [this Medium post](https://medium.com/@dima806/who-buys-organic-products-in-a-germany-online-shop-a-capstone-project-from-bertelsmann-arvato-5f6c7de58a75) for detailed description)

### Used libraries:


* [`matplotlib`](https://matplotlib.org/)
* [`numpy`](http://www.numpy.org/)
* [`pandas`](https://pandas.pydata.org/)
* [`sklearn`](http://scikit-learn.org/stable/index.html)
* [`tqdm`](https://pypi.org/project/tqdm/)
* [`xgboost`]()
* [`gc`]()
* [`collections`]()
* [`time`]()
* [`datetime`]()
* [`warnings`](https://docs.python.org/3/library/warnings.html)

### Libraries installation with [Pypi](https://pypi.org/)


`pip install matplotlib`

`pip install numpy`

`pip install pandas`

`pip install sklearn`

`pip install xgboost`

`pip install datetime`

`pip install tqdm`

### Motivation for the project

As described in the file provided by organizers,

> “In this project, you will analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. You’ll use unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, you’ll apply what you’ve learned on a third dataset with demographics information for targets of a marketing campaign for the company, and use a model to predict which individuals are most likely to convert into becoming customers for the company. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.”

### Files in the repository

* `Arvato_project_final.ipynb` - jupyter notebook with all details about preprocessing and analysis
* `README.md` - this file

### Summary of the results of the analysis

As a result of analysis of Bertelsmann Arvato data on organics products customers, it appears that the most significant target group of such customers consists of **elderly (1925–1975 years of birth) people with larger-than-average incomes and savings**. The **most interesting and unexpected result** was that a **simple method of selecting the target group (by identifying column values for which their ratio in `CUSTOMERS` dataset was significantly larger than in `AZDIAS` (general population) dataset, appears to be 3.7 times more effective than the clustering-based selection**. In my opinion, this can be a consequence of two possible effects:

* either **`CUSTOMERS`** and/or **`AZDIAS`** datasets are sufficiently biased (for example, **`AZDIAS`** dataset may not represent the whole target population, or **`CUSTOMERS`** dataset may be collected by using a set of very narrowly targeted ads);

* the clustering optimization metric used in [this study](https://medium.com/@shihaowen/investigating-customer-segmentation-for-arvato-financial-services-52ebcfc8501) may be far from being optimal, so alternative metrics such as the [mean Silhouette Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) can be used instead.

Finally, by choosing an appropriate ML model, after approx. 13 hours of the [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) procedure, the obtained [AUC for the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) for the unseen test data was 0.800, very close to the present highest score (0.808) available at [Kaggle competition](https://www.kaggle.com/c/udacity-arvato-identify-customers).

**Some future steps** can also be desirable (beyond the presented “fast-and-dirty” approach):

* **more detailed feature engineering**, by using, e.g., [specific domain knowledge](https://www.project-skills.com/domain-knowledge-important-project-management/) or [mean encoding](https://www.coursera.org/lecture/competitive-data-science/concept-of-mean-encoding-b5Gxv);

* **experimenting with NaN imputation strategies** (at the ML stage, I filled NaN to mean values in every column, other imputation methods, such as generating new **is_NaN** features, imputing with median/negative values, etc.);

* going with **different ML models**, e.g., by using [linear methods](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [SVM()](https://scikit-learn.org/stable/modules/svm.html), [KNN()](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [RandomForest()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [CatBoost()](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db), [neural nets](https://www.tensorflow.org/api_guides/python/nn), with **subsequent [stacking](https://www.quora.com/What-is-stacking-in-machine-learning) of model predictions**;

* using the **more detailed/efficient procedure for parameter tuning**, e.g. by using [hyperopt](https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0) library for [Bayesian optimization](https://sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf).

### Acknowledgements

Nice [work](https://medium.com/@shihaowen/investigating-customer-segmentation-for-arvato-financial-services-52ebcfc8501) of [Shihao Wen](https://medium.com/@shihaowen) about the same subject.
