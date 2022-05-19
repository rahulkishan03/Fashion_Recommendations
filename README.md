# Fashion_Recommendations
Developed Ensemble of machine learning models for predicting which article of clothing customers will buy in the next 7-day period

## Introduction
H&M Group is a multinational clothing company, whose online store offers shoppers a huge variety of clothes to browse through. As there are so many clothes option to choose from, it acts as a one-stop shopping destination for the customer and eliminates the need to churn to any other shop or website. But, if customer unable to quickly find the product of their interests, it may lead the customers to look for other websites to buy from. So, it is important to recommend relevant product to the customer to enhance their shopping experience and increase the sales of H&M.
In this competition, we need to develop product recommendations based on data from previous transactions, customer and product metadata which will generate top 12 product for each customer which will come to purchase on H&M platform.

In this article, we will discuss the steps which we followed to generate product recommendation, results and key learning from this project.

## Exploratory Data Analysis
Data has its own deficiencies like missing field, skewed data and outliers. If a model is built on such a data then it is going to perform sub-optimal. Exploratory Data Analysis (EDA) is used to anlayze this deficiencies. Moreover, Exploratory Data Analysis (EDA) plays major role in identifying key features, getting insights and understand the pattern in the data.

Following steps are performed for EDA:
1. Finding any missing values and replacing them with appropriate values like mean, median or default values eg. age replaced with mean
2. Checking the datatype and transform it into another datatype if required, e.g. t_dat in transaction data changed to datetime.
3. Univariate analysis - mean, median, min, max, standard deviation etc. for all the columns of the datasets i.e. Customer, transaction_train, article.
4. Bivariate analysis - plotting graph to check for trends. Example: checking purchase based on the age groups, club_member_status, product_type and grament_type etc.
5. Checking for outliers in price using box plot of the price.
6. Finding patterns in the dataset using graphs and checking the behavior of variable with respect to other variables.

## Key Insights from EDA

Key Insights from Univeriate Analysis
1. There is no visible trend in FN, Active and customer age .
2. Only one column out of columns product_type_no, product_type_name; graphi- cal_appearance_no, graphical_appearance_name etc. is required for the analysis as they represent the similar information.
3. Most of the text data variables have numerical variable associate with it. Therefore, dummy creation is not required.
Key Insights from Biveriate Analysis
1. Ladies wear accounts for a significant part of all dresses and Sportswear has the least portion as shown in figure no.1.
2. Jersey fancy is the most frequent garment as shown in figure no.2
3. Most number of customer’s age ranges from 20 to 40.
4. Ladieswear is the most popular group name and Baby/children is the least based on orders.
5. Customers seem to need a little more time when buying products with the same product code but different colors and size.
6. Customer seem to buy similar products within short period of time as shown in figure no.3.

## Methods
We create four models for this Kaggle problem and ensemble method to combine the predictions. In this section, we are going to discuss the model and the steps for implementing them.

### Model based on Learning To Rank
Learning to Rank(LTR)6 is a supervised machine learning algorithm for ranking problems. LTR is extensively used in the search ranking problems. We used the LambdaRank variant of LTR. This algorithm tries to minimize the inversion(incorrect order among pairs of results) in ranking.
Following steps are performed for implementation of Learning to Rank:

### Preprocessing
1. Assign -1 to all the missing age values.
2. Assign ‘unknown’ to fields FN, club_member_status and fashion_news_frequency.
3. Convert fields club_member_status, fashion_news_frequency and postal_code into label type using sklearn label encoder.
4. Create a week field which converts the date into week range from 0 to 104.
5. ‘Bestseller_week’ field is created which assigns the rank to articles according to the number of purchases of the article in that particular week.
6. Negative sampling is created using top 12 products of the week which were not bought by the customers.
7. Create a ‘purchase’ field which is equal to 1 if the article is bought by the customer in the row otherwise the field has value 0.

### Modeling LightGBM
1. LightGBM library is used for implementing LTR.
2. Purchase field is the target value.
3. Column used for model training: price, sales_channel_id, week, bestseller_rank, product_code, product_type_no, graphical_appearance_no,colour_group_code, per- ceived_colour_value_id, perceived_colour_master_id’, department_no, index_group_no, section_no, garment_group_no, age.
4. Model is created such that the model used last 10 weeks data to predict for current purchase.
5. Training data: 14603000,Testing Data: 3650749.
6. Hyperparameters of the model: objective: lambdarank, boosting_type: dart, n_estimators: 200, learning_rate: 0.01

### Key Findings
1. Bestseller_week feature comes out to be the most important feature for this model. 2. Increasing negative sampling doesn’t improve the accuracy much.
Model based on Weekly Trending Articles according to the Purchase
This model is based on the weekly trending articles on the basis of sales to predict the best models.

### Preprocessing
1. Calculate the week for t_dat and week wise sales of each article.
2. Number of articles overall purchased in a week can vary according to the season or trend. To resolve this , instead of using count, the purchase count of the article divided with total number of articles purchased on the week is used as the weekly score.
3. Users usually purchase similar items in nearby weeks to handle. So, we provide the higher weights to recently purchased items. We used the following equation:
25000 + 10000 · e−x·0.2 − 1000 (1) x0.5
x : number of days elapsed from the date Customer A purchased Product B to 2020-09-22
This equation has a decreasing curve with steep drop as visible in figure no.4 which makes it appropriate for this problem. This equation was suggested by Byfone6 in his notebook for this problem.
4. We create purchase dict for the previous customers where we store values for articles which the customer purchased previously. Value for each article for the customer is calculated by summing the score of the article in the week when the customer purchased the article.
5. For new customer, value of article is the sum of weekly scores for the article.

### Modeling
While prediction, customer which has previously purchased, top 12 articles on the basis of score for customer is used. Otherwise, top 12 articles according to the generic score are used as the predictions.
3.2.3 Key Findings
Scaling the sales on a weekly level and giving importance to recent purchases enhances the trending model on Kaggle leaderboard.
 4

### Age Based Modeling
With EDA and intuition, age seems to be a good feature to decide prediction for customers.

### reprocessing
1. First three preprocessing step of trending model3.2 is followed.
2. We divided the customer into age brackets:
[-1. 19). [19. 29), [29, 39), [39, 49), [49, 59), [59, 69), [69, 119).
3. Similar to the trending model3.2, score is calculated for each article in each age bracket.
Modeling
For each customer, top 12 articles of the customer’s age bracket is assigned as the prediction.
3.3.3 Key Findings
Scaling the sales on a weekly level and giving importance to the recent purchase is important.
3.4 GRU4Rec Model Using Recbole Library
GRU4Rec model6 is based on the concept of Session based recommendations with Recurrent Neural Networks (RNN). This RNN based recommendation system is a slight modification to regular RNN in terms of Ranking loss function.


### Preprocessing
1. Convert column article_id to item_id:token and user_id:token to customer_id:token as it is required for recbole model.
2. Filter the data for year greater than or equal to 2020 as data size is too large to handle.
3. Create column timestamp from t_dat for the Recbole model.

### Modeling
1. article_id field is the target value.
2. Column used for model training: item_id:token, user_id:token and timestamp.
3. Consider only customers who have more than 40 purchases.
4. Prediction of model on weekly sales trend3.2 is used where sequential model can’t predict. 5. Training data: 13142700, Testing Data: 1460300.
6. Hyperparameters of the model: epoch: 10, learner: adam, learning_rate: 0.001

### Key Findings
We experimented with different recommendation models BERT4Rec, GRU4Rec and LightFM etc. GRU4Rec gives the best score among all the recommendation model available in Recbole.

### Ensembling
Ensembling is a machine learning technique used to combine the results from multiple base estimators and make more accurate predictions. Ensembling helps in dealing with few issues of base estimators like high variance, low accuracy, noise and bias in features.
Please find the steps performed for ensembling on our best performing four models:
1. Created a framework which takes submission csv and test data to give Mean Average Precision(MAP) 6 for the model.

Model based on Learning To Rank
Model based on Weekly Trending Articles Model based on Age
GRU4Rec Model using Recbole Library
Kaggle Public Leaderboard Score Weight
0.0214 0.24 0.0231 1.01 0.0227 0.86 0.0225 0.74

2. Split the data into train (excluding last week) and validation (last week).
3. Use the generated predictive csv of the model and grid search to identify the best performing weights for each model in terms of the MAP @12.
Running Notebook
System Specification Used • Python: 3.7
• GCP Instance Used: deeplearning-1-vm • Instance Type: m1-megamem-96
Steps for Running Notebook
1. Keep transactions_train.csv, articles.csv and customers.csv in folder data which is in the same folder as the notebooks.
2. Run notebooks LTR.ipynb, Trending.ipynb, Age.ipynb and GRU4Rec.ipynb that will generate submission_ltr.csv, submission_trending.csv, submission_age.csv and submis- sion_reg.csv respectively in the same folder.
3. Run Ensemble.ipynb notebook to generate final submission.csv. For the notebook, you need to generate all the previous mentioned notebook.

## Results
• We are able to create product recommendation model which is ensemble of five models. Please find weights and public leaderboard score of each model in 6 

### Key Insights and Learnings
1. In Learning to Rank model, when we increase the number of negative samples, it improves the performance up to certain number of negative sample only beyond that it remains the same. Moreover, method for negative sampling deeply affects the performance of the model.
2. While using model based on weekly sales trend, you need to normalize your article count with respect to the week to resolve the issue of seasonality and bias in weekly article count.
3. Current purchase trends are more significant than the previous purchase trend. So, appropri- ate weighting method should be applied to give more importance to recent purchases.
4. We experimented with different loss function and optimizer in GRU4Rec model. One of the important findings was that appropriate loss function and optimizer for the problem can enhance performance of the model.
5. Ensembling is a good technique to combine multiple models to improve the performance of the model.

### References
[1] Dong, Xishuang & Chen, Xiaodong & Guan, Yi & Yu, Zhiming & Li, Sheng. (2009). An Overview of Learning to Rank for Information Retrieval. 2009 WRI World Congress on Computer Science and Information Engineering, CSIE 2009. 3. 600-606. 10.1109/CSIE.2009.1090.
[2] Byfone ̇(2022). H&M Trending Products Weekly. Kaggle.https://www.kaggle.com/code/byfone/h-m-trending- products-weekly/notebook.
[3] Hidasi, Balázs & Karatzoglou, Alexandros & Baltrunas, Linas & Tikk, Domonkos.(2016). Session-based Recommendations with Recurrent Neural Networks. Conference paper at ICLR 2016.
[4] Satwale, Sonya. (2016). Mean Average Precision (MAP) For Recommender Systems. https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html.

