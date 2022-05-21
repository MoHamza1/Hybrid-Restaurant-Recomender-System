# Hybrid Recomender Systems

I created a restaurant recomender system with a command line interface which uses machine learning to utilise user preferences and a hybrid approach consisting of collaberative and content based filtering with the K-nearest neighbours shallow learning method.

The scope was limited however proved to be an interesting project to work on.

Running instructions can be found in the `instructions.rtf` file in this repo.

The dataset provided in this repo consists of hand coded features and is significantly reduced from the original size used to find the results below of 300MB to approximately 7MB.
## 1. INTRODUCTION
*Domain of application*   

This RS has been built to recommend restaurants to eat, in any context from the Yelp dataset [6]. There are 166 terms that are within the domain, these can be found in recommender.py .

## 2. *Related work review*  

Hybrid recommender systems solve the main common issues  in  RS’s  namely,  the  cold  start  and  the  sparsity problem  [8], since there is no restriction on how hybrid techniques  can  be  combined  together,  there  is  much variation in researcher’s aproaches, especially apparent in [1]  who’s  results  find  that  using  explicit  previous  user ratings  in  adition  to  those  of  other  users  produce  better results than a single algorithm’s approach.  One content boosted CF approach explained in [7], uses a content-based predictor to create an augmented user profile, fed to a collaborative filter, and produces better recommendations than either of their techniques combined. They used a dataset produced by IMDB, for movie recommendations; that dataset has the benefit of including detailed item descriptions, which the Yelp set does not contain. 

## 3. *Purpose/Aim* 

The program built is one where a user given a user ID can  generate  any  number  of  restaurants  they  have  not reviewed  or  visited  before,  that  are  curated  to  their individual history with places locations rated positively and based on what others with similar taste have also liked. The user can additionally reveal some search parameters where they can also obtain recommendations based on what other users with similar tastes have liked. The user is also shown an  estimation  of  how  suited  each  recommendation  is  to them.  

## 2. METHODS
1. *Data description* 

The files used in the Yelp data set consists of yelp_dataset_business , yelp_dataset_review , yelp_dataset_user and yelp_dataset_covid.  

### *1) Summary of useful features:* 

- Each business contains, a list of associated keywords, its identifier and its review count.  
- Each user has their identifier and review count.  
- Reviews contain user/business identifier, and score. 
- COVID-19 information is useful to establish if the location is open for business. 

These files are extremely large, and so much work has been done to process useful information in order to fit in memory. 

### 2. *Data preparation and feature selection*  

Firstly, we only select businesses that have at least one keyword in our domain of 166 categories. Of those businesses any that have more than one branch have been merged into a single entity since this recommender does not necessarily have to recommend a nearby location; this also makes our User/item-rating matrix denser (91,412 businesses become 62,640 entities without losing user ratings).  

We then only select businesses that are open in spite of COVID-19, then of those that have been reviewed by at least 150 users who have also made at least 100 reviews themselves. We then only select reviews by those users on these businesses from yelp_dataset_review, from and use this as our reduced dataset. The restaurants data is stored as newline separated sequence of business objects in a json file, users are stored as an array of user id’s in a json file. Reviews are stored in a csv file consisting of each user, business ,rating. These files are of a much more manageable size than that of the original dataset. 

### 3. *Hybrid scheme*  

The Hybrid recommender system developed with a feature combination approach, where the actual recommender is a KKN memory based, item based collaborative filter, and the contributing recommender is a basic content-based filter. This scheme was chosen to adapt the procedure in [7], to reflect the differences in the implicit item data in the Yelp dataset as opposed to the IMDB one the authors used, using prefiltering instead of profile augmentation. 

### 4. *Recommendation techniques/algorithms*   

Since  the  data  provided  includes  only  explicit  user ratings, it would be impossible to create a context aware system, there is also have the added benefit of being able to get  meaningful  evaluation  metrics  for  the  algorithms elected. The  collaborative filter was configured to be item based; because  there  are  more  businesses  (61,640)  than  users (52,434). 

### 5. *Evaluation methods* 

The  evaluation  metric  used  for  this  system’s performance is MAE (mean average error) [4], which works by  calculating  the  average  deviation  between  known user/item  scores  and  model predicted  scores  as  shown  in formula 1. 

![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.001.png)

(1) 

The system was tested using 5-fold cross validation on the full dataset, with just the KNN collaberative filtering RS as our baseline. The behaviour of the hybrid RS leads to a different MAE for each user since the CB-filter produces a potentially different tailored set of businesses, which may produce a different overall MAE, and so we must run the CB-filter  for  each  user  and  calculate  the  average  MAE across all users. 

## 3. IMPLEMENTATION
### 1. *Input / Output interface* 

![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.002.png)
![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.004.png)

This RS is built with a command line interface, the on- screen instructions are clearly displayed for the user; once a user provides a valid ID, and the system is prepared, they can fetch any number of recommendations, based on their prior reviews. If any search parameters are configured, the model is retrained, and such if generic recommendations are required later, the user can configure no parameters at all.  

Once a user elects to configure some parameters the list of all categories in the domain are displayed, where they can manually input them and search. Images 1 and 2 show some possible user inputs, when recommendations are made; the user is also shown the predicted match score of that venue 

with them. The user is not able to rate the recommendations given. 

### 2. *Recommendation algorithm* 
1) *Content Based Filtering* 

The contributing recommender analyses the users past likes, and examines the liked businesses’ keywords, we calculate a weighted average of each keyword based on the rating given for each business liked. For each business in the dataset that match the set of all liked keywords and for which the predicted score of that business for that user is high, we can filter out the business we know that user will not like; the smaller dataset is denser, since there is also a requirement that all users in the dataset must have reviewed at least 150 businesses within the domain; the generated predictions are therefore more accurate. 

CB-filters typically use TF-IDF (formula 2), Cosine Similarity (formula 3) or Euclidean Distance in order to give weight to certain keywords, I originally developed this system to use TF-IDF; there were a few issues inherent to this dataset that make this technique redundant in practice. Businesses have at most 6 keywords and it is rare that a business includes more than 2 categories within the domain of 166 keywords, and the rest are typically too general such as “Catering” or “Food” and for that reason when running TF-IDF on the corpus of a given user’s reviews there were positive scores given to business’s that aren’t at all related to the domain; removing these results gives the same output as category filtering on a set of user preferences without the computational overhead; and scoring on term frequencies weighted by score the user assigned is done trivially. 

![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.006.png)

![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.007.png)

![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.008.png)

![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.009.png)

(2) 

TF-IDF would reveal much more insight if the corpus being examined were not the set of business-categories but rather their business-menus for example. The reason feature combination was used is also due to this; as a typical TF- IDF content-based filter on this dataset behaves similarly to a category filter, and so this assists the collaborative filter to produce some more accurate recommendations. Since the scoring mechanism is not that of a typical of a CB-filter, what makes this a hybrid RS is the use of my implementation of CB-filtering as a secondary knowledge  

1. source of user taste to improve the training data of the collaborative filter, and is therefore used as a pre-filtering technique for out CF. 

   2) *Collaberative filtering* 

The technique is the k-NN implementation in the surprise python library. The algorithm works by briefly, for each user $u$, and for each ranked item $i$ ; calculate the for each item in the data set $j$ , $i$ ≠ $j$ the cosine distance (formula 

3) of user preferences of $i,j:$ 

![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.010.png) (3) 

Then Select the top $k$ shortest distances, these are the closest recommendations based on each user ranked business. Then calculate a weighted average of the rating of each set of recommendations based on the rating of the item that elicited those recommendations as the prediction score. We select the top $n$ businesses output by the model as the recommendations for the user. 

![](assets/Aspose.Words.dce530c1-1c23-49d3-ab1d-81bd0021fe61.012.png)

Choosing a value for k is crucial, this is because it directly affects the accuracy of the recommendations produced [2], following the procedures in [2], we can plot the MAE by cross validation of the model on known scores vs model predicted scores for each value k (chart 1); and by the elbow technique [3] select the optimal value value for k. and so the value for k, is 40. 

## 4. EVALUATION
### 1. *Comparison against baseline implementation* 



TABLE I.   5 FOLD CROSS VALIDATION RESULTS

Algorithm/MAE |Fold 1 |Fold 2 |Fold 3 |Fold 4 |Fold 5 
--- |--- |--- |--- |--- |---
Hybrid RS K=40|0.76707176 |0.76705797 |0.76603408 |0.77019474 |0.76530331 
KNN  Baseline |0.81036462 |0.80870895 |0.80884571 |0.80676106 |0.80800455 

Fig. 1.  Table  showing MAE  for the  baseline  algorithm, and the  hybrid system 

We see that our results are similar to those for the hybrid approach produced by Sawant et al. [5], this aproach out  performs  our  baseline  algorithm.  This  is  a  result  of denser matrices when the contributing recomender asserts a smaller dataset for the collaberative filter to populate, and from a novelty perspective the user can expect to receive recommendations  that  are  produced  that  are  first  and formost  similar  to  those  they  already  like,  before collaberative filtering finds others. 

### 2. *Ethical analysis* 

The recommendations made by the system are not easily explainable,  while  the  prefiltered  candidates  can  be explained i.e. you will like business x because you also liked y, the recommendations the collaberative filter produces are not  as  transparent;  the  typical  user  will  not  understand probabilistic models, and may find that being given a score is intrusive, negative public sentiment towards recomender systems  arise  from  a  lack  of  understanding  of  such  ML techniques. User’s may be concerned that their behaviours and  preferences  are  readily  available  to  help  other  users, while their use of the system that produced this dataset, the user may not have been aware that it would now become available  for  researchers  to  use.  One  aproach  used  to aleviate this is to discard all user data except their identifier. The  preprocessing  steps  taken  have  discarded  businesses with fewer than 150 explecit user ratings; this will directly negatively impact their proprieters, one way to account for this  is  to  integrate  an  aditional  recomender  that  uses aditional data such as user location to inject serindipitous recommendations, and add an evaluation metric such as that in  [9] to support up and coming establishments 

## 5. REFERENCES
1. R. Burke, "Hybrid web recommender systems.," *The adaptive web,* pp. 377-408, 2007.  
1. M. B. S. N. a. M. N. S. Bahadorpour, ""Determining optimal number of neighbors in item-based kNN collaborative filtering algorithm for learning preferences of new users."," *Journal of Telecommunication, Electronic and Computer engineering,* vol. 9.3, pp. 163- 167, 2017.  
1. M. A. e. a. Syakur, ""Integration k-means clustering method and elbow method for identification of the best customer profile cluster."," *IOP Conference Series: Materials Science and Engineering,* vol. 336, no. 1, 2018.  
1. F. H. D. O. a. E. Gaudioso, "Evalation of recommender systems," *Expert systems with Applications,* vol. 35, no. 3, pp. 790-804, 2008.  
1. S. Sawant, ""Collaborative filtering using weighted bipartite graph projection: a recommendation system for yelp.".," *Proceedings of the CS224W: Social and information network analysis conference,* vol. 33, 2013.  
1. Yelp, "Yelp Dataset," yelp, 2014. [Online]. Available: http://www.yelp.com/. [Accessed 10 January 2021]. 
1. P. R. J. M. a. R. N. Melville, ""Content-boosted collaborative filtering for improved recommendations."," *Aaai/iaai ,* no. 23, pp. 187-192, 2002.  
1. V. Nikulin, "Hybrid recommender system for prediction of the yelp users preferences.," *Industrial Conference on Data Mining. Springer, Cham,* 2014.  
1. M. C. D.-B. a. D. J. Ge, ""Beyond accuracy: evaluating recommender systems by coverage and serendipity.," in *Proceedings of the fourth ACM conference on Recommender systems*, 2010.  
