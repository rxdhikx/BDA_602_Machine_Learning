<b> 01/30 </b>

Supervised ML
Has labelled data – needs effort 
(Includes response variable)

Unsupervised ML
Unlabelled data – cheap to obtain
No response variables

Bunch of features  – represent in lower dimensional space
Great when there is a massive amount of unlabelled data

Semi supervised ML
Subset of labelled data with a ton of unlabelled data
Some set of labelled data is given and it learns and makes pattern with the given unlabelled data

Self learning
Leads to AI systems - recommendation system, caption generating systems (all generation systems) 
AI is a system of many ML algos

Feature = variable = column = attribute
Observation = sample = element 

OLS Regression
Closed form solution - solved by linear algebra
Ridge/Lasso regression - ML algorithm
Neural Networks 
Are like black boxes (pilot reference)
Transformers (Advanced neural nw)
Chat GPT
Used in generative AI
Based on neural nw
Could be image generated
Can be obtained via pytorch,  huggingface has examples 

ML applications

Supervised learning -
Classification
Regression - numeric quantities
Forecasting - time series eg.ARIMA (auto regressive integrated moving average) - plane tickets september vs december (seasonality) - yearly trend repeats. It is called non stationary (changes over months) but converges over years so would be called stationary. (as yearly trends are same). Hence two features : stationary & non stationary.
Anomaly Detection - 
=outlier/extreme value
eg.credit limit exceeds, throws anomaly (SVM (support vector machines) algo will be great for this)
Object Detection eg.car finding through CNN (uses YOLO - you only look once)- boxes and identifies/detects
Segmentation - SAM (segment anything model) - cuts out any object in an image with a sigle click, developed by Meta AI. Another eg. Fast SAM
Decision Tree - Visualization for feature selection


Unsupervised Learning -
Clustering (eg organised customers as window shoppers/buyers (no labelled data but we can make with a source of some other data available)
Hierarchical
Centroid
Density Based Clustering (based on dense area of clusters)
Gaussian Mixture

Cmd
Ubuntu run (if wsl installed else skip this step)
Code .


<b> 02/06 - online </b>

Data Structures

Traditional Structured data 
MySQL, PostGreSQL
(indexed, well -labelled, removes duplicates)
Can manage and prevent duplication

Schema: Table Layout (relational DB)

Dynamic Schema: (non-relational DB)
MongoDB
Database without schema. Eg. key-value pairs (json file)
Python dictionary
JSON is serialized format, can be read to many languages

Demerits: Performance affected 

Semi Structured Data 
CSV, jSON, Tab-delineated

	Too many duplicates - useless space
	Too many missing values
	Updating is difficult - time consimung 

Unstructured Data
Web scraping
Text analysis
Images - pixel values
Audio - becuz waves of signal
Video - collection of images playing at a certain frame rate, so unstructured data

EXPLORATORY DATA ANALYSIS (EDA) -

Covariance & Correlations!
Covariance - amount of shared variance between 2 features of dataset
Variance - squared-spread of data
Std. deviation - 
	Cov (x1,x2) : larger it is, more related they are!
	(-infinity to +infinity)
	
	Corr (x1,x2) : normalizes between [-1,+1]
	
	If correlation is -1, what does that mean?
	> if one increase, the other decreases by same proportion
	
Refer Graphs for 1-3: (recorded ppt on canvas)

Negative correlation when?
> if x1 increases, x2 decreases 
Or x1 dec, x2 inc. 
Corr (x1, x2) < 0

Positive correlation when?
>if x1 increases and x2 also increases
(moving in same direction)
Corr (x1, x2) > 0

No correlation, is it possible? When?
>if x1 increases, x2 is unpredictable
No correlation 
Pearson’s coefficient —> Corr (x1, x2) = r x1, x2

Examining Bivariate Relationships
Manifold learning 

PRINCIPAL COMPONENT ANALYSIS

<Missed some, taken SS>

-DOES NOT work for non-linear data/relationships

SVD - SINGULAR VALUE DECOMPOSITION

Generalizes PCA
Applied to non square matrices, multidimensional arrays
Used in Image Processing 

Eg. if you have landscape image


<b> 02/13 - </b>

X squared tests-
F - Test
ANONA
ANCOVA
MANOVA
MANCOVA
 
PDF - probability density function
PMF - 

Descriptive, Inferential 

Probability (y/x =x) 
(conditional probability)

Pandas, Dask, Sklearn

Pandas - dataframe library, memory restricted 
DASK - chunks of pandas df or chunks of numpy arrays

Lazy loading - only load what is necessary 

Moderate data - pandas
Big data - Dask, Polar.s
Sklearn - machine learning workflow - pipelines

Interquartile range
Median = robust measure.

<b> 2/20- </b>

DISTRIBUTED VERSION CONTROL SYSTEM 
(PUSH from local to github. PULL from GIT into local)
(Branch - everyone works on their own branch away from main source, if you make changes in ur own branch, wont affect the main codebase. Isolated.

Source venv/activate/bin

Brew install git
Git --version
Mkdir localrepository # creates new directory
Cd localrepository
 
Git config --global user.name “rxdhikx”
Git config --global user.email “enter github email here.com”

Git clone https://github.com/rxdhikx/BDA602.git

Cd BDA602
Git add * #adds all the files to be tracked

New folder -> linear regression
New file -> linear_regression.ipynb

Git status

Git commit -m “Added a linear regression notebook” 
Git checkout -b new_analysis #create a new branch
Git branch 
Git branch --show-current

Git checkout main #to switch back to “main” branch. Switching between branches
Git checkout new_analysis #switch back to our branch to work on 
Git log
Git status (shows nothing is added to track , so add it now)

Git add *

<change something inside the jupyter nb>
Import numpy as np
<save it>
Git commit -m “Added a line of code” (do it only for the final code to be published to github not for each line - annoying)
Git add *
Git status #shows green modified
Git commit -m “Added code”

Git diff main new_analysis linearregression/linearregression.ipynb

Git push --set-upstream origin new_analysis



<b> 2/27- </b> 


-Linear Regression

Empirical rule

-Decision trees 

Non parametric approach 
Prone to overitting
Multi class classification -> categories 
-> entropy / Gini Index (loss function of decision trees)
Classification & Regression Tree -CART
Categorical Statistical Testing -X squared tests

Entropy = disorder 

Our goal = Reduce the entropy loss function (reduce the chaos to learn patterns in data)

Neural Networks
-hidden layers - form connections that reduce the loss function
Inputs - edges - outputs 

LSTM -
Time series data - sequential 


Transformers - in parallel
introduced first by google 
- read the research paper: “Attention is all you need”

Pytorch - has some of the best transformers (pre-trained model)


df = pd.read_csv(“”)
df.head(10)

Converting strings to datetime:

pd.to_datetime()

“2021-01-23”
“%Y-%m-%d”

“2021-07-31T08:54”
Format - “%Y-%m-%dT%H%M”

Converting datetime to strings:

Df[“date”].dt.strftime



PAPER COMPONENTS:

Abstract - what question are you attempting to answer with this ML approach? Short Summary (1 para)
Introduction - more depth on abstract. Datasets in short? Which ML algo u used? Methodologies in short. (3 paras max)
Methodology: Reasoning/information about your analysis, datasets in brief, methods in brief
EDA - figures,  visualizations, charts, descriptive stats
Analysis - what metrics you used for model selection/ML algorithms in brief & performance
Discussion & Limitations
Concluding Remarks - summarize your results
Appendix - github link

<br>



<b> 3/5/2024- </b>

Actually build ML algo
Make pipelines to include data into the ML algo

Pipelines- 
Helpful for tuning
Streamline the process 
w/ sci-kit learn

Parquet - column-oriented approach
V fast - if u wanna export/import big data

Csv - row-oriented 
Good for editing/mutating data

So both file types have different applications


to_parquet()

>> calculate distance between two points on earth: A (lat,long) to B (lat,long)
Euclidean distance (straight line)
Geodesic (curvature line - like a flight) - used when spherical object is concerned

Export Dataset:

Categorical
One hot encoding
Mutually exclusive columns - no overlapping between groups
Order of category does not matter
Eg weather of the day 
If it is stormy, foggy and sunny will be 0 and stormy = 1
If it is foggy, stormy and sunny will be 0 and foggy = 1

Categorical:				Numeric:
One Hot Encoding			Impute -> Standardize
		
       Both = Linear Regression

Bias and Variance:

Model has High Bias = model is not fitting into training data
Linear Regression is a biased model (too simple)
Model has HIgh Variance = model is fitting into training data too well
It learned the wrong things (noise in the training data) - so not an ideal model - accuracy can drop from 100% to 40% real quick

Cross validation:
Splits training data into folds

Most common - 5 folds
K folds = k number of folds = k folds cross validation

3/12/2024 -

Project Presentation

3/19/2024 - 

hotel_bookings_revised_NEW.csv
Decision-Trees-Intro.ipynb
Hyperparameter-Tuning-Elastic-Nets.ipynb















