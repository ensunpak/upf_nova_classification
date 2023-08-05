## Project description
This is a classification project that aims to classify a food product into a NOVA group. The NOVA food product grading system determines
how processed a food product is, from least processed to ultra-processed, and is used in Brazil and certain European countries. Data used to train the classification model was taken from [openfoodfacts.org](https://world.openfoodfacts.org/), an open-source food classification database. In this project, several machine learning classification algorithms were explored. During the project, it was also discovered that the dataset obtained had imbalanced labels, and the impact of this on the model performance was also observed.

### File organization
_Notebooks_<br>
* 0_extract_data.ipynb - Process the raw dataset obtained from openfoodfacts.org
* 1_data_eda.ipynb - Data clean up, EDA and PCA
* 2_model_training.ipynb - Preparation of data and model development
* 3_tensorflow_logistic_regression.ipynb - Neural network model development (neural network models only)

![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)

## 1.Background
In the modern day of food production, increasingly food products have been ultra-processed, and research has shown that consumption of these types of foods in large quantities is linked to health risks such as obesity, type two diabetes, heart disease, and cancer[1]. A food product is typically considered ultra-processed if it contains ingredients from foods that are processed, then reassembled to create shelf-stable, tasty, and convenient meals[1]. Easy access to such food products with typically lower cost barriers makes the health risk more prevalent in the community.

There is a food classification system called NOVA that was established by a team of nutritionists from the School of Public Health, University of São Paulo, Brazil led by Monteiro and Cannon2. It aims to classify food products into four distinct groups from unprocessed to ultra-processed depending on the type of food ingredients used. Details about the NOVA food grouping can be found in this [resource](https://www.fao.org/3/ca5644en/ca5644en.pdf).

Besides the NOVA food classification system, another food classification system called Nutri-Score aims to classify a food product into five buckets of nutritional quality by its nutritional composition[3].

## 2. Data science use case
The classification of a food product into NOVA groups is not straightforward. The application of using machine learning algorithms to predict the classification of a food product into a NOVA category based on nutritional value and Nutri-Score category based on training data is explored in this project. In this project, available information from the dataset such as Nutri-Score score and grade, food additives, and nutrition content of the food product is used as the features for predicting the NOVA group of a food product.

## 3. Summary
In this project, the large 8GB dataset was processed in batches by first splitting it into smaller files. Exploratory Data Analysis (EDA) was carried out to understand the distribution of NOVA grouping and Nutri-Score grades. The models were developed on a down-sampled dataset in addition to the full dataset to observe the effects of label imbalance on model performance. Food additives and nutrition content were used as features to predict the NOVA grouping of a food product. As 512 unique food additives were found in the dataset, this number was reduced to 120 using Principal Component Analysis (PCA) to reduce the complexity of model development. In the down-sampled dataset, 378 unique additives were reduced to 117.

A few classification models were developed in this model – logistic regression with LBFGS, decision tree, random forest, XGBoost, and logistic regression with a single-layer neural network. Out of all the models considered <indicate best model here> performed the best with an average accuracy of <insert accuracy score here>, while the logistic regression model with LBFGS performed the worst.

Challenges in the project:
1.	Labels in the dataset are imbalanced. Random down-sampling had to be implemented
2.	The dataset was large (8GB), and workarounds in the hardware used in this project had to be implemented
3.	Key feature: additives_tags – has a large dimension of 514. Successfully reduced to 120 by applying PCA with an explained variation of  >= 95%
4.	Key feature: ingredients_tags – has a large dimension of 356,210. Unable to reduce dimension due to hardware limitation, the need to switch to nutrition information as a proxy for NOVA group classification.

## 4. Project details
### 4.1 Large dataset
The original data source in tab-delimited format (TSV) taken from world.openfoodfacts.org was 8GB in size making it hard to process with the MacBook Pro used in this project. The dataset had to be first split into 15 smaller files in the terminal (command below), then each smaller file was read into memory in sequence and then appended into a Pandas dataframe for subsequent use.

```terminal
split -l 200000 en.openfoodfacts.org.products.csv data_
```
Link to data source [here](https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv).

### 4.2 Initial EDA
The initial dataset has 577,084 records with 142 columns of information. As the volume of data to work with is massive, the opportunity to reduce the size of the data will be taken advantage of in this project.

The first observation is that 15 countries make up about 90% of the records in the data, the remaining 10% of data that covers the rest of the countries were excluded. The following chart illustrates the distribution of data by countries where the product data was reported.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_top15_country.png" width="800">

⚠️ For the rest of the EDA task, the insights obtained are from a reduced dataset with only the top 15 countries.

Next, the composition of the NOVA group is examined. The following plots visualize this.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_NOVA_groups.png" width="500"> <img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_NOVA_pie.png" width="320">

Alarmingly, most food products reported fall in NOVA groups 3 and 4 which denotes high to ultra-processed foods.

From reviewing the composition of Nutri-Score grades, it is also alarming to observe that most food products fall in the worst end of the grading system that is, grades C, D, and E.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_nutriscore_grade.png" width="500"> <img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_nutriscore_grade_pie.png" width="320">

The Nutri-Score scores interestingly form a bi-modal distribution as can be seen in the following plot.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_nutriscore_score_hist.png" width="1000">

Zooming into the top 5 countries’ NOVA group composition, all 5 countries have the highest food product in the ultra-processed NOVA group (Group 4), with the United States being the largest and Italy being the smallest. However, the United States also has the most product classified under Group 1which is the unprocessed food product group.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_top5_NOVA_group.png" width="700">
<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_top5_NOVA_group_pct.png" width="700">

Nutri-Score grades show a similar trend to the NOVA group, where most of the top 5 countries reviewed had food products classified under the worse end of the Nutri-Score classification (i.e., grades D and E).

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_top5_nutriscore_grade.png" width="700">
<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_top5_nutriscore_grade_pct.png" width="700">

An analysis of the correlation between Nutri-Score grading and NOVA grouping reveals some proportionate correlation between the two systems. Lower and safer NOVA grouping tends to have a larger number of food products with better Nutri-Score grades, the converse is true: Worst NOVA grouping tends to have a major proportion of food products with very poor Nutri-Score grades.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_NOVA_by_nutriscore_grade.png" width="700">

This can be seen clearer in the following chart displaying the percentage composition of each Nutri-Score grade within each NOVA group.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_NOVA_by_nutrigrade_pct.png" width="700">

### 4.3 Feature engineering
Out of the many features identified, two features contained a list of values that could be extracted to form new features. These are the additives and ingredients features for each food product. Each feature contains a list of additives and ingredients used in a product specifically. These were extracted and each unique additive and ingredient then become new features to describe the NOVA group assigned to the particular food product. The new set of additive features would be structured similarly to one-hot-encoding, where each new additive feature would contain a binary value of 1 or 0, whether that particular additive compound is present in the food product or not.

The ingredients feature had to be excluded from the analysis because there were 356,210 unique ingredients or features that were too large to be of practical value during model development. 

### 4.4 Features with high dimensions
Two features were engineered which resulted in a new number of features:

Ingredients: From one feature to 356,210 features
Additives: From one feature to 512

As mentioned earlier, the new  set of features engineered from ingredients was too large and was discarded. To reduce complexity in model development, the number of new features derived from the original additive feature was reduced using PCA. From 512 new additive features, PCA revealed that only 120 features were needed as these were able to explain 95% of the variability in the feature set. While performing PCA, the features were not standardized as they were binary values so they would not affect the outcome of the PCA.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_PCA_full_dataset.png" width="700">

Similarly in the down-sampled dataset, the new set of additive features was also reduced using PCA. 378 unique additive features were reduced to 117.

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_PCA_downsampled_dataset.png" width="700">

### 4.5 Imbalanced labels in the dataset
The reduced dataset with the top 15 countries exhibited an imbalance in the label which is the NOVA grades. To address this, a version of the dataset will be randomly down-sampled to ensure that the composition of the labels is equal in the dataset.

| NOVA Group | Records | Records (%) |
| ---------- | ------- | ----------- |
| 1          | 59,233  | 10%         |
| 2          | 22,187  | 10%         |
| 3          | 116,627 | 10%         |
| 4          | 316,684 | 10%         |

The population in NOVA Groups 1, 3, and 4 will be randomly down-sampled close to NOVA Group 2 at 22,000 samples. The sampling was performed without any replacement.

## 5. Classification models
To classify the food product in the NOVA groups based on the food product’s features such as its ingredients, additive compounds, Nutri-Score grade and score as well as energy values, the following out-of-the-box machine learning algorithms were selected for this project:
1.	Logistic regression (Minimizer = LBFGS)
2.	Decision tree classifier
3.	Random forest classifier
4.	XGBoost
5.	Neural network logistic regression (single layer, SoftMax)

These models will be trained through cross-validation with 5 K-Folds on both the full dataset (with an imbalanced label) and the down-sampled dataset (with a balanced label) to observe the effect imbalanced labels have on model performance. The neural network model will be trained similarly on both datasets for a maximum of 100 epochs. In addition, different gradient descent optimizers will also be explored at a learning rate of 0.001 such as Stochastic Gradient Descent (SGD), Adam, and Adagrad.

### 5.1 Model performance
The models were trained and the 5-fold cross-validated accuracy metric of their performance is recorded in the table below:

| ML Algorithm | Full dataset (1) | Down-sampled dataset (2) | Accuracy difference (1) vs (2) | Training speed (Fast/Moderate/Slow) | 
| ------------ | :--------------: | :----------------------: | :----------------------------: | :---------------------------------: |
| Logistic regression (LBFGS) | 75.54% | 79.98% | -4.44% | Slow |
| Decision tree | 85.45% | 84.11% | +1.34% | Fast |
| Random forest | 89.31% | 88.59% | +0.72% | Moderate |
| XGBoost | 85.50% | 88.14% | -2.64% | Moderate |

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_model_performance_cv_full.png" width="900">
<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_model_performance_cv_downsampled.png" width="900">
<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_model_performance_cv_avg_full.png" width="550">
<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_model_performance_cv_avg_downsample.png" width="550">

Here, the set of models trained on both datasets had the same performance levels. The logistic regression model performed the worst among the models selected, and the random forest classifier was the best performer.

Single layer neural network: neurons = 4; layer activation = SoftMax; epochs = 100; optimizer learning rate = 0.001; momentum = 0.005 (for SGD)
| Optimizer | Train accuracy (Full dataset | Validation accuracy (Full dataset) | Train accuracy (Down sample dataset) | Validation accuracy (Down sample dataset) | 
| ------------ | :--------------: | :----------------------: | :----------------------------: | :---------------------------------: |
| SGD | 67.78% | 67.66% | 55.60% | 55.93% |
| Adam | 73.34% | 73.32% | 77.41% | 77.38% |
| Adagrad |74.11% | 74.13% | 73.32% | 73.13% |
| Adagrad (With L2 regularization) | 71.71% | 71.69% | 75.88% | 75.90% |

Model Performance - SGD optimizer (full dataset)

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_nn_sgd_full.png" width="800">

Model Performance - Adam optimizer (full dataset)

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_nn_adam_full.png" width="800">

Model Performance - Adagrad optimizer (full dataset)

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_nn_adagrad_full.png" width="800">

Model Performance - Adagrad optimizer (full dataset, with L2 regularization)

<img src="https://github.com/ensunpak/upf_nova_classification/blob/main/img/chart_nn_adagrad_l2_full.png" width="800">

Between the four neural network models, the model without any regularization with Adagrad optimizer gave the best result, the train and test loss and accuracy across the training epochs were stable
