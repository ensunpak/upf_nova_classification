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

The first thing that was observed is that 15 countries make up about 90% of the records in the data, the remaining 10% of data that covers the rest of the countries were excluded. The following chart illustrates the distribution of data by countries where the product data was reported.

