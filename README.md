# Telecom Customer Churn Prediction with Logistic Regression

<div align="center">
<img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=yellow"> </img>
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"></img>
</div>

**This is part of the first training course of https://www.datascienceacademy.com.br/ Data Scientist program.**

<center><img src="images/telecom_prediction.jpg"></center><br>

# Problem Context

**Customer Churn** refers to the imminent possibility of a client stop using a company's product or service. It is generally measured as a percentage of the customer base that leave the company during a certain period of time (usually monthly).

A telecom company has hired us to create a prediction model in order to identify these customers and be able to create commercial strategies to retain them. It was requested the use of a **Logistic Regression** algorithm for being a simple and efficient model.

# Exploratory Analysis

<center><img src="images/churn_proportion.png"></center><br>

<center><img src="images/churn_division.png"></center><br>

Some important conclutions from the exploratory analysis:

* The major part of the customers **don't have international plan**.
* Most of the customers **don't have voice mail plan**, as we previously observed.
* The current churn is around **14.5%**, higher than the average of similar companies in the Brazilian sector.
* The **period of time** a customer stay with the services seemed not impact the choice of leaving the company;
* There is a practically constant distribution of customers among the **code areas**;
* 42.4% of the customers with **international plan** are classified as positive churn;
* The churn is greater among customers that don't have a **voice mail plan**;
* In general, the customers that **use more the service** are more prone to leave the company.
* The features `Account length`, `Code area`, `Total day calls`,  `Total evening calls`, `Total night calls`, `Total evening charge`, `Total intl charge`, `Total night charge` presented similar patters between the two classes of churn and thus **don't seem to be good predictors**;
* Especially the features `Internation plan`, `Number voice mail messages`, `Number customer service calls` and `Total day minutes` **showed differentiation** between the data for customers with positive status and the rest of the base.

# Feature Engineering

In order to increase the number of **available features**, we identified the possibility to convert the `Number customer service calls` into a categorical column, called `Use customer service calls` and later we also condensated 4 similar features into one new feature, and then reduced the dimensionality.




