# Movie Box Office Analysis and Prediction
 Kyra Bresnahan, Khuyen Cao, Fareya Ikram, Yeggi Lee
 DS3001 Foundations of Data Science-Team Blockbusters

## 1 INTRODUCTION AND MOTIVATION
In February 2018, the newest movie of the Marvel Cinematic Universe, Black Panther demolished box office records, staying at the top for a consecutive six weeks and bringing in hundreds of millions of dollars. In fact, it currently sits as the highest-rated Marvel movie of all-time with a rating of 97% in Rotten Tomatoes [1]. This unprecedented success leads to the following questions: What makes a movie successful? Which common factors correlate to successful blockbusters or total flops? In this project, we hope to explore and answer some of these questions.

A data model that could run analysis on movie revenue in terms of specific movie traits could create business value. The data analysis would be about identifying patterns, cross-referencing with current data, and improving overall performance. Furthermore, the analysis would have a series of unbiased factors (quantitative and qualitative) that would allow the user to see how these factors influence each other and ultimately the revenue outcome.

## 2 METHODOLOGY
Data Acquisition and Description
The dataset that is being used was taken from data.world and contains over 5000 movie titles spanning across 100 years and 66 countries, with 28 distinct variables, 2399 director names and additional information on the main actors/actresses [2]. This was scraped using a combination of movie metadata from Thenumbers.com and IMDB.com. We understand that this data set is quite small, but the reason why we chose it is because this is the dataset with the most attributes and least missing values we could find. Looking at the dataset, we can see that the majority of our 28 variables has less than 100 missing values out of more than 5000 movies, and even the variable with highest missing values still has more than 80% of its values filled out. While we could have gone directly to IMDB to get larger movie data, we believe that with a less sparse dataset, we will not have to drop or fill in empty rows with an approximate values, thus avoiding possible bias and improving the accuracy of our decision tree model.

## Tools and Methods
The following tools will be used:
Programming Language: Python 3.6
Data Cleaning and Preprocessing: OpenRefine. This will be used initially for cleaning the data such as removing rows with blank values and removing extreme outliers.
Extension library for data analysis: pandas, seaborn (for data correlation heat map), missingno (for missing values visualization), matplotlib (additional plotting and visualization), numpy and scikit-learn (for calculation, decision tree)
Other tools: tableau, plot.ly for visualization
Website building and hosting service: we plan to host our web application on Github Pages (using HTML, CSS+Bootstrap, JavaScript if needed) to remain consistent with having all our code available there.

## Results and Evaluation
After the cleaning is complete, we will build a correlation matrix and determine which attributes give us the highest pearson correlation to determine which factors impact the overall revenue the most. Once the factors with significant correlation are consolidated, the dataset will be split into training and test data in order to make a decision tree. Through this, we will try several combination of attributes in order to minimize gini impurity and find the most accurate decision tree.  We will then visualize the tree, along with additional correlation and relationship we have found. Our hope is that we can put our analysis and visualizations onto a website.
References

## Instruction:
1) clone the project
2) run pip install all the packages needed
3) check out visualization and different type of classification methods in visualization.py
3) run app.py to run the local host of the web application

Again, website is also hosted on heroku at [https://ds3001-blockbuster.herokuapp.com](https://ds3001-blockbuster.herokuapp.com)

[1] Cole, Joe Robert, and Ryan Coogler. “Black Panther.”  (2018) - Rotten Tomatoes, 5 Apr. 2018, www.rottentomatoes.com/m/black_panther_2018/.
[2] Sun, Chaun.  “IMDB 5000 Movie Dataset.”  (2016) - Data.World, 8 Apr. 2018, https://data.world/popculture/imdb-5000-movie-dataset





