# LHL-Final-Project
Using NLP to determine genre of game based on description.

This project uses a multi-label binarizer on the training data to create an array of target genre variables. then using Tf-Idf on the description, and a one-vs-rest format running a linear regression model is used to predict the genres of the games. 

this is trained using a dataset consisting of 40,000 games and their descriptions from a web based game service called Steam. If it is retrained on other text data, provided there are also some type of label(type of product, features, models) a prediction can be made for other industries as well.



raw data can be accessed from: https://www.kaggle.com/trolukovich/steam-games-complete-dataset

3 Notebooks

Reduce_data    --->   takes the steam data found at the above link and extracts only the columns i will be using


Create_pickle  --->   used to create the 3 pickle files that have been already included in the repo


Pipeline       --->   test that i will deploy that uses the pickle files to generate a set of genre tags for a given game description
