# Topic-Modeling
LDA/NMF
""

Topic tree generation is in 2 parts:
1. We determine patterns in the users tweets using topic modeling 
2. Using disambiguity in wiki and the first two hierarchies in DMOZ to build the topic tree model deserved 

The code below is just implementation of the topic modeling and generating the patterns which is stored in csv file 
and later used in wiki and DMOZ. It takes only 3 mins to run on the 28 million rows 

We have 3 functions in the code: 
1. db() : this is a databsed used to stored large number of cleaned tweets since we cannot use pickle to serialize(only 
used on small datasets). We then read directly these values into pandas tables. The main reason for working with pandas
is to avoid loops which slows down the program.

2. create_dict() : transforms pandas table data into dictionary which is passed to the 3 function. It takes approximately
22 mins to execute. When done it stored in a pickle file which can be used instead of executing the whole process again.

3. creating_dataframe(): is a bad choice of function name but all is does is take the dictionary in step 2 apply 
all preprocessing steps(stopwords, stemming, etc) required for implementing the topic models. Then passed to the models 
to generate the patterns in documents. 

Two models are being useed here the LDA and NMF. I did this to check which one actually gives a good pattern recognition
to texts. The output is then read on to a csv file which can be used to building the topic tree

"""
####################################### LABELS #######################################################################

Below code predicts the label for a topic after genetrating the Topic Model using LDA/NMF/Genism: 

How it works: 
1. It takes the topic model above, searchs for candidate labels using Wikipedia and return the top 8 labels which 
   serves as the primary candidates for labeling. 
2. We then create bigrams of these candidate labels and using post-tags we search through Wikipedia again to create 
   secondary labels for them. Most of these bigrams are just noise and we eliminate most of them out by using RACO(
   Related Article Conceptual Overlap) and Dice's coefficient to calculate the average intersection between two labels 
   in the same category. 
3. We then assign labels to tweets by selecting features which best describe the tweets really well. This is done by 
   searching for most frequent words which occurs in topic labels and averaging them and returning the max label.


   NB: I realized most of the bigrams were returning null values so I decided to comment them out and just focused on the 
   labels returned. Feel free to go ahead and mess around with the code.

   signed - Jeremy Johnson
########################################################################################################################
"""


"""
