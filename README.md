# hierarchical-classification

Supervised learning algorithm, using text processing and k-NN, for the classification of 200k clothing products into hierarchical categories.



TASK DESCRIPTION

We want to predict the categories a product belongs to. Internally we do a mixture of machine learning and human work to do so. Obviously we want to automate this further! Your task at hand is to build a prototype that does so. We have attached you a CSV which contains 200k products in multiple languages along with some text data and labels that have been assigned by humans. Those labels belong to a category tree, which has four levels. Please also find attached a CSV which contains the labels alongside their levels. We want you to split the set into the first 80% products and use that to train a model. Then evaluate your model against the last 20%. Please do report the following metrics:

• percentage that you predicted at least one level 2 label correctly;

• percentage that you predicted at least one level 4 label correctly;

• percentage you did no wrong labeling in level 4 (i.e. you didn’t predict a label which wasn’t in the gold standard).


SOLUTION DESCRIPTION

Features are defined using both title and description of the products after some text processing (e.g. punctuation removal, lowering case, etc). Prediction of test labels is done by using a k-NN strategy: features of a test product are compared with features of all 160000 train products and a score for each train product is produced. The first k train products with higher score are considered: labels from these train products (at each different level) are sorted according to their frequency. Only the most frequent labels are retained and they define the predicted labels for the test product. In particular, only the following labels are retained: the first most frequent label at level 1; the first two most frequent labels at level 2; the three most frequent labels at level 3; the most frequent label at level 4.
The number of nearest neighbours k is selected by running the model with different values of k, for multiple (i.e. four) random selections of the train and test data set, on a smaller scale (i.e. train with 4000 and test with 1000 products), and selecting the one which provides best prediction at level 2 (as the variation of prediction scores at level 4, for different k, is smaller).


RESULTS

The following results are obtained by considering that at level 4 only ONE label is predicted (or alternatively, no label):

• percentage of at least one correct level 2 label: 94.5%;

• percentage of at least one correct level 4 label: 83.3%, where I consider when the label is correctly predicted (e.g. Keilsandaletten = Keilsandaletten) and also when I correctly predict the absence of a label (i.e. no label = no label);

• percentage of no wrong labelling at level 4: 83.3%, which is the same as at the second point, because we are predicting only ONE label at level 4.

By increasing the number of possible predicted labels at level 4 to maximum two labels, the following results are obtained:

• percentage of at least one correct level 4 label: 86.2%, where I consider when the label is correctly predicted (e.g. Keilsandaletten = Keilsandaletten) and also when I correctly predict the absence of a label (i.e. no label = no label);

• percentage of no wrong labelling at level 4: 75.5%.

Reported results is the average obtained for 1000 test products using a train data of 160000 product, for three
different random selections of the data set. Parallelisation is really straightforward, as tests are independent.


CODE

The implementation is done both in R and in Python. An additional C++ code is given to possibly improve the speed of the Python implementation during the learning part (see Python code for details). As the original data file size is more than 70 Mb, only a sample file with 6k (instead of 200k) is here provided. The original data can be found here: https://drive.google.com/file/d/0B8BtI9K-FBCVQjIybUpHZDV4WjQ/view
