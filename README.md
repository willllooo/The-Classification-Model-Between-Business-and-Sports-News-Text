# The Classification Model Between Business and Sports News Text

## Project Introduction

This project is a news text classifier based on the BERT model, which can classify given news texts into two categories: "business" or "sports". It also supports sentence-level classification, where each sentence in the input text is classified individually.

## Running Environment

- Python 3.6+
- PyTorch
- Transformers
- Scikit-learn
- Matplotlib
- NLTK

## Dataset

It's a dataset of 2,584 news items from sources and Kaggle, tagged "business" and "sports." We split it 80% into the training set and 20% into the test set.

## Usage

1. Ensure that you have installed all the required Python packages and the Spyder IDE.
2. This project builds a BERT model, but freezes and initializes its parameters, and then builds a new model through linear layer and pooling layer. And this model is saved as 
   model.pt, which is convenient for subsequent calls and saves the cost of each input and output.
3. In Spyder, open the project folder and locate the Python file named proj.py.
4. Right-click on the proj.py file and select "Run File".
5. The program will prompt you to enter a news text in the Spyder IPython console. You can copy and paste or manually enter the news text, then press Enter to confirm.
6. The program will classify the input news text and output the following information in the IPython console:
>* The overall classification result for the entire news text ("business" or "sports")
>* The classification result for each sentence, along with the corresponding probability values
7. If you want to input a new news text for classification, repeat step 5. 
8. The evaluation diagram of the training process of the model is located in  W02_5\code\output\plot\curve. If you want to get this diagram yourself, make sure that the Ipython under the tools category in your Spyder IDE changes from inline to Automatic.
 

## Input Format

The program accepts news texts in any format as input, including plain text or HTML format. The input text length should not exceed 512 characters (the maximum input length for the BERT model).

## Example Output
**Enter a news content:** KARACHI: The Sindh government has decided to bring down public transport fares by 7 per cent due to massive reduction in petroleum product prices by the federal government, Geo News reported.Sources said reduction in fares will be applicable on public transport, rickshaw, taxi and other means of traveling.Meanwhile, Karachi Transport Ittehad (KTI) has refused to abide by the government decision.KTI President Irshad Bukhari said the commuters are charged the lowest fares in Karachi as compare to other parts of the country, adding that 80pc vehicles run on Compressed Natural Gas (CNG). Bukhari said Karachi transporters will cut fares when decrease in CNG prices will be made.

__input: KARACHI: The Sindh government has decided to bring down public transport fares by 7 per cent due to massive reduction in petroleum product prices by the federal government, Geo News reported.Sources said reduction in fares will be applicable on public transport, rickshaw, taxi and other means of traveling.Meanwhile, Karachi Transport Ittehad (KTI) has refused to abide by the government decision.KTI President Irshad Bukhari said the commuters are charged the lowest fares in Karachi as compare to other parts of the country, adding that 80pc vehicles run on Compressed Natural Gas (CNG). Bukhari said Karachi transporters will cut fares when decrease in CNG prices will be made.
redict result: [[0.5909872 0.4090128]], business  
input: KARACHI: The Sindh government has decided to bring down public transport fares by 7 per cent due to massive reduction in petroleum product prices by the federal government, Geo News reported
predict result: [[0.5677481 0.4322519]], business  
input: Sources said reduction in fares will be applicable on public transport, rickshaw, taxi and other means of traveling
predict result: [[0.4808352  0.51916474]], sports  
input: Meanwhile, Karachi Transport Ittehad (KTI) has refused to abide by the government decision
predict result: [[0.50017107 0.4998289 ]], business  
input: KTI President Irshad Bukhari said the commuters are charged the lowest fares in Karachi as compare to other parts of the country, adding that 80pc vehicles run on Compressed Natural Gas (CNG)
predict result: [[0.47800198 0.52199805]], sports  
input:  Bukhari said Karachi transporters will cut fares when decrease in CNG prices will be made
predict result: [[0.50900495 0.49099502]], business  
input:  
predict result: [[0.32871604 0.671284  ]], sports__

In this example, the entire news text is classified as "business", and the classification result for each sentence is also output, along with the corresponding probability values.

##Other Classifiers
We have a separate python file in W02_5\code\classifiers.ipynb, we used Naive Bayes classifier, In the case of TF-IDF and Word2Vec text analysis by SVM, the training results of three classifiers were obtained under this data set. This data is then compared with the model we built to find the model that best fits the data set. The evaluation results and graphs of these three classifiers are in W02_5\code\classifiers.