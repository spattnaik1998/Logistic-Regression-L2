Spark code that builds a dictionary that includes the 20,000 most frequent words in the training corpus. This dictionary is essentially an RDD that has words as the keys, and the relative frequency position of the word as the value. For example, the value is zero for the most frequent word, and 19,999 for the least frequent word in the dictionary. 

Use a gradient descent algorithm to learn a logistic regression model that can decide whether a document is describing an Australian court case or not. The model should use L2 regularization; you can change the regularization parameter to determine how the parameter impacts the results and get a sense of the extent of the regularization. Gradient descent is run until the L2 norm of the difference in the parameter vector across iterations is very small. Once the task is completed, print out the five words with the largest regression coefficients. These five words are most strongly related to an Australian court case. 

After training the model, it is time to evaluate it. Use your model to predict which of the test docs are an Australian court case. To get credit for this task, you need to compute the F1 score obtained by your classifier.

Data: gs://metcs777-bucket101/TrainingData.txt, gs://metcs777-bucket101/TestingData.txt
