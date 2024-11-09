# Exploratory Data Analysis

## Global information

- **Train Set:** 241,483 articles (20%, or 48,297 articles, will be used as a validation set)
- **Test Set:** 44,738 articles
- **Features:** 128
- **Categories:** 101 unique categories in both the training and test sets, indicating no category discrepancies.

![image](https://github.com/user-attachments/assets/b6021b11-45ed-48b5-afce-665f1165166f)

There are no nodes without parents or with more than one parent.

### Category Repartition

Category distribution is similar across the training and test sets, although it is notably imbalanced. Each category represents between 0.4% and 4% of the dataset.

![image](https://github.com/user-attachments/assets/0c3dcede-21df-4999-b470-c609cdedc81c)


### Feature Values

When examining the features, we observed that f100 have mean value that differ significantly (e.g., a mean of around 100). To ensure consistent normalization across features, we will reapply normalization and save the processed data as parquet files in the data/processed directory.
![image](https://github.com/user-attachments/assets/c218e02a-7040-4d6e-938c-5ce258e21e23)


### Outlier detection

After normalizing the features, we applied a 2-component PCA to both the training and test sets. Plotting the distributions revealed no obvious outliers or discrepancies between the training and test distributions.


![image](https://github.com/user-attachments/assets/04b72de3-beee-4c14-a6c5-b12180a8a753)


Using a 13-component PCA, we are able to explain 96% of the variance in the dataset, indicating a high level of data coverage.

### Feature correlation with target

Some initial features appear to exhibit a correlation (linear as we used the Pearson correlation) with the target variable, suggesting potential for feature engineering if the nature of these features were available.

![image](https://github.com/user-attachments/assets/f557fdf5-a01c-498e-b035-793b9849b7e6)


# Experimentation and Results

## Algorithm Selection

Given the multi-class classification nature of the task, we have chosen two algorithms:

- **LinearSVC**: This model is suitable given the observed linear correlation in feature relationships. It's efficient and well-suited for high-dimensional data.
- **Neural Network**: To compare performance, we also implemented a 3-layer neural network using TensorFlow.

## Prediction Strategy

Two primary strategies were evaluated:

- **Direct Leaf Node Prediction**: This approach directly predicts the final category (leaf node) for each item, increasing the number of target classes. However, the inherent class imbalance at this level may impact performance.

- **Hierarchical Level-by-Level Prediction**: In this method, predictions are made progressively from the root node to deeper levels, with each subsequent prediction conditioned on the previous. If prediction confidence is low, an item can remain in a non-leaf category for manual handling via a backoffice. This approach offers more control over classification accuracy at each level.

## Training Approach

Given the manageable data size, all data is loaded into memory. For larger datasets, mini-batch training would be necessary. For the neural network, several callbacks were employed:
- **Early Stopping**: Halts training to avoid overfitting.
- **Checkpointing**: Saves model weights at optimal points for potential resumption of interrupted training.
- **TensorBoard Integration**: Monitors training and validation loss progression:

![image](https://github.com/user-attachments/assets/58507d4a-a855-493c-ac56-653e01f7d76a)

## Results

### SVM (Direct Leaf Node Prediction)

The results for the SVM model, predicting the final category, are as follows:

```
Predicting with Support Vector Machine.
66.72% of global accuracy
65.36% of global precision
66.08% of weighted global precision
65.70% of global recall
66.72% of weighted global recall
Best classified category: Class 14, Precision: 95.98%, Recall: 98.96%, Number of elements: 869
Worst classified category: Class 89, Precision: 44.83%, Recall: 28.36%, Number of elements: 596

```

### SVM (Level-by-Level Prediction)

A level-by-level SVM approach was implemented as a proof of concept in the notebook. Results showed that only **39%** of articles reached a leaf node, achieving **68.28% accuracy**. While accuracy improved slightly, the number of confidently classified articles decreased significantly.

### Neural Network Results

The neural network results were as follows:

```
Predicting with Neural Network.
59.87% of global accuracy
56.79% of global precision
57.87% of weighted global precision
56.24% of global recall
59.87% of weighted global recall
Best classified category: Class 98, Precision: 100.00%, Recall: 6.34%, Number of elements: 142
Worst classified category: Class 2, Precision: 0.00%, Recall: 0.00%, Number of elements: 310
```

### Summary and Insights

Across models, accuracy, precision, and weighted precision were similar, indicating minimal impact from class imbalance. The SVM model generally outperformed the neural network, with no classes left entirely unclassified (lowest precision was 44%). The neural network struggled with certain classes, showing 0% precision in some cases.

A deeper look at the confusion matrix could also help understand the mistake made: some categories could be very close in terms of content and thus tolerate errors.

![image](https://github.com/user-attachments/assets/cf4e0b64-a11c-4cc9-968f-2116b09cddef)

## Future Work

While the current dataset is relatively small and anonymous, more in-depth data analysis and feature engineering will be crucial to improve the performance of the model. Here are some potential strategies for future improvement:

- Implement **Grid Search** and **Cross Validation** for hyperparameter tuning to further optimize both models and avoid overfitting
- Try **Ensemble Method** by using SVM and Neural Network outcome and train a new model above these predictions
- Handle imbalanced classes by **oversampling** small classes of **undersampling** the bigger ones.
- **Hierarchical Prediction** in Neural Networks to add more sense to the leaf node.
