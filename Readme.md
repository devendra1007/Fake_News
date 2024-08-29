# Fake News Detection Analysis

This project analyzes a dataset of news articles to detect fake news using a machine learning approach. The code performs the following tasks:

1. **Data Loading**: Reads the dataset from a CSV file.
2. **Text Processing**: Uses TF-IDF vectorization to process the text data.
3. **Model Training**: Trains a Passive Aggressive Classifier on the processed data.
4. **Evaluation**: Evaluates the model's performance and generates visualizations.
5. **Additional Analysis**: Identifies the most important features and generates a word cloud.

## Requirements

To run this code, you'll need to set up a virtual environment and install the required libraries. Here’s how you can do it:

### Setting Up the Virtual Environment

1. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**

   - **On Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **On macOS/Linux:**

     ```bash
     source venv/bin/activate
     ```

3. **Install Required Libraries:**

   Install the libraries using:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

1. **Start an IPython Kernel:**

   Ensure you have Jupyter installed in your virtual environment. If not, install it using:

   ```bash
   pip install jupyter
   ```

   Start an IPython kernel with:

   ```bash
   ipython
   ```


### Description of Outputs

- **accuracy.txt**: Contains the accuracy score of the model.
- **confusion_matrix.png**: Visual representation of the confusion matrix.
- **word_cloud.png**: Word cloud showing the most frequent words in the dataset.
- **top_features.txt**: List of the top 20 important features (words) used by the model.

### Additional Notes

- Ensure that the `news.csv` file is located in the `Data` folder.

Feel free to modify the code as needed for further analysis or additional features.

---
