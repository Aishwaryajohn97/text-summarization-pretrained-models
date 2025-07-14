**Effect of Pre-Trained Models in Text Summarization for Proceedings**

Project Description

This repository contains the code and resources for a Master's project focused on evaluating the effectiveness of various pre-trained language models (BART, BERT, T5, and GPT-2) for summarizing legal court proceedings. The goal is to create shorter, yet informative, variations of lengthy legal documents to make them more accessible.

Traditional text summarization methods often rely on manual techniques, which can be time-consuming and error-prone. This research explores the use of advanced pre-trained models, trained on vast text and code corpora, to generate more accurate and informative summaries. The project utilizes archival records from the Old Bailey website for its dataset and evaluates the models using ROUGE-1, ROUGE-2, and ROUGE-L metrics for both single-sentence and full-text summarization.

Key Features & Technologies

Text Summarization: Implementation and evaluation of abstractive text summarization techniques.

Pre-trained Models: Application and comparative analysis of BART, BERT, T5, and GPT-2 models.

Dataset: Criminal trial proceedings from the historical Old Bailey website (1674-1913).

Evaluation Metrics: ROUGE-1, ROUGE-2, and ROUGE-L scores to assess summary quality.

Programming Language: Python

Key Libraries: transformers, pandas, numpy, rouge-score, torch, sentencepiece.

Development Environment: Jupyter Notebooks for data processing, model implementation, and evaluation.

Dataset

The dataset used in this project consists of archival records of criminal trial proceedings obtained from the Old Bailey website.

Source: Old Bailey Online

Format: Initially in .xlsx format (or a similar structured format), processed into Data.csv.

Content: Includes date of bail, link to original proceedings, category of the case, and the full proceedings content. The provided Old Bailey Proceedings.xlsx - Sheet1.csv and Data.csv are derived from this source.

Getting Started
Prerequisites
Python 3.8+ (or compatible version for the libraries)

pip (Python package installer)

Installation
Clone the repository:

git clone https://github.com/Aishwaryajohn97/text-summarization-pretrained-models.git
cd text-summarization-pretrained-models

Create a virtual environment (recommended):

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install the required Python libraries:

pip install pandas numpy transformers rouge-score torch sentencepiece

Data Preparation

The raw data from Old Bailey proceedings needs initial processing.

The 1- Old Bailey Proceedings Files collection.ipynb notebook handles the initial data loading (from Old Bailey Proceedings.xlsx or Old Bailey Proceedings.xlsx - Sheet1.csv), performs exploratory data analysis (EDA), drops unnecessary columns (Date, link), and saves the cleaned data as Data.csv.

Before running other notebooks, ensure Data.csv is present in your project directory. You can generate it by running 1- Old Bailey Proceedings Files collection.ipynb.

Running the Models
Each pre-trained model's implementation and evaluation is contained within its respective Jupyter Notebook.
To run the analysis for each model:

Open Jupyter Notebook:

jupyter notebook

This will open a browser window with the Jupyter interface.

Navigate and open the notebooks:

2- Bart.ipynb: Implements and evaluates the BART model.

3 -Bert.ipynb: Implements and evaluates the BERT model.

4 -T5.ipynb: Implements and evaluates the T5 model.

5 -GPT2.ipynb: Implements and evaluates the GPT-2 model.

Run all cells: Inside each notebook, you can run all cells sequentially to perform the summarization and evaluation.

Results & Findings
The research indicates that pre-trained models like BERT and GPT-2 are effective in generating summarized text that retains similar vocabulary to the original Old Bailey court proceedings.

BERT and GPT-2: Achieved nearly identical maximum precision of 100% and 99% respectively.

Recall and F1 Score: Both models showed a recall of 18% and an F1 score of 31%.

Limitation: A significant finding is that while these models excel in word usage similarity, they are less effective at preserving the original word order, suggesting challenges in generating highly coherent and grammatically similar summaries to the source text.

Future Work
Based on the current research, the following areas are recommended for future improvements:

Advanced Evaluation Metrics: Utilize more sophisticated evaluation metrics beyond ROUGE, such as METEOR or CIDEr, to gain a more comprehensive understanding of model efficacy, including fluency and precision of the generated text.

Larger Dataset: Employ a larger and more diverse dataset to enhance model accuracy and allow for more generalizable claims about the performance of pre-trained methods for summarizing proceedings.

Novel Summarization Techniques: Explore and implement new techniques for summarizing proceedings, such as reinforcement learning or neural machine translation, to potentially improve accuracy and overall effectiveness.

Contact
Aishwarya John Pole Madhu
Student ID: 19059835
