XGBoost Ensemble Training for Avazu CTR prediction 
====================================================

Acknowledgments
I learned a lot from this repository "4 Idiots' Approach for Click-through Rate Prediction by Yu-Chin Juan et al", including

- feature engineering: encoding site and app data, adding counting features
- training techniques: training using subsets of the train dataset improved performance

Thank you to the team for sharing their knowledge and expertise .

Libraries required:
- scipy
- sklearn
- XGBoost

Getting started, the initial files are:
- README.md
- preprocess.py
- get_uid_counter.py
- train.py

Solution:
- Ensemble of multiple base models, which are XGBoost classifiers without fine-tuning.
    - Each model making inference on Kaggle achieved log loss between 0.405 ~ 0.42.
    - Ensemble of multiple base models achieved log loss betwee 0.405 ~ 0.412.

Workflow:
1. Prepare train dataset
    - create a data dir under the main dir
    - save train.gz file to the data dir

2. run gen_uid_counter.py, it will...
    - scan the whole train.gz and count each uid's appearance
    - store the counting data into a Counter object
    - save the Counter object under the data dir, named uid_counter.pkl

3. run train.py
    - select training mode by using the -m option: debug, 1, n, full
        - for example, use the following command:
            - python train -m debug
        - mode options:
            - debug: using 1000 samples
            - 1: 4M samples, about 10% of the full train dataset
            - n: 4M * n samples, reading train sequentially
            - full: the whole train dataset
    - before running, make sure uid_counter.pkl is already generated
    - for debug or 1, it generates a pair of clf and encoder
    - for n or full, it generates multiple pairs of clfs and encoders
    - mkdir named clfs under the main dir
    - save model(s) and encoder(s) in clfs

4. Making inference on Kaggle
    - Script is not included in this repo. Please find codes on Kaggle
        https://www.kaggle.com/code/shaochiehlin/avazu-ctrprediction-submission