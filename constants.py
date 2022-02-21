imputation_cols = ['Income', 'Age', 'Educational_Years', 'Spending']
scaling_cols = ['Income', 'Seniority', 'Spending']
seed = 42
train_size = 0.7
prep_cols = ['Age', 'MntWines', 'MntFruits', 'MntMeatProducts',
             'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases']

prep_cols_v2 = ['Age', 'Income', 'Recency' 'MntWines', 'MntFruits', 'MntMeatProducts',
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                'Educational_Years', 'Spending', 'Seniority', 'Cluster']

MODEL_PATH = 'C:\\Users\\Filip\\PycharmProjects\\MarketingCampaignAnalysis\\storage\\{}.json'
EVALUATION_PATH = 'C:\\Users\\Filip\\PycharmProjects\\MarketingCampaignAnalysis\\storage\\evaluation'
default_figsize = (6.4, 4.8)
num_features = 31 # num. features after preprocessing
TARGET = 'Response'
shap_plot_size = (20, 10)
training_file_path =  'C:\\Users\\Filip\\PycharmProjects\\MarketingCampaignAnalysis\\training.yaml'
PARAM_GRID_PATH = 'C:\\Users\\Filip\\PycharmProjects\\MarketingCampaignAnalysis\\storage\\CV\\{}.json'


