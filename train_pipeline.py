import pandas as pd
import numpy as np


# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima                              # for determining ARIMA orders

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")



def run_training():
    """Train the model."""

    # read training data
    df = pd.read_excel(config.TRAINING_DATA_FILE)

    df = df.groupby(df['Date'].dt.strftime('%m-%Y'))['Count'].sum()
    df.index = pd.to_datetime(df.index)

    # divide train and test

    train = df.iloc[:920]
    test = df.iloc[920:]
    model = SARIMAX(train['Count'],order=(0,1,3),seasonal_order=(1,0,1,12))
    results = model.fit()

    joblib.dump(results, config.PIPELINE_NAME)
 # we are setting the seed here

    # transform the target
    # y_train = np.log(y_train)
    # y_test = np.log(y_test)




if __name__ == '__main__':
    run_training()
