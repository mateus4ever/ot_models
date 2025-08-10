from matplotlib import pyplot as plt

from src.experiments.algorithm.import_data import Import
from src.experiments.algorithm.primal_functions import Primal

class Signal:
    pass

    @staticmethod
    def signal_alpha(data):
        data = Primal.add_column(data,5)
        for i in range(len(data)):
            try:
                #Bullish Alpha
                if data[i, 2] < data[i - 5, 2] and data[i, 2] < data[i - 13, 2] \
                        and data[i, 2] > data[i - 21, 2] and \
                        data[i, 3] > data[i - 1, 3] and data[i, 4] == 0:
                    data[i + 1, 4] = 1

                # Bearish Alpha
                elif data[i, 1] > data[i - 5, 1] and data[i, 1] > data[i - 13, 1] \
                        and data[i, 1] < data[i - 21, 1] and \
                        data[i, 3] < data[i - 1, 3] and data[i, 5] == 0:

                    data[i + 1, 5] = -1
            except IndexError:
                pass
        return data

    @staticmethod
    def ohlc_plot_bars(data, window):
        sample = data[-window:, ]
        for i in range(len(sample)):
            plt.vlines(x=i, ymin=sample[i, 2], ymax=sample[i, 1],
                       color='black', linewidth=1)

            if sample[i, 3] > sample[i, 2]:
                plt.vlines(x=i, ymin=sample[i, 0], ymax=sample[i, 3],
                           color='black', linewidth=1)

            if sample[i, 3] < sample[i, 2]:
                plt.vlines(x=i, ymin=sample[i, 3], ymax=sample[i, 0],
                           color='black', linewidth=1)

            if sample[i, 3] == sample[i, 0]:
                plt.vlines(x=i, ymin=sample[i, 3], ymax=sample[i, 0] + 0.00003,
                           color='black', linewidth=1.00)

        plt.grid()

    def signal_chart(data, position, buy_column, sell_column, window=500):
        sample = data[-window:, ]
        fig, ax = plt.subplots(figsize=(10, 5))
        Signal.ohlc_plot_bars(data, window)

        for i in range(len(sample)):
            if sample[i, buy_column] == 1:
                x = i
                y = sample[i, position]
                ax.annotate(' ', xy=(x, y),
                            arrowprops=dict(width=9, headlength=11,
                                            headwidth=11, facecolor='green', color='green'))

            elif sample[i, sell_column] == -1:
                x = i
                y = sample[i, position]
                ax.annotate(' ', xy=(x, y),
                            arrowprops=dict(width=9, headlength=-11,
                                            headwidth=-11, facecolor='red', color='red'))


assets = ['EURUSD','USDCHF','GBPUSD','ETHUSD','XAUUSD','USDCAD','BTCUSD','XAGUSD','SP500m','UK100']


# Importing the asset as an array
assets = ['EURUSD','USDCHF','GBPUSD','ETHUSD','XAUUSD','USDCAD','BTCUSD','XAGUSD','SP500m','UK100']
my_data = Import.mass_import(5,'D1')
my_data2 = Import.mass_import(2, 'H1')
# Charting the latest 150 signals
# Signal.signal_chart(my_data, 0, 4, 5, window = 150)

print("hallo")
