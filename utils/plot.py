import pandas as pd
import matplotlib.pyplot as plt


def save_graph_tb_log_metrics(first_csv_path, second_csv_path, name_ox, name_oy, loc = 'lower right', pth_save = '../pics/metrics_plot.png'):
    """
    Builds a plot of metric dependencies from two CSV files and saves the image.
    Parameters:
    - first_csv_path (str): Path to the first CSV file (Train).
    - second_csv_path (str): Path to the second CSV file (Validation).
    - name_ox (str): Name for the X axis.
    - name_oy (str): Name for the Y axis.
    - plot (bool): If True, shows the plot.
    - saving (bool): If True, saves the plot to a file named `metrics_plot.png`.
    """
    m_train, m_val  = pd.read_csv(first_csv_path), pd.read_csv(second_csv_path)
    epochs_train, values_train = m_train['Step'], m_train['Value']
    epochs_val, values_val = m_val['Step'], m_val['Value']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, values_train, linestyle='--', color='red', label=f"Train {name_oy}")
    plt.plot(epochs_val, values_val, linestyle='-', color='blue', label=f"Validation {name_oy}")
    plt.xlabel(name_ox)
    plt.ylabel(name_oy)
    plt.legend(loc=loc) 
    plt.grid(True)
    plt.savefig(pth_save, dpi=300)
    print(f"График сохранён в {pth_save}")