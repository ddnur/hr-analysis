# importing libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
from matplotlib.colors import LinearSegmentedColormap


# functions
def float_plot(df: pd.DataFrame | pd.Series, df_name="Dataframe"):
    """visualises float values with histogram"""

    # neccessary columns
    float_columns = df.drop(columns='id', errors='ignore').select_dtypes(include='float').columns

    # height variable
    h = len(float_columns) // 2 + (len(float_columns) % 2 > 0)

    # creating subplots
    fig, axes = plt.subplots(h, 2, figsize=(15, (h * 5)))
    plt.subplots_adjust(hspace=0.5, wspace=0.7)
    plt.suptitle(f"{df_name}")

    for ax, val in zip(axes.flatten(), float_columns):
        df_pivot = df[val]
        ax.hist(df_pivot, bins=15)
        ax.grid(True)
        ax.set_title(f'Распределение по {val}')
        ax.set_ylabel('Количество пользователей')
        ax.set_xlabel(f'{val}')
        ax.text(-0.2, 0.5,
                df[val].describe().to_string(),
                ha='right',
                va='center',
                fontfamily='monospace',
                bbox=dict(facecolor='lightgrey',
                          edgecolor='black',
                          boxstyle='round,pad=1'
                          ),
                linespacing=1.5,
                transform=ax.transAxes
                )

    # deleting excess subplots
    for ax in axes.flatten()[len(float_columns):]:
        ax.axis('off')

    # diplaying graphs
    plt.show()


def int_plot(df: pd.DataFrame | pd.Series, df_name="Dataframe"):
    """visualises integer values with barplot"""

    # neccessary columns
    int_columns = df.drop(columns='id', errors='ignore').select_dtypes(include='int').columns

    # height variable
    h = len(int_columns) // 2 + (len(int_columns) % 2 > 0)

    # creating subplots
    fig, axes = plt.subplots(h, 2, figsize=(15, (h * 5)))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle(f"{df_name}")

    for ax, val in zip(axes.flatten(), int_columns):
        df_pivot = df[val].value_counts().sort_index()

        ax.bar(df_pivot.index, df_pivot.values)
        ax.grid(True)
        ax.set_title(f'Распределение по {val}')
        ax.set_ylabel('Количество пользователей')
        ax.set_xlabel(f'{val}')
        ax.text(-0.15, 0.5,
                df_pivot.describe().to_string(),
                ha='right',
                va='center',
                fontfamily='monospace',
                bbox=dict(facecolor='lightgrey',
                          edgecolor='black',
                          boxstyle='round,pad=1'
                          ),
                linespacing=1.5,
                transform=ax.transAxes
                )

    # deleting excess subplots
    for ax in axes.flatten()[len(int_columns):]:
        ax.axis('off')

    # diplaying graphs
    plt.show()


def object_plot(df: pd.DataFrame | pd.Series, df_name="Dataframe"):
    """visualises string data with pieplot"""

    # neccessary columns
    str_columns = df.select_dtypes(include='object').columns

    # height variable
    h = len(str_columns) // 2 + (len(str_columns) % 2 > 0)

    # creating subplots
    fig, axes = plt.subplots(h, 2, figsize=(15, (h * 7)))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle(f"{df_name}")

    for ax, val in zip(axes.flatten(), str_columns):
        df_pivot = df.pivot_table(values='id', index=val, aggfunc='count')
        df_pivot.columns = ['count']
        ax.pie(df_pivot['count'],
               labels=df_pivot.index,
               wedgeprops={'linewidth': 1.5, 'edgecolor': 'black'},
               startangle=90,
               autopct='%1.2f%%',
               colors=['#1f77b4',
                       '#d62728',
                       '#2ca02c',
                       'yellow',
                       'cyan',
                       'lightgrey'
                       ]
               )
        ax.set_title(f'Распределение по {val}')
        ax.text(0, -0.2,
                df_pivot.to_string(index=True),
                ha='center',
                va='center',
                fontfamily='monospace',
                bbox=dict(facecolor='lightgrey',
                          edgecolor='black',
                          boxstyle='round,pad=1'
                          ),
                linespacing=1.5,
                transform=ax.transAxes
                )

    # deleting excess subplots
    for ax in axes.flatten()[len(str_columns):]:
        ax.axis('off')

    # diplaying graphs
    plt.show()


def scat_plot(df, float_cols=None, hue=None):
    """scatterplot displaying"""

    # selecting cols if None
    if float_cols is None:
        float_cols = df.select_dtypes(include='float').columns

    # creating list with combinations of features
    num_combs = list(it.combinations(float_cols, 2))

    # variables for size selection
    n_graphs = len(num_combs) * 2
    n_rows = (n_graphs // 2 + (n_graphs % 2 > 0))

    # creating subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 5))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    axes = axes.flatten()

    for (col_num1, col_num2), ax in zip(num_combs, axes):
        sns.scatterplot(data=df, x=col_num1, y=col_num2, ax=ax, hue=hue)

        ax.grid(True)

    # deleting excess subplots
    for ax in axes[len(num_combs):]:
        ax.axis('off')

    # displaying graphs
    plt.show()


def box_plot(df, x):
    """boxplot displaying"""

    # neccessary columns
    num_cols = (df
                .drop(columns=['id', x], errors='ignore')
                .select_dtypes(exclude='object')
                .columns
                )

    # height variables
    h = (len(num_cols) // 2 + (len(num_cols) % 2 > 0))

    # creating subplots
    fig, axes = plt.subplots(h, 2, figsize=(15, h * 5))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    axes = axes.flatten()

    for ax, num_col in zip(axes, num_cols):
        sns.boxplot(y=num_col, x=x, data=df, ax=ax)
        ax.grid(True)
        ax.text(-0.08, -0.25,
                df[num_col].describe().to_string(),
                ha='right',
                va='center',
                fontfamily='monospace',
                bbox=dict(facecolor='lightgrey',
                          edgecolor='black',
                          boxstyle='round,pad=1'
                          ),
                linespacing=1.5,
                transform=ax.transAxes
                )

    # deleting excess subplots
    for ax in axes[len(num_cols):]:
        ax.axis('off')

    # displaying graphs
    plt.show()


def corr_plot(df, method='pearson'):
    """correlation matrix visualisation"""

    # dropping unneccessary column
    df = df.drop(columns='id', errors='ignore')

    # creating colormap for matrix
    cmap = LinearSegmentedColormap.from_list(name='test',
                                             colors=['red', 'white', 'green']
                                             )

    # creating correlation matrix
    try:
        correlation_matrix = df.corr(method=method, numeric_only=True)
    except TypeError:
        correlation_matrix = df.select_dtypes(exclude='object').corr(method=method)

    # creating heatmap for matrix
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap=cmap,
                linewidths=0.5,
                linecolor='black'
                )

    # displaying graph
    plt.show()
