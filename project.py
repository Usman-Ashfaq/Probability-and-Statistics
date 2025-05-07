import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

def loadData():
    filePath = "PKcovid.csv"
    if os.path.exists(filePath):
        try:
            data = pd.read_csv(filePath)
            st.success("Data loaded successfully from PKcovid.csv")
            return data
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        st.error("PKcovid.csv not found in the same directory as app.py")
        st.error("Please ensure:")
        st.error("1. The CSV file is in the same folder as your Python script")
        st.error("2. The filename is exactly 'PKcovid.csv' (case-sensitive)")
        return None

def showData(data):
    st.subheader("Raw Dataset")
    st.write(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")
    st.dataframe(data)
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

def plotSingleBarChart(data):
    st.subheader("Single Bar Chart")
    categoricalCols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categoricalCols:
        st.warning("No categorical columns found in the data.")
        return
    column = st.selectbox("Select a column for Bar Chart", categoricalCols, key='singleBar')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=data[column], ax=ax, order=data[column].value_counts().index)
    plt.xticks(rotation=45)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)

def plotMultipleBarChart(data):
    st.subheader("Multiple Bar Chart")
    categoricalCols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categoricalCols) < 2:
        st.warning("Need at least 2 categorical columns for multiple bar chart.")
        return
    cols = st.multiselect("Select 2+ categorical columns", categoricalCols, key='multiBar')
    if len(cols) >= 2:
        grouped = data.groupby(cols).size().reset_index(name='Count')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=cols[0], y='Count', hue=cols[1], data=grouped, ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f"Distribution by {cols[0]} and {cols[1]}")
        st.pyplot(fig)

def plotComponentBarChart(data):
    st.subheader("Component (Stacked) Bar Chart")
    categoricalCols = data.select_dtypes(include='object').columns.tolist()
    if len(categoricalCols) < 2:
        st.warning("Need at least 2 categorical columns for stacked chart.")
        return
    catCol1 = st.selectbox("X-axis category", categoricalCols, key='stackedX')
    remainingCols = [c for c in categoricalCols if c != catCol1]
    catCol2 = st.selectbox("Stack by", remainingCols, key='stackedY')
    stacked = data.groupby([catCol1, catCol2]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    stacked.plot(kind='bar', stacked=True, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title(f"Stacked Distribution of {catCol2} by {catCol1}")
    ax.legend(title=catCol2, bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

def plotHistogram(data):
    st.subheader("Histogram")
    numericCols = data.select_dtypes(include='number').columns.tolist()
    if not numericCols:
        st.warning("No numeric columns found.")
        return
    numCol = st.selectbox("Select numeric column", numericCols, key='hist')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[numCol], bins=20, kde=True, ax=ax)
    ax.set_title(f"Distribution of {numCol}")
    ax.set_xlabel(numCol)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def plotPieChart(data):
    st.subheader("Pie Chart")
    categoricalCols = data.select_dtypes(include='object').columns.tolist()
    if not categoricalCols:
        st.warning("No categorical columns found.")
        return
    catCol = st.selectbox("Select category", categoricalCols, key='pie')
    counts = data[catCol].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Distribution of {catCol}")
    st.pyplot(fig)

def plotBoxPlot(data):
    st.subheader("Box Plot")
    numericCols = data.select_dtypes(include='number').columns.tolist()
    if not numericCols:
        st.warning("No numeric columns found.")
        return
    numCol = st.selectbox("Select numeric column", numericCols, key='box')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=data[numCol], ax=ax)
    ax.set_title(f"Box Plot of {numCol}")
    st.pyplot(fig)

def plotProbabilityDistribution(data):
    st.subheader("Probability Distribution")
    numericCols = data.select_dtypes(include='number').columns.tolist()
    if not numericCols:
        st.warning("No numeric columns found in the data.")
        return
    numCol = st.selectbox("Select numeric column", numericCols, key='probDist')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[numCol], kde=True, stat='density', ax=ax)
    ax.set_title(f"Probability Distribution of {numCol}")
    ax.set_xlabel(numCol)
    ax.set_ylabel("Density")
    st.pyplot(fig)

def plotRegression(data):
    st.subheader("Regression Analysis")
    numericCols = data.select_dtypes(include='number').columns.tolist()
    if len(numericCols) < 2:
        st.warning("Need at least 2 numeric columns for regression.")
        return
    col1 = st.selectbox("Independent Variable (X)", numericCols, key='regX')
    col2 = st.selectbox("Dependent Variable (Y)", [c for c in numericCols if c != col1], key='regY')
    X = data[[col1]]
    y = data[col2]
    model = LinearRegression()
    model.fit(X, y)
    yPred = model.predict(X)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=data[col1], y=data[col2], ax=ax)
    ax.set_title(f"Regression: {col2} ~ {col1}")
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    st.pyplot(fig)
    st.subheader("Regression Results")
    XSm = sm.add_constant(X)
    modelSm = sm.OLS(y, XSm).fit()
    st.text(str(modelSm.summary()))

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: navy;'>Covid Cases Analysis in Pakistan</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    data = loadData()
    if data is None:
        st.stop()

    analysisOptions = [
        "Data Overview",
        "Single Bar Chart",
        "Multiple Bar Chart",
        "Component Bar Chart",
        "Histogram",
        "Pie Chart",
        "Box Plot",
        "Probability Distribution",
        "Regression Analysis"
    ]

    analysis = st.sidebar.selectbox("Choose Analysis", analysisOptions)

    if analysis == "Data Overview":
        showData(data)
    elif analysis == "Single Bar Chart":
        plotSingleBarChart(data)
    elif analysis == "Multiple Bar Chart":
        plotMultipleBarChart(data)
    elif analysis == "Component Bar Chart":
        plotComponentBarChart(data)
    elif analysis == "Histogram":
        plotHistogram(data)
    elif analysis == "Pie Chart":
        plotPieChart(data)
    elif analysis == "Box Plot":
        plotBoxPlot(data)
    elif analysis == "Probability Distribution":
        plotProbabilityDistribution(data)
    elif analysis == "Regression Analysis":
        plotRegression(data)

if _name_ == "_main_":
    main()
