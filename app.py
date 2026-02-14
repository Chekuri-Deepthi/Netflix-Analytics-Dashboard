import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix Analytics Pro", layout="wide")

st.title("🎬 Netflix Intelligence Dashboard")
st.markdown("Advanced Business & Data Insights Platform")

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['duration'] = df['duration'].str.extract('(\d+)')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    return df

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

type_filter = st.sidebar.multiselect(
    "Content Type",
    df['type'].unique(),
    default=df['type'].unique()
)

rating_filter = st.sidebar.multiselect(
    "Rating",
    df['rating'].dropna().unique(),
    default=df['rating'].dropna().unique()
)

filtered_df = df[
    (df['type'].isin(type_filter)) &
    (df['rating'].isin(rating_filter))
]

# ---------------- KPI CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(filtered_df))
col2.metric("Movies", len(filtered_df[filtered_df['type']=="Movie"]))
col3.metric("TV Shows", len(filtered_df[filtered_df['type']=="TV Show"]))
col4.metric("Countries", filtered_df['country'].nunique())

st.markdown("---")

# ---------------- MOVIES VS TV SHOWS ----------------
fig1 = px.pie(filtered_df, names='type', title="Content Distribution")
st.plotly_chart(fig1, use_container_width=True)

# ---------------- TOP COUNTRIES ----------------
top_countries = filtered_df['country'].value_counts().head(10)
fig2 = px.bar(top_countries, x=top_countries.index, y=top_countries.values,
              title="Top Countries Producing Content")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- RATING DISTRIBUTION ----------------
fig3 = px.histogram(filtered_df, x='rating', title="Rating Distribution")
st.plotly_chart(fig3, use_container_width=True)

# ---------------- DURATION ANALYSIS ----------------
movies = filtered_df[filtered_df['type']=="Movie"]
fig4 = px.histogram(movies, x='duration', nbins=30,
                    title="Movie Duration Distribution (Minutes)")
st.plotly_chart(fig4, use_container_width=True)

# ---------------- CONTENT GROWTH ----------------
year_data = filtered_df['year_added'].value_counts().sort_index()
fig5 = px.line(x=year_data.index, y=year_data.values,
               labels={'x':'Year','y':'Titles Added'},
               title="Content Added Over Time")
st.plotly_chart(fig5, use_container_width=True)

# ---------------- WORDCLOUD ----------------
st.subheader("Genre Popularity WordCloud")
genres = filtered_df['listed_in'].dropna().str.cat(sep=",")
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(genres)

fig_wc, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wc)

# ---------------- TOP ACTORS ----------------
st.subheader("Top 10 Most Frequent Actors")

actors = filtered_df['cast'].dropna().str.split(',').explode()
top_actors = actors.value_counts().head(10)

fig6 = px.bar(top_actors, x=top_actors.index, y=top_actors.values,
              title="Top Actors Appearing in Content")
st.plotly_chart(fig6, use_container_width=True)

# ---------------- RECOMMENDATION SYSTEM ----------------
st.subheader("🎯 Content Recommendation")

df['description'] = df['description'].fillna("")

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title_input = st.selectbox("Select a Title", df['title'].values)

if st.button("Recommend Similar Content"):
    idx = df[df['title']==title_input].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    st.write(df['title'].iloc[movie_indices].values)

st.success("🚀 Full Data Science Portfolio Project Ready!")
