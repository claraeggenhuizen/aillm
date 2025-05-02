#This file contains the code to collect and visualize the Reddit data
#It requires a prior authorization by Reddit to access the API as well as a config file with the credentials
#neither of which are on GitHub because they contain my personal information (client secret, username, password, etc)

#importing libraries
import praw
import pandas as pd
import datetime
import time

#setting wd
import os
os.chdir("/Users/Clara/Desktop/M2/AI & LLM/Project")

import config #my custom config file 

#I had some issues with code that I wrote based on the praw documentation and Reddit API documentation 
#so I had Claude do some debugging; it helped mostly with setting up the authentication
#it also wrote error handling

def collect_reddit_data(output_dir="data"):   
    #create authenticated Reddit instance using config
    try:
        reddit = praw.Reddit(
            client_id=config.client_id,
            client_secret=config.client_secret,
            refresh_token=config.refresh_token,
            user_agent=config.user_agent
        )
        
        #verify authentication worked
        username = reddit.user.me().name
        print(f"Authentication successful! Logged in as: {username}")
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        print("Please check your credentials in config.py")
        return None
    
    #subreddits by category
    right_wing_subreddits = [
        "conspiracy", "conservative", "askaconservative", "AskConservatives", "conservatives", 
        "4chan", "politicalcompassmemes", "conservativememes", "conspiracy_commons", "darkenlightenment"
    ]

    mainstream_subreddits = [
        "worldnews", "politics", "news", "InternationalNews", "AnythingGoesNews"
    ]

    left_wing_subreddits = [ 
        "LateStageCapitalism", "antiwork", "Palestine", "socialism", "Socialism101", 
        "Hasan_Piker", "gaza"
    ]
    
    #map subreddits to their category code
    subreddit_categories = {}
    for subreddit in left_wing_subreddits:
        subreddit_categories[subreddit] = 0  #0 = left-wing
    for subreddit in mainstream_subreddits:
        subreddit_categories[subreddit] = 1  #1 = neutral/mainstream
    for subreddit in right_wing_subreddits:
        subreddit_categories[subreddit] = 2  #2 = right-wing
    
    #full list
    all_subreddits = left_wing_subreddits + mainstream_subreddits + right_wing_subreddits
    
    #all search terms
    search_terms = [
        "apartheid state", 
        "two-state solution",
        "ceasefire",
        "boycott israel",
        "settler colonialism",
        "right to return",
        "israeli regime",
        "palestinian resistance",
        "from the river to the sea",
        "kike",
        "heeb",
        "yid",
        "oven dodger",
        "jew rat",
        "greedy jew",
        "jewish cabal",
        "rothschilds",
        "jewish media",
        "globalist jews",
        "jewish bankers",
        "international jewry",
        "jewish supremacy",
        "jewish world order",
        "jewish world domination",
        "holocaust denial",
        "six million lie",
        "hitler was right",
        "jewish parasite",
        "inferior race",
        "jew blood",
        "deep state",
        "globalists",
        "1488",
        "zog",
        "zionist occupied government",
        "zionist jews",
        "jewish lobby",
        "zionist lobby",
        "one-state solution",
        "bds",
        "free palestine",
        "jewish question",
        "(((them)))", 
        "they control the media",
        "jewish money"
    ]
    
    #data list
    all_data = []
    
    #function to process a single subreddit
    def process_subreddit(subreddit_name):
        category_code = subreddit_categories.get(subreddit_name, -1) 
        subreddit = reddit.subreddit(subreddit_name)
        
        for term in search_terms:
            print(f"Searching for comments containing '{term}' in r/{subreddit_name}...")
            
            try:
                post_results = subreddit.search(term, limit=100, sort='new')
                
                for post in post_results:
                    #get comments for each post
                    try:
                        post.comments.replace_more(limit=0)
                        for comment in post.comments.list():
                            if not hasattr(comment, 'body'):
                                continue
                            
                            #check if the comment contains the phrase "i am a bot", ignore if it does
                            if "i am a bot" in comment.body.lower():
                                continue
                            
                            #check if the comment contains the search term
                            if term.lower() in comment.body.lower():
                                comment_data = {
                                    "category_code": category_code,
                                    "subreddit": subreddit_name,
                                    "search_term": term,
                                    "post_id": comment.link_id,
                                    "comment_id": comment.id,
                                    "parent_id": comment.parent_id,
                                    "body": comment.body,
                                    "created_utc": datetime.datetime.fromtimestamp(comment.created_utc),
                                    "score": comment.score,
                                    "permalink": f"https://reddit.com{comment.permalink}",
                                    "url": f"https://reddit.com{comment.permalink}",
                                    "author": str(comment.author) if comment.author else "[deleted]",
                                    "is_comment": True
                                }
                                all_data.append(comment_data)
                    except Exception as e:
                        print(f"Error fetching comments for post {post.id}: {str(e)}")
                
                #avoid rate limiting 
                time.sleep(1)
            
            except Exception as e:
                print(f"Error while searching r/{subreddit_name} for '{term}': {str(e)}")
                #continue with next term/subreddit
    
    #process all subreddits
    print("Processing all subreddits...")
    for subreddit_name in all_subreddits:
        process_subreddit(subreddit_name)
    
    #check if it collected any data
    if not all_data:
        print("No data collected. Please check your search terms and subreddits.")
        return None
    
    #convert to df
    df = pd.DataFrame(all_data)
    
    #summary
    print(f"Collected {len(df)} comments total.")
    print(f"  Left-wing (0): {len(df[df['category_code'] == 0])} comments")
    print(f"  Neutral (1): {len(df[df['category_code'] == 1])} comments")
    print(f"  Right-wing (2): {len(df[df['category_code'] == 2])} comments")
    
    #save to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"reddit_antisemitism_data_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"All data saved to {filepath}")
    
    return df

def main():
    """
    function to execute the data collection
    """
    df = collect_reddit_data()
    
    if df is not None:
        print("\nData collection completed successfully!")
    else:
        print("\nData collection failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

#just to see what we collected in a bit more detail
def analyze_data(df):
    """
    Basic analysis of the collected data.
    
    Args:
        df (pd.DataFrame): The collected data
    """
    if df is None or df.empty:
        print("No data to analyze.")
        return
    
    #make a copy to avoid modifying the original dataframe
    analysis_df = df.copy()
    
    #make sure created_utc is datetime type
    try:
        analysis_df['created_utc'] = pd.to_datetime(analysis_df['created_utc'])
    except Exception as e:
        print(f"Warning: Error converting created_utc to datetime: {e}")
        print("Skipping time-based analysis.")
        do_time_analysis = False
    else:
        do_time_analysis = True
        
    print("\n===== BASIC ANALYSIS =====")
    
    #subreddit category distribution
    if 'category_code' in analysis_df.columns:
        print("\nDistribution by category:")
        category_names = {0: "Left-wing", 1: "Neutral/Mainstream", 2: "Right-wing"}
        category_counts = analysis_df['category_code'].value_counts().sort_index()
        for category_code, count in category_counts.items():
            category_name = category_names.get(category_code, f"Unknown ({category_code})")
            print(f"  {category_name}: {count} entries")
    
    #subreddit distribution
    print("\nDistribution by subreddit:")
    subreddit_counts = analysis_df['subreddit'].value_counts()
    for subreddit, count in subreddit_counts.items():
        print(f"  r/{subreddit}: {count} entries")
    
    #search term distribution
    print("\nDistribution by search term:")
    term_counts = analysis_df['search_term'].value_counts()
    for term, count in term_counts.items():
        print(f"  '{term}': {count} entries")
    
    #posts over time
    if do_time_analysis:
        print("\nEntries by month:")
        analysis_df['month'] = analysis_df['created_utc'].dt.strftime('%Y-%m')
        month_counts = analysis_df['month'].value_counts().sort_index()
        for month, count in month_counts.items():
            print(f"  {month}: {count} entries")

#loading from csv (new session because VScode loves crashing) and analyzing
df = pd.read_csv("/Users/Clara/Desktop/M2/AI & LLM/Project/data/full_df_reddit.csv")
analyze_data(df)

#####VISUALIZING#####
#importing libraries
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np

#Claude did the style elements and debugging
def visualize_reddit_data(df, output_dir="visualizations"):
    """
    visualizations for the Reddit data
    """
    if df is None or df.empty:
        print("No data to visualize.")
        return
    
    #output directory
    os.makedirs(output_dir, exist_ok=True)
    
    #copy to avoid modifying the original dataframe
    viz_df = df.copy()
    
    #datetime check
    try:
        viz_df['created_utc'] = pd.to_datetime(viz_df['created_utc'])
    except Exception as e:
        print(f"Error converting dates: {e}")
        return
    
    #set the style for all plots
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    #set common colors for categories
    category_colors = {0: '#3498db', 1: '#9b59b6', 2: '#e74c3c'}
    category_names = {0: "Left-wing", 1: "Neutral", 2: "Right-wing"}
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    #####1. Simple timeline of all comments
    print("Creating simplified timeline visualization...")
    plt.figure(figsize=(14, 8))
    
    #group by day and count entries
    viz_df['date'] = viz_df['created_utc'].dt.date
    timeline_data = viz_df.groupby('date').size()
    
    #create line chart
    plt.plot(timeline_data.index, timeline_data.values, color='#1f77b4', linewidth=1.5)
    plt.title('Reddit Posts/Comments Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Entries', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'simplified_timeline_{timestamp}.png'), dpi=300)
    plt.close()
    
    ######2. Distribution by subreddit with categories (top 10, sorted from lowest to highest)
    #didn'y work for a long time so had Claude fix it
    print("Creating categorized subreddit distribution visualization...")
    plt.figure(figsize=(14, 10))
    
    #get top 10 subreddits by total post count
    top_subreddits = viz_df['subreddit'].value_counts().nlargest(10).index
    subreddit_data = viz_df[viz_df['subreddit'].isin(top_subreddits)]
    
    #create a crosstab of subreddit vs category
    subreddit_category = pd.crosstab(
        subreddit_data['subreddit'], 
        subreddit_data['category_code'],
        margins=False
    )
    
    #fill in missing categories if needed
    for cat in range(3):  # 0, 1, 2
        if cat not in subreddit_category.columns:
            subreddit_category[cat] = 0
    
    #sort by total count (sum across all categories)
    subreddit_category['total'] = subreddit_category.sum(axis=1)
    subreddit_category = subreddit_category.sort_values('total', ascending=True)  # Changed to ascending=True
    subreddit_category = subreddit_category.drop('total', axis=1)
    
    #reindex to ensure proper category order
    subreddit_category = subreddit_category.reindex(sorted(subreddit_category.columns), axis=1)
    
    #create the horizontal bar chart
    subreddit_category.plot(
        kind='barh', 
        stacked=True, 
        figsize=(14, 10),
        color=[category_colors.get(i, f'C{i}') for i in subreddit_category.columns]
    )
    
    legend_labels = [category_names.get(i, f"Category {i}") for i in subreddit_category.columns]

    #plotting
    plt.legend(legend_labels, title="Category", loc='lower right')
    plt.title('Distribution of Entries by Subreddit (Top 10)', fontsize=16)
    plt.xlabel('Number of Entries', fontsize=14)
    plt.ylabel('Subreddit', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'categorized_subreddit_distribution_{timestamp}.png'), dpi=300)
    plt.close()
    
    ######3. Distribution by search Term with categories
    print("Creating categorized search term distribution visualization...")
    plt.figure(figsize=(14, 10))
    
    #crosstab of search term vs category
    term_category = pd.crosstab(
        viz_df['search_term'], 
        viz_df['category_code'],
        margins=False
    )
    
    #fill in missing categories if needed
    for cat in range(3):  # 0, 1, 2
        if cat not in term_category.columns:
            term_category[cat] = 0
    
    #calculate totals and remove empty entries
    term_category['total'] = term_category.sum(axis=1)
    term_category = term_category[term_category['total'] >= 5] 
    #sort
    term_category = term_category.sort_values('total', ascending=True)  # Changed to ascending=True
    term_category = term_category.drop('total', axis=1)
    term_category = term_category.reindex(sorted(term_category.columns), axis=1)
    
    #plot
    term_category.plot(
        kind='barh', 
        stacked=True, 
        figsize=(14, 10),
        color=[category_colors.get(i, f'C{i}') for i in term_category.columns]
    )
    
    legend_labels = [category_names.get(i, f"Category {i}") for i in term_category.columns]
    plt.legend(legend_labels, title="Category", loc='lower right')
    plt.title('Distribution of Entries by Search Term', fontsize=16)
    plt.xlabel('Number of Entries', fontsize=14)
    plt.ylabel('Search Term', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'categorized_search_term_distribution_{timestamp}.png'), dpi=300)
    plt.close()

visualize_reddit_data(df)

#same as above but now normalized subreddit distribution by user count
def visualize_normalized_subreddit_data(df, subscribers_dict, output_dir="visualizations"):
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    category_colors = {0: '#3498db', 1: '#9b59b6', 2: '#e74c3c'}
    category_names = {0: "Left-wing", 1: "Neutral", 2: "Right-wing"}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    top_subreddits = df['subreddit'].value_counts().nlargest(10).index
    subreddit_data = df[df['subreddit'].isin(top_subreddits)]
    
    subreddit_category = pd.crosstab(
        subreddit_data['subreddit'], 
        subreddit_data['category_code']
    )
    
    #normalize by subscriber count (comments per million)
    for subreddit in subreddit_category.index:
        sub_lower = subreddit.lower()
        if sub_lower in subscribers_dict:
            subscriber_millions = subscribers_dict[sub_lower] / 1000000
            for cat in subreddit_category.columns:
                subreddit_category.loc[subreddit, cat] /= subscriber_millions
    
    #sort
    subreddit_category['total'] = subreddit_category.sum(axis=1)
    subreddit_category = subreddit_category.sort_values('total', ascending=True)
    subreddit_category = subreddit_category.drop('total', axis=1)
    
    #plot
    plt.figure(figsize=(14, 10))
    subreddit_category.plot(
        kind='barh', 
        stacked=True, 
        figsize=(14, 10),
        color=[category_colors.get(i, f'C{i}') for i in sorted(subreddit_category.columns)]
    )
    
    legend_labels = [category_names.get(i, f"Category {i}") for i in sorted(subreddit_category.columns)]
    plt.legend(legend_labels, title="Category", loc='lower right')
    plt.title('Distribution of Entries by Subreddit (Top 10, Normalized per Million Subscribers)', fontsize=16)
    plt.xlabel('Number of Entries per Million Subscribers', fontsize=14)
    plt.ylabel('Subreddit', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'normalized_subreddit_distribution_{timestamp}.png'), dpi=300)
    plt.close()
    
    return

#manually collected user count from top 10 subreddits
subscribers_dict = {
    'worldnews': 45000000,
    'politics': 8800000,
    'news': 30000000,
    'politicalcompassmemes': 585000,
    'conspiracy': 2200000,
    'palestine': 280000,
    'latestagecapitalism': 834000,
    'hasan_piker': 158000,
    'askconservatives': 34000,
    'conservative': 1300000
 }

visualize_normalized_subreddit_data(df, subscribers_dict)

#making selection for manual annotation
#Claude added the stratification and sampling
def create_annotation_file(df, num_samples=500, output_dir="annotation"):
    """
    Creates Excel file for manual annotation
    """
    if df is None or df.empty:
        print("No data.")
        return None

    #output directory 
    os.makedirs(output_dir, exist_ok=True)

    #copy df
    annotation_df = df.copy()
    annotation_df['full_text'] = annotation_df.apply(
        lambda row: row.get('body') or f"{row.get('title', '')}\n\n{row.get('selftext', '')}".strip(),
        axis=1
    )

    #add subreddit category info and stratify
    category_names = {0: "Left-wing", 1: "Neutral", 2: "Right-wing"}
    annotation_df['subreddit_category'] = annotation_df['category_code'].map(
        lambda x: category_names.get(x, "Unknown")
    )
    samples = []
    groups = annotation_df.groupby('subreddit_category')

    #calculate samples per group and sample from each
    n_groups = len(groups)
    base_samples_per_group = num_samples // n_groups
    extra_samples = num_samples % n_groups

    for group_name, group_df in groups:
        group_samples = base_samples_per_group + (1 if extra_samples > 0 else 0)
        extra_samples -= 1 if extra_samples > 0 else 0

        if len(group_df) <= group_samples:
            sampled = group_df
        else:
            sampled = group_df.sample(group_samples, random_state=42)

        samples.append(sampled)
        print(f"Sampled {len(sampled)} items from {group_name}")

    #combine and shuffle
    annotation_df = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

    #select columns
    columns_to_keep = [
        'subreddit', 'subreddit_category', 'search_term',
        'full_text', 'score', 'permalink'
    ]
    for col in ['post_id', 'comment_id']:
        if col in annotation_df.columns:
            columns_to_keep.append(col)

    annotation_df = annotation_df[columns_to_keep]

    #add annotation columns
    annotation_df['israel_critique_antisemitism'] = ''
    annotation_df['traditional_antisemitism'] = ''
    annotation_df['israel_critique_no_antisemitism'] = ''
    annotation_df['notes'] = ''

    #save to excel
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"antisemitism_annotation_{timestamp}.xlsx"
    filepath = os.path.join(output_dir, filename)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        annotation_df.to_excel(writer, sheet_name='Annotation', index=False)
        worksheet = writer.sheets['Annotation']
        worksheet.column_dimensions['D'].width = 100  # full_text
        worksheet.column_dimensions['H'].width = 15   # annotation columns
        worksheet.column_dimensions['I'].width = 15
        worksheet.column_dimensions['J'].width = 15
        worksheet.column_dimensions['K'].width = 30   # notes

    print(f"\nAnnotation file created: {filepath}")

    return filepath

create_annotation_file(df, num_samples=500)
