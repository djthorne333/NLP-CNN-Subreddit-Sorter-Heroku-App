import praw
import numpy as np
import pandas as pd
from praw.models import MoreComments

reddit = praw.Reddit(client_id='FsygEwCxCa9tzXGA1k_KCg',
                     client_secret='9Ni1tYtC0xy4z60cVFKvrf9fkZHy1g', password='Xquack33',
                     user_agent='ye', username='datadave333')

sub_list = ['datascience', 'MachineLearning', 'learnmachinelearning', 'LanguageTechnology', 'deeplearning',
'datasets', 'visualization', 'learnpython', 'rstats', 'statistics',
'AskStatistics', 'cscareerquestions', 'dataengineering', 'dataanalysis', 'MLQuestions', 'neuralnetworks',
'datamining', 'DataScienceJobs', 'datasciencenews', 'DataScienceProjects', 'SQL', 'coding', 'compsci', 'learnprogramming', 'programming',
'Python', 'Rlanguage']








title_list = []
label_list = []
con_top_text_list = []

for sub in sub_list2:
    #title_list = []
    subreddit = reddit.subreddit(sub)
    print('////////////////////////////////////////NEW SUB//////////////////////////////////////', '')
    print('')
    print(sub)
    print('')
    print('/////////////////////////////////////////NEW SUB///////////////////////////////////////', '')
    #categories = ['All', 'Hot', 'Controversial', 'Top', 'Rising']
    time_filters = ["all", "day",  "month", "week", "year"]
    for time in time_filters:
        hot_python = subreddit.controversial(limit=100, time_filter=time)
        for submission in hot_python:
            if not submission.stickied:
                print('###############  NEW POST HEEERE  #########################')
                print('Title: {}, ups: {}, downs: {}, Have we visited?: {}'.format(submission.title,
                                                                                   submission.ups,
                                                                                   submission.downs,
                                                                                   submission.visited))
                title_list = np.append(title_list, submission.title)##############################################
                label_list = np.append(label_list, "%s" % sub)
                #####################comments#################################
                com_array = []
                submission.comment_sort = 'top'
                comments_np = np.array(submission.comments)
                for comment in comments_np:
                    if isinstance(comment, MoreComments):
                        continue
                    if len(comment.body) <= 200 and "https:/" not in comment.body and "[deleted]" not in comment.body \
                            and "Typescript" not in comment.body and ">" not in comment.body \
                            and "[removed]" not in comment.body and "http:/" not in comment.body:
                        com_array = np.append(com_array, comment)
                comments_py = list(com_array)
                for comment in comments_py:
                    if len(comments_py) >= 2:
                        del comments_py[2:]  # grabs only 2 comments or less
                # print(comments) #should be 2 ID's max
                for comment in comments_py:
                    print(20 * '-')
                    print(comment.body)
                    print(len(comment.body))
                    con_top_text_list = np.append(con_top_text_list, comment.body)


print(con_top_text_list)
con_top_labels = np.repeat("%s" % sub, repeats=len(con_top_text_list))
print("label_length:", len(con_top_labels))
con_top_text_list = pd.DataFrame(con_top_text_list)
con_top_labels = pd.DataFrame(con_top_labels)
con_top_comment_df = [con_top_text_list, con_top_labels]
con_top_comment_df = pd.concat(con_top_comment_df, axis=1)
con_top_comment_df.columns = ['text', 'label']
print(con_top_comment_df)





title_list = pd.DataFrame(title_list)
label_list = pd.DataFrame(label_list)
reddit_df = [title_list, label_list]
reddit_df_con = pd.concat(reddit_df, axis = 1)
reddit_df_con.columns = ['text', 'label']
print(reddit_df_con)



#TOPPPPPPPPTPTPTPTTTTTTTTTTTTTTOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP///////////////////////
title_list_top = []
label_list_top = []
top_top_text_list = []

for sub in sub_list2:
    #title_list = []
    subreddit = reddit.subreddit(sub)
    print('////////////////////////////////////////NEW SUB//////////////////////////////////////', '')
    print('')
    print(sub)
    print('')
    print('/////////////////////////////////////////NEW SUB///////////////////////////////////////', '')
    #categories = ['All', 'Hot', 'Controversial', 'Top', 'Rising']
    time_filters = ["all", "day",  "month", "week", "year"]
    for time in time_filters:
        hot_python = subreddit.top(limit=100, time_filter=time)
        for submission in hot_python:
            if not submission.stickied:
                print('###############  NEW POST HEEERE  #########################')
                print('Title: {}, ups: {}, downs: {}, Have we visited?: {}'.format(submission.title,
                                                                                   submission.ups,
                                                                                   submission.downs,
                                                                                   submission.visited))
                title_list_top = np.append(title_list_top, submission.title)##############################################
                label_list_top = np.append(label_list_top, "%s" % sub)
                com_array = []
                submission.comment_sort = 'top'
                comments_np = np.array(submission.comments)
                for comment in comments_np:
                    if isinstance(comment, MoreComments):
                        continue
                    if len(comment.body) <= 200 and "https:/" not in comment.body and "[deleted]" not in comment.body \
                            and "Typescript" not in comment.body and ">" not in comment.body \
                            and "[removed]" not in comment.body and "http:/" not in comment.body:
                        com_array = np.append(com_array, comment)
                comments_py = list(com_array)
                for comment in comments_py:
                    if len(comments_py) >= 2:
                        del comments_py[2:]
                for comment in comments_py:
                    print(20 * '-')
                    print(comment.body)
                    print(len(comment.body))
                    top_top_text_list = np.append(top_top_text_list, comment.body)


print(top_top_text_list)
top_top_labels = np.repeat("%s" % sub, repeats=len(top_top_text_list))
print("label_length:", len(top_top_labels))
top_top_text_list = pd.DataFrame(top_top_text_list)
top_top_labels = pd.DataFrame(top_top_labels)
top_top_comment_df = [top_top_text_list, top_top_labels]
top_top_comment_df = pd.concat(top_top_comment_df,
                               axis=1)
top_top_comment_df.columns = ['text', 'label']
print(top_top_comment_df)


title_list_top = pd.DataFrame(title_list_top)
label_list_top = pd.DataFrame(label_list_top)
reddit_df = [title_list_top, label_list_top]
reddit_df_top = pd.concat(reddit_df, axis = 1)
reddit_df_top.columns = ['text', 'label']
print(reddit_df_top)


full_reddit_df_orig = pd.concat([reddit_df_con, reddit_df_top])
# print(full_comment_df)
full_reddit_df = full_reddit_df_orig.drop_duplicates(subset=['text'], keep=False) #dropping all duplicates
#since we are unsure
pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(full_comment_df, "///////POOOSSTT DROP DUPS/////////")
full_reddit_df = full_reddit_df.sample(frac=1) #shuffle
#print(full_comment_df, "///////POOOSSTT SHUFFFLEE/////////")
full_reddit_df.to_csv(r'C:\Users\Dave\Desktop\datadata\Practice\csgooo\reddit_proj\reddit_titles.csv')





full_comments2_df = pd.concat([con_top_comment_df, top_top_comment_df, full_reddit_df_orig])
# print(full_comment_df)
full_comments2_df = full_comments2_df.drop_duplicates(subset=['text'], keep=False) #dropping all duplicates
#since we are unsure
pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(full_comment_df, "///////POOOSSTT DROP DUPS/////////")
full_comments2_df = full_comments2_df.sample(frac=1) #shuffle
#print(full_comment_df, "///////POOOSSTT SHUFFFLEE/////////")
full_comments2_df.to_csv(r'C:\Users\Dave\Desktop\datadata\Practice\csgooo\reddit_proj\full_comments_titles.csv')

