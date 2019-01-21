
import requests
from collections import defaultdict
from bs4 import BeautifulSoup
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import operator
from helpers import *

sns.set()

# overall horror listing
# find top 200 and take only those released after 1980
# url = "https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&explore=title_type,genres"
# url = "https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=51&explore=title_type,genres&ref_=adv_nxt"
# url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=101&explore=title_type,genres&ref_=adv_nxt'
# url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=151&explore=title_type,genres&ref_=adv_nxt'
# url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=201&explore=title_type,genres&ref_=adv_nxt'
url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=251&explore=title_type,genres&ref_=adv_nxt'

include_list_url, include_list_title, year, title_text = getTitle_getURL(url)

######################################
movies_dict = dict.fromkeys(include_list_title)
for i, movie in enumerate(include_list_url):
    movies_dict[include_list_title[i]] = onemoviestats(movie)

# movies_dict_backup = movies_dict.copy()
n = include_list_title.index("Shutter")
include_list_title = include_list_title[n:]
include_list_url = include_list_url[n:]

movies_dict_backup.update(movies_dict)

df = pd.DataFrame.from_dict(movies_dict_backup, orient='index')
#######################################
# supplemental information
# optional
# cast_size = dict.fromkeys(include_list_title)
# for i, url in enumerate(include_list_url):
#     cast_size[include_list_title[i]] = getCastSize(url)
##########################################
# # post-processing popular actors (optional)
# actors_pool = []
# for key, item in top100_dict.items():
#     actors_ = item['actor_names']
#     actors_pool.extend(actors_)
#
# # actors and how many times they've appeared
# actors_dict = defaultdict(int)
# for actor in actors_pool:
#     if actor not in actors_dict:
#         actors_dict[actor] = 1
#     else:
#         actors_dict[actor] += 1
#
# # number of first billed for each movie
# actors_num_dict = dict.fromkeys(top100_dict.keys())
# for key, item in top100_dict.items():
#     actors_num_dict[key] = len(top100_dict[key]['actor_names'])
#
# sorted(actors_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
# sorted(actors_num_dict.items(), key=operator.itemgetter(1), reverse=True)
##########################################

# df_cast = pd.DataFrame.from_dict(cast_size, orient='index')
# df_cast.columns = ['num_actors']
# df = df.merge(df_cast, left_index=True, right_index=True)

# a few movies don't have opening weekend recorded
df[df.isnull().any(axis=1)]
df = df.dropna()
df['year'] = df['release_date'].dt.year

# cpi for inflation adjustment
cpi = pd.read_csv('inflation_cpi.csv')
cpi = cpi.copy()[['Year', 'Annual Average']]
cpi = cpi[cpi['Year'] >= 1980]

# rebase to 2018 dollars
cpi['Rebase'] = cpi['Annual Average']/cpi['Annual Average'].tail(1).values
df = df.reset_index().merge(cpi.drop(columns='Annual Average'), left_on='year', right_on='Year').set_index('index')
df[['budget', 'openingwknd', 'gross']] = df[['budget', 'openingwknd', 'gross']].div(df['Rebase'], axis=0)
df = df.drop(columns=['Year', 'Rebase'])

df.index.names = ['title']

# back up data to file
# df.to_csv('df_merged_4.csv')
# df = pd.read_csv('df_merged_4.csv', index_col=0)
# df['release_date'] = pd.to_datetime(df['release_date'])

# add cci to gauge consumer sentiment
cci = pd.read_csv('cci.csv', header=None)
cci = cci.iloc[:, 1:3]
cci.columns = ['month', 'cci']
cci['month'] = pd.to_datetime(cci['month'], format='%Y-%m')
cci['month'] = tuple(zip(cci['month'].dt.month, cci['month'].dt.year))

df['month'] = tuple(zip(df['release_date'].dt.month, df['release_date'].dt.year))
df = df.reset_index().merge(cci, on='month').set_index('title')
df = df.drop_duplicates()

# whether the movie is a sequel
# df = pd.read_csv('df_merged_test.csv', index_col=0)
# df.reset_index(inplace=True)
# df = df.rename(columns={'index': 'title'})

sequel = pd.read_csv('sequel_flag_test.csv')
df = df.merge(sequel, on='title')

# run content score section below
content_df = pd.DataFrame(content_scores_backup)  # change back to content_scores if no multiple runs
content_df = content_df.T

df = df.merge(content_df, left_index=True, right_index=True)
df_logs = df_logs.merge(content_df, left_index=True, right_index=True)

df.to_csv('df_merged_test.csv')

# add distance to nearest holiday
import holidays
us_holidays = holidays.UnitedStates(years=list(range(1980, 2019)))
us_holidays = list(us_holidays.keys())

date_diff = []
for x in list(df.release_date):
    date_diff_prep = []
    for y in us_holidays:
        dd = abs((x - y).days)
        date_diff_prep.append(dd)
    min_diff = min(date_diff_prep)
    date_diff.append(min_diff)

df = pd.concat([df, pd.Series(date_diff, name='date_diff')], axis=1)

# whether the movie is a sequel
sequel = pd.read_csv('sequel_flag_4.csv')
df = df.reset_index().merge(sequel, on='title').set_index('title')

# df = df.drop(['Piranha 3D', 'Texas Chainsaw 3D'], axis=0)
# df.to_csv('df_merged_4.csv', index=False)

###################################
# model selection - round 2
# Paranormal Activity, The Blair Witch Project and The First Purge were removed
# set(df3.release_date).symmetric_difference(set(df2.release_date))
# df = pd.concat([df, df3], join='inner')
# df[df.isnull().any(axis=1)]
# df = df.reset_index().drop(columns=['index'])
# df.to_csv('df_merged_final.csv', index=False)

# interaction, e.g., review * budget
lm1 = smf.ols('gross ~ awards + metacritic + budget + locs + num_actors + cci + date_diff + sequel', data=df)
fit1 = lm1.fit()
fit1.summary()

lm2 = smf.ols('gross ~ awards + metacritic + budget + locs + cci + sequel', data=df)
fit2 = lm2.fit()
fit2.summary()

lm3 = smf.ols('gross ~ metacritic + locs + cci', data=df)
fit3 = lm3.fit()
fit3.summary()

df_logs = df.copy()
df_logs[['budget', 'openingwknd', 'gross']] = df_logs[['budget', 'openingwknd', 'gross']].apply(np.log)

lm4 = smf.ols('gross ~ metacritic + locs + cci', data=df_logs)
fit4 = lm4.fit()
fit4.summary()

lm5 = smf.ols('gross ~ metacritic + locs + cci + openingwknd', data=df_logs)
fit5 = lm5.fit()
fit5.summary()  # 0.468 R-square

# error plots
# fig = plt.figure(figsize=(12,8))
# fig = sm.graphics.plot_partregress_grid(fit5, fig=fig)
# plt.show()

lm6 = smf.ols('gross ~ metacritic + locs + cci + openingwknd',
              data=df_logs)
fit6 = lm6.fit()
fit6.summary()  # 0.506 R-square

# influence plot
# fig, ax = plt.subplots(figsize=(12,8))
# fig = sm.graphics.influence_plot(fit6, ax=ax, criterion="cooks")
# plt.show()

# fit plot (against a single predictor)
# fig, ax = plt.subplots(figsize=(12, 8))
# fig = sm.graphics.plot_fit(fit6, "metacritic", ax=ax)
# plt.show()

################################
# content_scores = dict.fromkeys(include_list_title)
# for i, url in enumerate(include_list_url):
#     content_scores[include_list_title[i]] = getContentScore(url).copy()
#
# # content_scores_backup = content_scores.copy()
# n = include_list_title.index('Resident Evil: Apocalypse')
# include_list_title = include_list_title[n:]
# include_list_url = include_list_url[n:]
#
# content_scores_backup.update(content_scores)
#
# content_df = pd.DataFrame(content_scores)
# content_df = content_df.T
#
# df = df.merge(content_df, left_index=True, right_index=True)
# df_logs = df_logs.merge(content_df, left_index=True, right_index=True)
# df.to_csv('df_merged_3.csv', index=False)

lm7 = smf.ols('gross ~ metacritic + locs + cci + openingwknd + alcohol_score + '
              'frightening_score + nudity_score + profanity_score + violence_score',
              data=df_logs[df_logs.index!='I Know What You Did Last Summer'])
fit7 = lm7.fit()
fit7.summary()  # 0.558 R-square

# alcohol, nudity and violence don't seem to matter much; bucketing them together doesn't add much
# df_logs['agg_score'] = df_logs['alcohol_score'] + df_logs['nudity_score'] + df_logs['violence_score'] + df_logs['profanity_score'] + df_logs['frightening_score']
lm8 = smf.ols('gross ~ metacritic + locs + cci + openingwknd + '
              'frightening_score:profanity_score',
              data=df_logs[df_logs.index!='I Know What You Did Last Summer'])
fit8 = lm8.fit()
fit8.summary()  # 0.563 R-square

#####################################
df_test = pd.read_csv('df_merged_test.csv')
df_test = df_test[df_test['title']!='The Shining']  # The Shining opened to limited release
df_test = df_test[df_test['title']!='28 Days Later...']  # the way the movie counts locales is too granular
# 3-D movies were a fad for a while
three_d = ['Jaws3-D', 'Saw 3D: The Final Chapter', 'My Bloody Valentine']
not_three_d = [x for x in df_test.title if x not in three_d]
df_test = df_test[df_test['title'].isin(not_three_d)]
df_test_tile = df_test['title']

df_test['year'] = pd.to_datetime(df_test['release_date']).dt.year
df_test = df_test.loc[:, ['budget', 'openingwknd', 'metacritic', 'locs',
                     'cci', 'sequel', 'alcohol_score', 'frightening_score',
                     'nudity_score', 'profanity_score', 'violence_score', 'year', 'gross']]
df_logs_test = df_test.copy()
df_logs_test[['budget', 'openingwknd', 'gross']] = df_logs_test[['budget', 'openingwknd', 'gross']].apply(np.log)

df_train = pd.read_csv('df_merged_final.csv')
df_train['year'] = pd.to_datetime(df_train['release_date']).dt.year
df_train = df_train.loc[:, ['budget', 'openingwknd', 'metacritic', 'locs',
                     'cci', 'sequel', 'alcohol_score', 'frightening_score',
                     'nudity_score', 'profanity_score', 'violence_score', 'year', 'gross']]
df_logs_train = df_train.copy()
df_logs_train[['budget', 'openingwknd', 'gross']] = df_logs_train[['budget', 'openingwknd', 'gross']].apply(np.log)

df_x_train = df_train.drop(columns=['gross'])
df_x_train_logs = df_logs_train.drop(columns=['gross'])
df_y_train = df_train.loc[:, 'gross']
df_y_train_logs = df_logs_train.loc[:, 'gross']

df_x_test = df_test.drop(columns=['gross'])
df_x_test_logs = df_logs_test.drop(columns=['gross'])
df_y_test = df_test.loc[:, 'gross']
df_y_test_logs = df_logs_test.loc[:, 'gross']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

std = StandardScaler()  # standard scaler here or after polynomialfeatures transform??
std.fit(df_x_train)
x_train = std.transform(df_x_train)
x_test = std.transform(df_x_test)

# x_train = df_x_train
# x_test = df_x_test

# fit Lasso CV
lasso_model = LassoCV(cv=5)
reg = lasso_model.fit(x_train, df_y_train)
reg.score(x_train, df_y_train)  # 0.703, final: 0.74
reg.score(x_test, df_y_test)  # -1.362, final: 0.05

# reg_linear = LinearRegression().fit(x_train, df_y_train)
# reg_linear.score(x_train, df_y_train)
# reg_linear.score(x_test, df_y_test)

lasso_model = LassoCV(cv=5)
reg = lasso_model.fit(df_x_train_logs, df_y_train_logs)
reg.score(df_x_train_logs, df_y_train_logs)  # 0.699, final: 0.682
reg.score(df_x_test_logs, df_y_test_logs)  # -0.133, final: 0.451

reg.alpha_
reg.coef_

# fit polynomial model with interactions only
polynomial_model = PolynomialFeatures(degree=2, interaction_only=True)
x2_train = polynomial_model.fit_transform(x_train)
x2_test = polynomial_model.fit_transform(x_test)
lasso_model2 = LassoCV(cv=5)
reg2 = lasso_model2.fit(x2_train, df_y_train)

polynomial_model.get_feature_names()
polynomial_model.get_feature_names(df_x_train.columns)
features_poly = pd.DataFrame({'features': polynomial_model.get_feature_names(df_x_train.columns), 'coefs': reg2.coef_})
features_poly[features_poly.coefs != 0]
features_poly['abs_coefs'] = abs(features_poly['coefs'])
features_poly.sort_values(by='abs_coefs', ascending=False).head(12)

reg2.score(x2_train, df_y_train)  # 0.786, final: 0.808
reg2.score(x2_test, df_y_test)  # -1.5, final: 0.13

# choosing top 10ish features and fitting them to the statsmodels OLS
lm9 = smf.ols('gross ~ metacritic + cci + openingwknd + year + sequel +'
              'openingwknd:metacritic + openingwknd:profanity_score +'
              'openingwknd:sequel + metacritic:cci + metacritic:sequel +'
              'locs:profanity_score + budget:openingwknd',
              data=df_train)
fit9 = lm9.fit()
fit9.summary()  # 0.693 R-square

# fit polynomial model with full set
polynomial_model = PolynomialFeatures(degree=2)
x3_train = polynomial_model.fit_transform(x_train)
x3_test = polynomial_model.fit_transform(x_test)
lasso_model3 = LassoCV(cv=5)
reg3 = lasso_model3.fit(x3_train, df_y_train)

polynomial_model.get_feature_names()
polynomial_model.get_feature_names(df_x_train.columns)
features_poly = pd.DataFrame({'features': polynomial_model.get_feature_names(df_x_train.columns), 'coefs': reg3.coef_})
features_poly[features_poly.coefs != 0]
features_poly['abs_coefs'] = abs(features_poly['coefs'])
features_poly.sort_values(by='abs_coefs', ascending=False).head(12)

reg3.score(x3_train, df_y_train)  # 0.821, final: 0.565
reg3.score(x3_test, df_y_test)  # -5.37, final: 0.320

# choosing top n features combined with features selected based on lm9 t-scores and fitting them to the statsmodels OLS
# removing the individual polynomial degree 2 terms and replacing with log transformation
lm10 = smf.ols('gross ~ metacritic + cci + openingwknd + year + sequel +'
              'openingwknd:metacritic + openingwknd:profanity_score +'
              'openingwknd:sequel + metacritic:cci + metacritic:sequel +'
              'locs:profanity_score + budget:openingwknd',
              data=df_logs_train)
fit10 = lm10.fit()
fit10.summary()  # 0.767, final: 0.716

# predictions on the test set
pred = fit10.predict(df_logs_test)

# calculating test set r-squared
mean_test = df_logs_test.gross.mean()
tot_ss = sum([(x - mean_test)**2 for x in df_logs_test.gross])
# resid_ss = sum([(y - x)**2 for y in pred for x in df_logs.gross])  # this will loop over each element of b for each element of a
resid_ss = sum([(y - x)**2 for y, x in zip(pred, df_logs_test.gross)])
test_r_squared = 1 - (resid_ss / tot_ss)
test_r_squared  # 0.277

fig, ax = plt.subplots()
ax.plot(df_logs_test.gross, 'o', label="test data")
ax.plot(fit10.predict(df_logs_test), 'o', c='red', label="OLS prediction")
ax.legend(loc="best")
ax.set_xticklabels('')
ax.set_ylabel('log of gross')
plt.savefig('test_v_pred.png', dpi=600, bbox_inches="tight")
# plt.show()

# error plots
fig = plt.figure(figsize=(12,20))
fig = sm.graphics.plot_partregress_grid(fit10, fig=fig)
plt.savefig('partial_reg.png', dpi=600, bbox_inches="tight")
# plt.show()

# removing components based on partial regression plots
lm12 = smf.ols('gross ~ metacritic + cci + openingwknd + year + sequel +'
              'openingwknd:metacritic + '
              'openingwknd:sequel + metacritic:cci + metacritic:sequel +'
              'budget:openingwknd',
              data=df_logs_train)
fit12 = lm12.fit()
fit12.summary()  # 0.687 R-square

# calculating test set r-squared
pred = fit12.predict(df_logs_test)
resid_ss = sum([(y - x)**2 for y, x in zip(pred, df_logs_test.gross)])
test_r_squared = 1 - (resid_ss / tot_ss)
test_r_squared  # 0.231

# keep only baseline features
lm13 = smf.ols('gross ~ metacritic + cci + openingwknd + '
               'year + sequel',
              data=df_logs_train)
fit13 = lm13.fit()
fit13.summary()  # 0.669 R-square

# calculating test set r-squared
pred = fit13.predict(df_logs_test)
resid_ss = sum([(y - x)**2 for y, x in zip(pred, df_logs_test.gross)])
test_r_squared = 1 - (resid_ss / tot_ss)
test_r_squared  # 0.372

# if using lm10 specification on test
lm_test = smf.ols('gross ~ metacritic + cci + openingwknd + year + sequel +'
              'openingwknd:metacritic + openingwknd:profanity_score +'
              'openingwknd:sequel + metacritic:cci + metacritic:sequel +'
              'locs:profanity_score + budget:openingwknd',
              data=df_logs_test)
fit_test = lm_test.fit()
fit_test.summary()  # 0.941 R-square! (adjusted 0.915)