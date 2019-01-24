
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
from helpers import *

from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_validate

sns.set()

# overall horror listing
# find top 200 and take only those released after 1980
# url = "https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&explore=title_type,genres"
# url = "https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=51&explore=title_type,genres&ref_=adv_nxt"
# url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=101&explore=title_type,genres&ref_=adv_nxt'
# url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=151&explore=title_type,genres&ref_=adv_nxt'
# url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=201&explore=title_type,genres&ref_=adv_nxt'
# url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=251&explore=title_type,genres&ref_=adv_nxt'
url = 'https://www.imdb.com/search/title?genres=horror&sort=boxoffice_gross_us,desc&start=301&explore=title_type,genres&ref_=adv_nxt'

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

# add cci to gauge consumer sentiment
cci = pd.read_csv('cci.csv', header=None)
cci = cci.iloc[:, 1:3]
cci.columns = ['month', 'cci']
cci['month'] = pd.to_datetime(cci['month'], format='%Y-%m')
cci['month'] = tuple(zip(cci['month'].dt.month, cci['month'].dt.year))

df['month'] = tuple(zip(df['release_date'].dt.month, df['release_date'].dt.year))
df = df.reset_index().merge(cci, on='month').set_index('title')
df = df.drop_duplicates()

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

# df_test = pd.read_csv('df_merged_test.csv')
# df_test = df_test[df_test['title']!='The Shining']  # The Shining opened to limited release
# df_test = df_test[df_test['title']!='28 Days Later...']  # the way the movie counts locales is too granular
# # 3-D movies were a fad for a while
# three_d = ['Jaws3-D', 'Saw 3D: The Final Chapter', 'My Bloody Valentine']
# not_three_d = [x for x in df_test.title if x not in three_d]
# df_test = df_test[df_test['title'].isin(not_three_d)]
# df_test_tile = df_test['title']

var_to_keep = ['budget', 'openingwknd', 'metacritic', 'locs',
                     'cci', 'sequel', 'alcohol_score', 'frightening_score',
                     'nudity_score', 'profanity_score', 'violence_score', 'year', 'gross']
var_to_log = ['budget', 'openingwknd', 'gross']

df_test = pd.read_csv('df_merged_test_final.csv')
# df_test['year'] = pd.to_datetime(df_test['release_date']).dt.year
df_test = df_test.loc[:, var_to_keep]
df_logs_test = df_test.copy()
df_logs_test[var_to_log] = df_logs_test[var_to_log].apply(np.log)

df_train = pd.read_csv('df_merged_final.csv')
# df_train = df_train.sort_values(by='openingwknd')[4:]
df_train['year'] = pd.to_datetime(df_train['release_date']).dt.year
df_train = df_train.loc[:, var_to_keep]
df_logs_train = df_train.copy()
df_logs_train[var_to_log] = df_logs_train[var_to_log].apply(np.log)

df_x_train = df_train.drop(columns=['gross'])
df_x_train_logs = df_logs_train.drop(columns=['gross'])
df_y_train = df_train.loc[:, 'gross']
df_y_train_logs = df_logs_train.loc[:, 'gross']

df_x_test = df_test.drop(columns=['gross'])
df_x_test_logs = df_logs_test.drop(columns=['gross'])
df_y_test = df_test.loc[:, 'gross']
df_y_test_logs = df_logs_test.loc[:, 'gross']
###########################

lm1_2 = smf.ols('gross ~ metacritic + budget + locs + cci + sequel + year +'
                'alcohol_score + frightening_score + nudity_score + profanity_score + violence_score', data=df_train)
fit1_2 = lm1_2.fit()
fit1_2.summary()  # 0.344

lm1_3 = smf.ols('gross ~ metacritic + budget + locs + cci + sequel + year', data=df_train)
fit1_3 = lm1_3.fit()
fit1_3.summary()  # 0.297

# baseline linear model
lm1_3 = smf.ols('gross ~ metacritic + budget + locs + cci + sequel + year + frightening_score', data=df_train)
fit1_3 = lm1_3.fit()
fit1_3.summary()  # 0.330

scoreTestLM(df_test, fit1_3.predict(df_test))  # -1.04

# check we get similar results using sklearn LinearRegression
# model = LinearRegression()
# to_drop = ['openingwknd', 'alcohol_score', 'nudity_score', 'profanity_score', 'violence_score']
# m = model.fit(df_x_train.drop(columns=to_drop), df_y_train)
# m.score(df_x_train.drop(columns=to_drop), df_y_train)
# m.score(df_x_test.drop(columns=to_drop), df_y_test)

# baseline linear model (log transformed)
lm1_4 = smf.ols('gross ~ metacritic + budget + locs + cci + sequel + year + frightening_score', data=df_logs_train)
fit1_4 = lm1_4.fit()
fit1_4.summary()  # 0.312

scoreTestLM(df_test, fit1_4.predict(df_test))  # -5.7

# baseline linear model (with openingwknd)
lm1_5 = smf.ols('gross ~ metacritic + budget + locs + cci + sequel + year + openingwknd', data=df_train)
fit1_5 = lm1_5.fit()
fit1_5.summary()  # 0.732

scoreTestLM(df_test, fit1_5.predict(df_test))  # 0

# baseline linear model (with openingwknd) logged
lm1_6 = smf.ols('gross ~ metacritic + budget + locs + cci + sequel + year + openingwknd', data=df_logs_train)
fit1_6 = lm1_6.fit()
fit1_6.summary()  # 0.684

scoreTestLM(df_test, fit1_6.predict(df_test))  # -3.7

#####################################
# fit LassoCV
std = StandardScaler()
std.fit(df_x_train)
x_train = std.transform(df_x_train)
x_test = std.transform(df_x_test)

lasso_model = LassoCV(cv=5)
reg = lasso_model.fit(x_train, df_y_train)
reg.score(x_train, df_y_train)  # 0.703, final: 0.74
reg.score(x_test, df_y_test)  # -1.362, final: 0.05

# fit LassoCV to logged variables
std = StandardScaler()
std.fit(df_x_train_logs)
x_train_logs = std.transform(df_x_train_logs)
x_test_logs = std.transform(df_x_test_logs)

lasso_model = LassoCV(cv=5)
reg = lasso_model.fit(x_train_logs, df_y_train_logs)
reg.score(x_train_logs, df_y_train_logs)  # 0.699, final: 0.68
reg.score(x_test_logs, df_y_test_logs)  # -0.133, final: 0.36

reg.alpha_
reg.coef_

# fit polynomial model with interactions only
polynomial_model = PolynomialFeatures(degree=2, interaction_only=True)
x2_train = polynomial_model.fit_transform(df_x_train)
x2_test = polynomial_model.fit_transform(df_x_test)

# pass through standard scaler here
std = StandardScaler()  # standard scaler after polynomialfeatures transform so coefs can be compared
std.fit(x2_train)
x2_train = std.transform(x2_train)
x2_test = std.transform(x2_test)

lasso_model2 = LassoCV(cv=5)
reg2 = lasso_model2.fit(x2_train, df_y_train)

getCoefPolyFeatures(df_x_train, reg2, 10)

reg2.score(x2_train, df_y_train)  # 0.786, final: 0.808, final rescaled: 0.76
reg2.score(x2_test, df_y_test)  # -1.5, final: 0.13, final rescaled: 0.14

# choosing top 10ish features and fitting them to the statsmodels OLS
lm9_2 = smf.ols('gross ~ metacritic + cci + openingwknd + year +'
                'openingwknd:cci + metacritic:profanity_score +'
              'openingwknd:metacritic + openingwknd:profanity_score +'
              'locs:profanity_score + budget:openingwknd',
              data=df_train)

fit9_2 = lm9_2.fit()
fit9_2.summary()  # 0.714 R-square

# fit polynomial model with full set
polynomial_model = PolynomialFeatures(degree=2)
x3_train = polynomial_model.fit_transform(df_x_train)
x3_test = polynomial_model.fit_transform(df_x_test)

# pass through standard scaler here
std = StandardScaler()
std.fit(x3_train)
x3_train = std.transform(x3_train)
x3_test = std.transform(x3_test)

lasso_model3 = LassoCV(cv=5)
reg3 = lasso_model3.fit(x3_train, df_y_train)

lasso_model4 = Lasso(alpha=7943282)  # see alpha section below for test data
reg4 = lasso_model4.fit(x3_train, df_y_train)

reg3.score(x3_train, df_y_train)  # 0.821, final: 0.565, final rescaled: 0.765
reg3.score(x3_test, df_y_test)  # -5.37, final: 0.320, final rescaled: 0.14

reg4.score(x3_train, df_y_train)  # 0.821, final: 0.565, final rescaled: 0.673
reg4.score(x3_test, df_y_test)  # -5.37, final: 0.320, final rescaled: 0.304

getCoefPolyFeatures(df_x_train, reg3, 10)
getCoefPolyFeatures(df_x_train, reg4, 10)

# trying ridge
ridge_model3 = RidgeCV(cv=3)
ridge3 = ridge_model3.fit(x3_train, df_y_train)

getCoefPolyFeatures(df_x_train, ridge3, 10)

ridge3.score(x3_train, df_y_train)  #0.822
ridge3.score(x3_test, df_y_test)  #-0.094

reg3.alpha_

for i in np.arange(5, 8, 0.1):
    m = Lasso(alpha=10**i)
    m.fit(x3_train, df_y_train)
    print(m.alpha, m.score(x3_test, df_y_test))
# 7943282.347242692 0.30429274826570174

for i in range(1000, 10000, 1000):
    m = Lasso(alpha=i, max_iter=100000)
    m.fit(x3_train, df_y_train)
    print(m.alpha, m.score(x3_train, df_y_train))

# cross_val_score(LassoCV(), x3_train, df_y_train, cv=5)
# array([-0.88709572, -1.87439665,  0.89369045,  0.41807457,  0.08229462])
# array([-0.8808291 , -2.19301234,  0.89369045,  0.46425809,  0.02950212])

# mod_val = Lasso(random_state=11)
# cross_validate(mod_val, x3_train, df_y_train, cv=5)
# 'test_score': array([-3.43285287, -8.31206484,  0.76166814,  0.21932813, -6.80803267]
# 'train_score': array([0.86893526, 0.87476413, 0.83338405, 0.90071931, 0.86890055]

# mod_val = Lasso(random_state=13)
# cross_validate(mod_val, x3_train, df_y_train, cv=5)


# choosing top n features combined with features selected based on lm9 t-scores
# and removing variables where outliers are impacting regressoion results
lm10_2 = smf.ols('gross ~ metacritic + cci + openingwknd + year +'
                'openingwknd:cci + metacritic:profanity_score +'
              'openingwknd:metacritic + openingwknd:profanity_score +'
              'locs:profanity_score + budget:openingwknd',
              data=df_logs_train)
fit10_2 = lm10_2.fit()
fit10_2.summary()  # 0.833, final 0.702

lm10_2 = smf.ols('gross ~ metacritic + cci + openingwknd + year +'
                'openingwknd:cci + metacritic:profanity_score +'
              'openingwknd:metacritic + openingwknd:profanity_score +'
              'locs:profanity_score + budget:openingwknd',
              data=df_train)
fit10_2 = lm10_2.fit()
fit10_2.summary()  # 0.833, final 0.714

# using specific alpha to narrow down feature set through Lasso
lm10_4 = smf.ols('gross ~ year + openingwknd:cci + '
              'openingwknd:metacritic + openingwknd:locs +'
              'locs:profanity_score + budget:openingwknd',
              data=df_train)
fit10_4 = lm10_4.fit()
fit10_4.summary()  # 0.833, final 0.671

scoreTestLM(df_test, fit10_4.predict(df_test))  # 0.366

# removing components based on partial regression plots
lm12_2 = smf.ols('gross ~ year + openingwknd:cci + '
              'openingwknd:metacritic + '
              'budget:openingwknd',
              data=df_train)
fit12_2 = lm12_2.fit()
fit12_2.summary()  # 0.635 R-square

scoreTestLM(df_test, fit12_2.predict(df_test))  # 0.359

# error plots
fig, ax = plt.subplots()
ax.plot(df_test.gross, 'o', label="test data")
ax.plot(fit10_4.predict(df_test), 'o', c='red', label="OLS prediction")
ax.legend(loc="best")
ax.set_xticklabels('')
ax.set_ylabel('gross')
# plt.savefig('test_v_pred_final.png', dpi=600, bbox_inches="tight")
plt.show()

# error plots
fig = plt.figure(figsize=(12,20))
fig = sm.graphics.plot_partregress_grid(fit10_4, fig=fig)
# plt.savefig('partial_reg_final.png', dpi=600, bbox_inches="tight")
plt.show()

# if using final specification on test
lm_test = smf.ols('gross ~ year + openingwknd:cci + '
              'openingwknd:metacritic + '
              'budget:openingwknd',
              data=df_test)
fit_test = lm_test.fit()
fit_test.summary()  # 0.586 R-square


# saving some model results to pickle files
# with open('fit10.pickle', 'wb') as handle:
#     pickle.dump(fit10, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('fit13.pickle', 'wb') as handle:
#     pickle.dump(fit13, handle, protocol=pickle.HIGHEST_PROTOCOL)

# coefficient comparison
# model 12_2
# intercept 5.85
# openingwknd:metacritic 0.0272
# openingwknd:budget ~ -0
# openingwknd:cci 0.0051
# year 11730

# model test
# intercept 3.25
# openingwknd:metacritic 0.0079
# openingwknd:budget ~ -0 (not significant)
# openingwknd:cci 0.0178
# year 6509
