
def getTitle_getURL(url, year_filter=1980):
    import requests
    from bs4 import BeautifulSoup
    import re
    url = url
    response = requests.get(url)
    soup1 = BeautifulSoup(response.text, 'html.parser')
    # soup.find(class_='lister-item-index unbold text-primary').findNextSibling()
    title_loc_prep = soup1.find_all(class_='lister-item-index unbold text-primary')
    year_loc_prep = soup1.find_all(class_='lister-item-year text-muted unbold')
    title_loc = []
    for x in title_loc_prep:
        title_loc.append(x.findNextSibling())

    title_for_url = []
    title_text = []

    for x in title_loc:
        s1 = str(x).split('/')
        s2 = []
        for item in s1:
            s2.extend(item.split('"'))  # this way we avoid nested lists with append
        s3 = ('/').join(s2[2:5])
        s_title = s2[5]
        title_for_url.append(s3)
        title_text.append(s_title)

    title_text = [title.replace('<', '>').strip('>') for title in title_text]

    year = []
    for substr in year_loc_prep:
        substr = str(substr)
        year.append(int(re.findall(r'\d+', substr)[0]))

    # url list for loop through
    url_list = []
    for item in title_for_url:
        newitem = 'https://www.imdb.com/' + item
        url_list.append(newitem)

    # update url_list and title_text to include movies after 1990
    # can adjust the filter by changing year parameter
    year_filter = year_filter
    exclude_list_title = []
    for i, j in zip(title_text, year):
        if j < year_filter:
            exclude_list_title.append(i)

    include_list_url = []
    for i, j in zip(url_list, year):
        if j >= year_filter:
            include_list_url.append(i)

    include_list_title = [x for x in title_text if x not in exclude_list_title]

    return include_list_url, include_list_title, year, title_text

# function for extracting number of shooting locations
def locationCount(url):
    import requests
    from bs4 import BeautifulSoup
    import re
    # count the number of shooting locations
    url_loc = url[:url.find('?')] + 'locations?ref_=ttspec_sa_5'
    response = requests.get(url_loc)
    soup2 = BeautifulSoup(response.text, 'html.parser')
    # just need to count occurrences of this class
    filmloc = len(list(re.finditer('soda sodavote odd', str(soup2.find_all(class_='soda sodavote odd'))))) + \
              len(list(re.finditer('soda sodavote even', str(soup2.find_all(class_='soda sodavote even')))))
    return filmloc

def stripDollar(strng):
    import re
    strng = str(strng).splitlines()[1]  # happen to work for both budget and opening weekend
    idx_start = strng.find('$')
    substr = strng[idx_start: idx_start + 12]  # 9 digits 2 comma separators 1 dollar sign
    dollar = int(('').join(re.findall(r'\d+', substr)))
    return dollar

# function to extract the number of nominations and wins
def numAwards(strng):
    import re
    # wins = 0
    noms = 0
    # only need to find number, sometimes major awards are mentioned before
    # sometimes there are no nominations or awards, sometimes there's one not the other
    strng = str(strng)
    wins = int(re.findall(r'\d+', strng)[0])  # we only need the total so mislabeling noms as wins here is okay\
    try:
        noms = int(re.findall(r'\d+', strng)[1])
    except IndexError:
        pass
    awards = wins + noms
    return awards

# return dict of stats for one movie
def onemoviestats(url):
    import requests
    from collections import defaultdict
    from bs4 import BeautifulSoup
    import re
    import pandas as pd

    url = url
    response = requests.get(url)
    soup3 = BeautifulSoup(response.text, 'html.parser')
    movie_dict = defaultdict(int)

    # metacritic score
    metacritic_prep1 = soup3.find(class_="metacriticScore score_favorable titleReviewBarSubItem")
    metacritic_prep2 = soup3.find(class_="metacriticScore score_mixed titleReviewBarSubItem")
    metacritic_prep3 = soup3.find(class_="metacriticScore score_unfavorable titleReviewBarSubItem")

    if (metacritic_prep1 is None) and (metacritic_prep2 is None) and (metacritic_prep3 is None):
        metacritic = int(float(soup3.find(itemprop='ratingValue').get_text())*10)
    elif metacritic_prep1 is not None:
        metacritic = int(re.findall(r'\d+', str(metacritic_prep1))[0])
    elif metacritic_prep2 is not None:
        metacritic = int(re.findall(r'\d+', str(metacritic_prep2))[0])
    else:
        metacritic = int(re.findall(r'\d+', str(metacritic_prep3))[0])

    # opening weekend box office
    # only need to find number
    if soup3.find(text="Opening Weekend USA:") is None:
        openingwknd = None
    else:
        openingwknd_prep = soup3.find(text="Opening Weekend USA:").parent.parent  # need to extract dollar string
        openingwknd = stripDollar(openingwknd_prep)

    # movie budget
    if soup3.find(text="Budget:") is None:
        budget = None
    else:
        budget_prep = soup3.find(text="Budget:").parent.parent  # need to extract dollar string
        budget = stripDollar(budget_prep)

    # awards processing--there may not be an awards-blurb class
    try:
        awards_prep = soup3.find_all(class_="awards-blurb")[-1]
        awards = numAwards(awards_prep)
    except IndexError:
        awards = 0

    # US gross sales
    gross_prep = soup3.find(text="Gross USA:").parent.parent
    gross = stripDollar(gross_prep)

    locs = locationCount(url)

    release_date_prep = soup3.find(text='Release Date:').parent.parent
    release_date = str(release_date_prep).replace('</h4>', '!').replace('<span', '!').split('!')[1].strip()
    release_date = release_date.split('(')
    release_date = release_date[0].strip()
    release_date = pd.to_datetime(release_date, format='%d %B %Y')

    movie_dict['awards'] = awards
    movie_dict['budget'] = budget
    movie_dict['openingwknd'] = openingwknd
    movie_dict['metacritic'] = metacritic
    movie_dict['locs'] = locs
    movie_dict['release_date'] = release_date
    movie_dict['gross'] = gross
    movie_dict['nudity_score'], movie_dict['violence_score'], movie_dict['profanity_score'], movie_dict['alcohol_score'], movie_dict['frightening_score'] = getContentScore(url)

    return movie_dict

# find size of full cast
def getCastSize(url):
    import requests
    from bs4 import BeautifulSoup
    url_cast = url[:url.find('?')] + 'fullcredits?ref_=ttco_ql_1'
    response = requests.get(url_cast)
    soup4 = BeautifulSoup(response.text, 'html.parser')
    cast_size = len(soup4.find_all('td', class_='primary_photo'))
    return cast_size

# get release date
def getReleaseDate(url):
    import requests
    from bs4 import BeautifulSoup
    url = url
    response = requests.get(url)
    soup5 = BeautifulSoup(response.text, 'html.parser')
    release_date_prep = soup5.find(text='Release Date:').parent.parent
    release_date = str(release_date_prep).replace('</h4>', '!').replace('<span', '!').split('!')[1].strip()
    return release_date

# parental guide section
# url = 'https://www.imdb.com/title/tt0081505/parentalguide?ref_=tt_stry_pg'
def getContentScore(url):
    from collections import defaultdict
    import requests
    from bs4 import BeautifulSoup
    import re

    content_score = defaultdict(int)
    url_content = url[:url.find('?')] + 'parentalguide?ref_=tt_stry_pg'
    response = requests.get(url_content)

    soup6 = BeautifulSoup(response.text, 'html.parser')

    content_prep1 = soup6.find(href="#advisory-nudity")
    content_prep2 = soup6.find(href="#advisory-violence")
    content_prep3 = soup6.find(href="#advisory-profanity")
    content_prep4 = soup6.find(href="#advisory-alcohol")
    content_prep5 = soup6.find(href="#advisory-frightening")

    nudity_score, violence_score, profanity_score, alcohol_score, frightening_score = 0, 0, 0, 0, 0
    if content_prep1 is None:
        nudity_score = 0
    else:
        nudity_score = int(re.findall(r'\d+', str(content_prep1.parent))[0])

    if content_prep2 is None:
        violence_score = 0
    else:
        violence_score = int(re.findall(r'\d+', str(content_prep2.parent))[0])

    if content_prep3 is None:
        profanity_score = 0
    else:
        profanity_score = int(re.findall(r'\d+', str(content_prep3.parent))[0])

    if content_prep4 is None:
        alcohol_score = 0
    else:
        alcohol_score = int(re.findall(r'\d+', str(content_prep4.parent))[0])

    if content_prep5 is None:
        frightening_score = 0
    else:
        frightening_score = int(re.findall(r'\d+', str(content_prep5.parent))[0])

    return nudity_score, violence_score, profanity_score, alcohol_score, frightening_score