# imdb
Horror movie box office prediction

Using scraped data from imdb website, I did regression analysis of horror movies, using different features to predict domestic gross.

Horror genre is predictably difficult, given that its popularity wanes and waxes and there are a lot of remakes and sequels that simply tried to ride the coattails of previous successes. The crucial information of marketing spend is absent from the data and supplemental data/features such as consumer sentiment, release date proximity to major holidays, credited cast size, etc were not found to improve prediction power much. 

The analysis found that predicting ahead of release is extremely difficult, but with opening weekend box office information, the prediction accuracy improved drastically and the ways in which opening gross interacts with other feautres proved informative: such as interactions with budget, critical reception, and a few content scores (as measured by occurences of scenes that MPAA uses for audience rating, appropriate in this case).
