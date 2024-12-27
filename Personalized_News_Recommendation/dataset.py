"""
Line format for yahoo events:
1241160900 109513 0 |user 2:0.000012 3:0.000000 4:0.000006 5:0.000023 6:0.999958 1:1.000000 |109498 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 |109509 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 [[...more article features omitted...]] |109453 2:0.421669 3:0.000011 4:0.010902 5:0.309585 6:0.257833 1:1.000000

Some log files contain rows with erroneous data.

After the first 10 columns are the articles and their features.
Each article has 7 columns (articleid + 6 features)
Therefore number_of_columns-10 % 7 = 0
"""
import re
import numpy as np
import fileinput


# def get_yahoo_events(filenames):
#     """
#     Reads a stream of events from the list of given files.
    
#     Parameters
#     ----------
#     filenames : list
#         List of filenames
    
#     Stores
#     -------    
#     articles : [article_ids]
#     features : [[article_1_features] .. [article_n_features]]
#     events : [
#                  0 : displayed_article_index (relative to the pool),
#                  1 : user_click,
#                  2 : [user_features],
#                  3 : [pool_indexes]
#              ]
#     """

#     global articles, features, events, n_arms, n_events
#     articles = []
#     features = []
#     events = []

#     skipped = 0

#     pattern =  r'\|(\d{6})'

#     with fileinput.input(files=filenames) as f:
#         for line in f:
#             cols = line.split()
#             cols_raw = line
#             pool_ids = re.findall(pattern,cols_raw)

#             article = cols[1]
#             if article not in articles:
#                 articles.append(article)
#                 feature = []
#                 for x in cols[4:10]:
#                     feature.append(float(x[2:]))
#                 features.append(feature)
#             events.append(
#                 [
#                     pool_ids.index(article),
#                     int(cols[2]),
#                     [float(x[2:]) for x in cols[4:10]],
#                     pool_ids,
#                 ]
#             )
#     for i in range(len(events)):
#         pool_idx = []
#         for item in events[i][-1]:
#             pool_idx.append(articles.index(item))
#         events[i][-1] = pool_idx

#     features = np.array(features)
#     n_arms = len(articles)
#     n_events = len(events)
#     print(n_events, "events with", n_arms, "articles")

#     return articles,features,events

import re
import numpy as np
import fileinput

def get_yahoo_events(filenames):
    """
    Reads a stream of events from the list of given files.
    
    Parameters
    ----------
    filenames : list
        List of filenames
    
    Returns
    -------
    articles : list
        List of article IDs
    features : numpy.ndarray
        2D array of features for each article
    events : list
        List of events containing:
            - Displayed article index (relative to the pool)
            - User click (1 or 0)
            - User features (list)
            - Pool indexes (list of article IDs)
    """
    global articles, features, events, n_arms, n_events,kernel_features
    articles = []  # List to store unique article IDs
    features = []  # List to store article features
    events = []    # List to store events

    skipped = 0

    pattern = r'\|(\d{6})'

    # Create a dictionary to map article ID to index (for faster lookups)
    article_to_index = {}

    # Iterate over the files to process events
    with fileinput.input(files=filenames) as f:
        for line in f:
            cols = line.split()
            cols_raw = line
            pool_ids = re.findall(pattern, cols_raw)

            article = cols[1]

            # If the article is not already in the list, add it
            if article not in article_to_index:
                article_to_index[article] = len(articles)
                articles.append(article)
                # Extract features for the article
                feature = [float(x[2:]) for x in cols[4:10]]  # Removing the first two characters
                features.append(feature)

            # Append the event (article index, user click, user features, pool IDs)
            events.append([
                pool_ids.index(article),  # Get the index of the article in the pool
                int(cols[2]),             # User click (0 or 1)
                [float(x[2:]) for x in cols[4:10]],  # User features
                pool_ids                   # List of pool article IDs
            ])

    # Now update pool_ids in each event with the index of articles
    for event in events:
        event[-1] = [article_to_index[item] for item in event[-1]]

    # Convert features to a numpy array for further processing
    features = np.array(features)

    # Print event and arm counts for verification
    n_arms = len(articles)  # Number of unique articles
    n_events = len(events)  # Number of events
    print(f"{n_events} events with {n_arms} articles")

    # return articles, features, events



def max_articles(n_articles):
    """
    Reduces the number of articles to the threshold provided.
    Therefore the number of events will also be reduced.

    Parameters
    ----------
    n_articles : number
        number of max articles after reduction
    """

    global articles, features, events, n_arms, n_events
    assert n_articles < n_arms

    n_arms = n_articles
    articles = articles[:n_articles]
    features = features[:n_articles]

    for i in reversed(range(len(events))):
        displayed_pool_idx = events[i][0]  # index relative to the pool
        displayed_article_idx = events[i][3][
            displayed_pool_idx
        ]  # index relative to the articles

        if displayed_article_idx < n_arms:
            events[i][0] = displayed_article_idx
            events[i][3] = np.arange(0, n_arms)  # pool = all available articles
        else:
            del events[i]

    n_events = len(events)
    print("Number of events:", n_events)
