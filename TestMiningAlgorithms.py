import argparse
import numpy as np
from apyori import apriori
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from DataBaseConnection import DataBaseConnection
import mysql.connector as mysql
from Rule import Rule
import time

np.random.seed(0)

'Slow function'


def getRecommendedFilmsUsingApriori(customerID: int):
    start = time.time()
    'open connection with database'
    dbConn = DataBaseConnection()

    Cutomers_IDs = dbConn.get_all_customersIDs()

    'create orders list'
    orders = []
    bought_films_by_user = None
    for id in Cutomers_IDs:
        customer_films = dbConn.get_theBought_films_by_customer(id)
        if id == customerID:
            bought_films_by_user = customer_films
        orders.append(customer_films)
    print(time.time() - start)
    print('Begin Mining')
    association_rules = apriori(orders, min_support=0.01, min_confidence=0.2, min_lift=2, min_length=2)
    association_results = list(association_rules)

    """
    example about rules:
    RelationRecord(
    items=frozenset({'A Few Good Men', 'A beautiful mind'}),                             index:0
    support=0.021336680263570756,                                                        index:1
    ordered_statistics=[OrderedStatistic(items_base=frozenset({'A Few Good Men'}),       index:2
    items_add=frozenset({'A beautiful mind'}),                                   
    confidence=0.3655913978494624, 
    lift=3.2455147212987088)]
       )

    index:2   
    OrderedStatistic(
    items_base=frozenset({'A Few Good Men'}),       index:0
    items_add=frozenset({'A beautiful mind'}),      index:1                               
    confidence=0.3655913978494624,                  index:2
    lift=3.2455147212987088)                        index:3

    """

    'sort rules depending on confidence of rule'
    association_results.sort(key=lambda tup: tup[2][0][2], reverse=True)

    rules = []
    recommended_films = []

    for rule in association_results:
        rule_members = rule[2][0]
        rules.append(Rule(list(rule_members[0]), list(rule_members[1]), rule_members[2]))
        if rule_members[0].issubset(set(bought_films_by_user)) and not rule_members[1].issubset(set(recommended_films)):
            recommended_films.append(list(rule_members[1])[0])

    print(time.time() - start)
    return recommended_films


'Fast function'


def getRecommendedFilmsUsingApriori1(customerID: int):
    start = time.time()
    'open connection with database'
    dbConn = DataBaseConnection()

    all_customers = dbConn.get_all_customersIDs_with_films()

    'create orders list'
    orders = []
    bought_films_by_user = None
    for (customer_id, customer_films) in all_customers:
        if customer_id == customerID:
            bought_films_by_user = customer_films
        orders.append(customer_films)
    print(time.time() - start)
    print('Begin Mining')
    association_rules = apriori(orders, min_support=0.01, min_confidence=0.2, min_lift=2, min_length=2)
    association_results = list(association_rules)

    """
    example about rules:
    RelationRecord(
    items=frozenset({'A Few Good Men', 'A beautiful mind'}),                             index:0
    support=0.021336680263570756,                                                        index:1
    ordered_statistics=[OrderedStatistic(items_base=frozenset({'A Few Good Men'}),       index:2
    items_add=frozenset({'A beautiful mind'}),                                   
    confidence=0.3655913978494624, 
    lift=3.2455147212987088)]
       )

    index:2   
    OrderedStatistic(
    items_base=frozenset({'A Few Good Men'}),       index:0
    items_add=frozenset({'A beautiful mind'}),      index:1                               
    confidence=0.3655913978494624,                  index:2
    lift=3.2455147212987088)                        index:3

    """

    'sort rules depending on confidence of rule'
    association_results.sort(key=lambda tup: tup[2][0][2], reverse=True)

    rules = []
    recommended_films = []

    for rule in association_results:
        rule_members = rule[2][0]
        rules.append(Rule(list(rule_members[0]), list(rule_members[1]), rule_members[2]))
        if rule_members[0].issubset(set(bought_films_by_user)) and not rule_members[1].issubset(set(recommended_films)):
            recommended_films.append(list(rule_members[1])[0])

    print(time.time() - start)
    return recommended_films


'Slow function'


def getRecommendedFilmsByFilmNameUsingApriori(film_name: str):
    film_name = film_name.strip()
    start = time.time()
    'open connection with database'
    dbConn = DataBaseConnection()

    cutomers_IDs = dbConn.get_all_customersIDs()

    'create orders list'
    orders = []
    for id in cutomers_IDs:
        customer_films = dbConn.get_theBought_films_by_customer(id)
        orders.append(customer_films)
    print(time.time() - start)
    print('Begin Mining')
    association_rules = apriori(orders, min_support=0.010, min_confidence=0.1, min_lift=1, min_length=1)
    association_results = list(association_rules)

    """
    example about rules:
    RelationRecord(
    items=frozenset({'A Few Good Men', 'A beautiful mind'}),                             index:0
    support=0.021336680263570756,                                                        index:1
    ordered_statistics=[OrderedStatistic(items_base=frozenset({'A Few Good Men'}),       index:2
    items_add=frozenset({'A beautiful mind'}),                                   
    confidence=0.3655913978494624, 
    lift=3.2455147212987088)]
       )

    index:2   
    OrderedStatistic(
    items_base=frozenset({'A Few Good Men'}),       index:0
    items_add=frozenset({'A beautiful mind'}),      index:1                               
    confidence=0.3655913978494624,                  index:2
    lift=3.2455147212987088)                        index:3


    """

    'sort rules depending on confidence of rule'
    association_results.sort(key=lambda tup: tup[2][0][2], reverse=True)

    rules = []
    recommended_films = []
    one_rule = []

    for rule in association_results:
        rule_members = rule[2][0]
        rules.append(Rule(list(rule_members[0]), list(rule_members[1]), rule_members[2]))
        # if len(list(rule_members[0])) == 1 and len(list(rule_members[1])) == 1:
        #     one_rule.append((list(rule_members[0])[0],list(rule_members[1])[0]))
        if len(list(rule_members[0])) == 1:
            left_side = list(rule_members[0])[0].strip()
        if len(list(rule_members[0])) == 1 and left_side == film_name.strip() and \
                list(rule_members[1])[0] not in recommended_films:
            recommended_films.append(list(rule_members[1])[0])

    recommended_films = list(set(recommended_films))
    if recommended_films.__contains__(film_name.strip()):
        recommended_films.remove(film_name.strip())

    for rule in one_rule:
        print(rule)
    return recommended_films


'Fast function'


def getRecommendedFilmsByFilmNameUsingApriori1(film_name: str):
    film_name = film_name.strip()
    start = time.time()
    'open connection with database'
    dbConn = DataBaseConnection()

    all_customers = dbConn.get_all_customersIDs_with_films()

    'create orders list'
    orders = []
    bought_films_by_user = None
    for (customer_id, customer_films) in all_customers:
        orders.append(customer_films)
    print(time.time() - start)
    print('Begin Mining')
    association_rules = apriori(orders, min_support=0.010, min_confidence=0.1, min_lift=1, min_length=1)
    association_results = list(association_rules)

    """
    example about rules:
    RelationRecord(
    items=frozenset({'A Few Good Men', 'A beautiful mind'}),                             index:0
    support=0.021336680263570756,                                                        index:1
    ordered_statistics=[OrderedStatistic(items_base=frozenset({'A Few Good Men'}),       index:2
    items_add=frozenset({'A beautiful mind'}),                                   
    confidence=0.3655913978494624, 
    lift=3.2455147212987088)]
       )

    index:2   
    OrderedStatistic(
    items_base=frozenset({'A Few Good Men'}),       index:0
    items_add=frozenset({'A beautiful mind'}),      index:1                               
    confidence=0.3655913978494624,                  index:2
    lift=3.2455147212987088)                        index:3


    """

    'sort rules depending on confidence of rule'
    association_results.sort(key=lambda tup: tup[2][0][2], reverse=True)

    rules = []
    recommended_films = []
    one_rule = []

    for rule in association_results:
        rule_members = rule[2][0]
        rules.append(Rule(list(rule_members[0]), list(rule_members[1]), rule_members[2]))
        # if len(list(rule_members[0])) == 1 and len(list(rule_members[1])) == 1:
        #     one_rule.append((list(rule_members[0])[0],list(rule_members[1])[0]))
        left_side = ''
        if len(list(rule_members[0])) == 1:
            left_side = list(rule_members[0])[0].strip()
        if len(list(rule_members[0])) == 1 and left_side == film_name.strip() and \
                list(rule_members[1])[0] not in recommended_films:
            recommended_films.append(list(rule_members[1])[0])

    recommended_films = list(set(recommended_films))
    if recommended_films.__contains__(film_name.strip()):
        recommended_films.remove(film_name.strip())

    for rule in one_rule:
        print(rule)
    return recommended_films


def getRecommendedFilmsUsingClustering(customerID: int):
    'open connection with database'
    dbConn = DataBaseConnection()

    customers_features, customers_count, features_count = dbConn.get_customers_features()
    customers_details = dbConn.get_customer_details(customerID)

    customers_features.extend(customers_details)
    features_before_encode = np.array(customers_features).reshape((customers_count + 1, features_count))

    customers_IDs = features_before_encode[:, 0]
    'Remove the last one from the array'
    customers_IDs = customers_IDs[:customers_IDs.shape[0] - 1]

    'encode features'
    le = preprocessing.LabelEncoder()
    encoded_features = le.fit_transform(customers_features)
    features = np.array(encoded_features).reshape((customers_count + 1, features_count))

    'Remove Customers IDs from array'
    features = features[:, 1:]
    encoded_customer_details = features[features.shape[0] - 1]
    'remove the last element from features array'
    features = features[:features.shape[0] - 1]

    k_means = KMeans(init='k-means++', n_clusters=15, n_init=10, max_iter=1000)
    predictions = k_means.fit_predict(features)

    'create dictionary to link between clusters and customers_IDs'
    dictionary_of_clusters = {key: [] for key in set(predictions)}
    for i in range(len(predictions)):
        lable = predictions[i]
        dictionary_of_clusters[lable].append(customers_IDs[i])

    customer_cluster = k_means.predict(encoded_customer_details.reshape((1, -1)))

    cluster_customers = dictionary_of_clusters[customer_cluster[0]]

    'get the most frequent films in customer cluster '
    frequent_films = dbConn.get_most_frequent_films_in_cluster(cluster_customers)

    'sorted frequent films depending on count of it'
    'frequent_films is list of tuples like (film name,count of this film)'
    frequent_films.sort(key=lambda tup: tup[1], reverse=True)
    frequent_films = [(film.strip(), count) for film, count in frequent_films]
    print(frequent_films)
    return frequent_films


def getMovieSelectorUsingClassification(customer_ID):
    'open connection with database'
    dbConn = DataBaseConnection()

    all_customers = dbConn.get_all_customers()
    customers_details = dbConn.get_customer_details(customer_ID)

    'get the movieSelector column and delete it from customer details'
    movies_selectors = []

    customers_features = []
    for customer in all_customers:
        customers_features.extend(customer)

    customers_features.extend(customers_details)

    customers_features = np.array(customers_features).reshape((len(all_customers) + 1, len(customers_details)))
    customersIDs = customers_features[:, 0]
    movies_selectors = customers_features[:, 1]
    customers_features = customers_features[:, 2:]

    'encode features'
    le = preprocessing.LabelEncoder()
    encoded_features = le.fit_transform(list(customers_features.flat))
    features = np.array(encoded_features).reshape((len(all_customers) + 1, len(customers_details) - 2))

    encoded_customer_details = features[features.shape[0] - 1]
    'remove the last element from features array'
    features = features[:features.shape[0] - 1]

    encoded_labels = le.fit_transform(movies_selectors)
    encoded_labels = encoded_labels[: len(encoded_labels) - 1]

    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=.25)

    'Classifiers'
    MLP_classifier = MLPClassifier(alpha=0.1, hidden_layer_sizes=200, max_iter=2000, learning_rate_init=0.01)
    MLP_classifier.fit(X_train, y_train)

    SVC_classifier = SVC(kernel="rbf", C=0.095)
    SVC_classifier.fit(X_train, y_train)

    prediction = SVC_classifier.predict(encoded_customer_details.reshape((1, -1)))
    predictions_dictionary = {}
    for i in range(len(encoded_labels)):
        lable = encoded_labels[i]
        predictions_dictionary[lable] = movies_selectors[i]

    movie_selector = predictions_dictionary[prediction[0]]
    return movie_selector


'Slow function'


def getRecommendedActors(actor_name: str):
    actor_name = actor_name.strip()
    start = time.time()
    'open connection with database'
    dbConn = DataBaseConnection()

    Cutomers_IDs = dbConn.get_all_customersIDs()

    orders = []
    bought_films_by_user = None
    for id in Cutomers_IDs:
        customer_actors = dbConn.get_the_actors_of_customer(id)
        orders.append(customer_actors)

    print(time.time() - start)
    print('Begin Mining')
    association_rules = apriori(orders, min_support=0.010, min_confidence=0.1, min_lift=2, min_length=1)
    association_results = list(association_rules)
    print()
    """
    example about rules:
    RelationRecord(
    items=frozenset({'Affleck, Ben', 'Cage, Nicolas'}),                                                index:0
    support=0.029861111111111113,                                                                      index:1
    ordered_statistics=[OrderedStatistic(items_base=frozenset({'Affleck, Ben'}),                       index:2
    items_add=frozenset({'Cage, Nicolas'}), 
    confidence=0.32209737827715357,
    lift=2.4605847465204302)]
    ) 


    index:2
    OrderedStatistic(items_base=frozenset({'Affleck, Ben'}),                       index:0
    items_add=frozenset({'Cage, Nicolas'}),                                        index:1
    confidence=0.32209737827715357,                                                index:2
    lift=2.4605847465204302)                                                       index:3


    """

    'sort rules depending on confidence of rule'
    association_results.sort(key=lambda tup: tup[2][0][2], reverse=True)

    rules = []
    recommended_actors = []

    for rule in association_results:
        rule_members = rule[2][0]
        rules.append(Rule(list(rule_members[0]), list(rule_members[1]), rule_members[2]))
        if (
                len(list(rule_members[0])) == 1 and
                len(list(rule_members[1])) == 1 and
                list(rule_members[0])[0] == actor_name.strip() and
                not list(rule_members[1])[0] in recommended_actors
        ):
            recommended_actors.append(list(rule_members[1])[0])

    print(time.time() - start)
    if recommended_actors.__contains__(actor_name.strip()):
        recommended_actors.remove(actor_name.strip())

    return recommended_actors


'Fast function'


def getRecommendedActors1(actor_name: str):
    actor_name = actor_name.strip()
    start = time.time()
    'open connection with database'
    dbConn = DataBaseConnection()

    all_customers = dbConn.get_all_customersIDs_with_actors()

    'create orders list'
    orders = []

    for (customer_id, customer_films) in all_customers:
        orders.append(customer_films)

    print(time.time() - start)
    print('Begin Mining')
    association_rules = apriori(orders, min_support=0.010, min_confidence=0.1, min_lift=2, min_length=1)
    association_results = list(association_rules)
    print()
    """
    example about rules:
    RelationRecord(
    items=frozenset({'Affleck, Ben', 'Cage, Nicolas'}),                                                index:0
    support=0.029861111111111113,                                                                      index:1
    ordered_statistics=[OrderedStatistic(items_base=frozenset({'Affleck, Ben'}),                       index:2
    items_add=frozenset({'Cage, Nicolas'}), 
    confidence=0.32209737827715357,
    lift=2.4605847465204302)]
    ) 


    index:2
    OrderedStatistic(items_base=frozenset({'Affleck, Ben'}),                       index:0
    items_add=frozenset({'Cage, Nicolas'}),                                        index:1
    confidence=0.32209737827715357,                                                index:2
    lift=2.4605847465204302)                                                       index:1


    """

    'sort rules depending on confidence of rule'
    association_results.sort(key=lambda tup: tup[2][0][2], reverse=True)

    rules = []
    recommended_actors = []

    for rule in association_results:
        rule_members = rule[2][0]
        rules.append(Rule(list(rule_members[0]), list(rule_members[1]), rule_members[2]))
        if len(list(rule_members[0])) == 1 and len(list(rule_members[1])) == 1 and list(rule_members[0])[
            0] == actor_name.strip() and not list(rule_members[1])[0] in recommended_actors:
            recommended_actors.append(list(rule_members[1])[0])

    print(time.time() - start)
    if recommended_actors.__contains__(actor_name.strip()):
        recommended_actors.remove(actor_name.strip())

    return recommended_actors


def parse_args():
    parser = argparse.ArgumentParser(prog="Movies Recommendation System")

    parser.add_argument('--method', nargs='?', default='getRecommendedFilmsUsingApriori',
                        help='The method name that you want using it')

    parser.add_argument('--customerID', nargs='?',
                        help='The ID of current customer')

    parser.add_argument('--movie_name', nargs='?', default='',
                        help='The movie name that you want get similar movies for it')

    parser.add_argument('--actor_name', nargs='?', default='',
                        help='The actor name that you want get similar movies for it')

    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    # method = args.method
    # customer_id = args.customerID
    # movie_name = args.movie_name
    # actor_name = args.actor_name
    # top_n = args.top_n

    # print(getRecommendedFilmsUsingApriori(879061))

    # print(getRecommendedFilmsUsingApriori1(879061))
    # print(getRecommendedFilmsByFilmNameUsingApriori("Godfather, The"))
    # print(getRecommendedFilmsByFilmNameUsingApriori1("Braveheart"))

    # print(getRecommendedFilmsUsingClustering(878908))
    # print( getMovieSelectorUsingClassification(878908))
    # print(getRecommendedActors("Duchovny, David"))
    print(getRecommendedActors1("Hopkins, Anthony"))
