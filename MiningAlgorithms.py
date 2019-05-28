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


class Mining:
    def __init__(self, connection: DataBaseConnection) -> None:
        super().__init__()
        self.connection = connection

    def get_recommended_movies(self, movie_name: str):
        film_name = movie_name.strip()
        start = time.time()
        'open connection with database'

        all_customers = self.connection.get_all_customersIDs_with_films()

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

    def get_user_recommended_movies(self, customer_id):
        customer_id = int(customer_id)

        start = time.time()
        'open connection with database'

        all_customers = self.connection.get_all_customersIDs_with_films()

        'create orders list'
        orders = []
        bought_films_by_user = None
        for (c_id, c_films) in all_customers:
            if c_id == customer_id:
                bought_films_by_user = c_films
            orders.append(c_films)

        if bought_films_by_user is None:
            return []
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
            if rule_members[0].issubset(set(bought_films_by_user)) and not rule_members[1].issubset(
                    set(recommended_films)):
                recommended_films.append(list(rule_members[1])[0])

        print(time.time() - start)
        return recommended_films

    def get_movie_selectors(self, customer_id):

        all_customers = self.connection.get_all_customers()
        customers_details = self.connection.get_customer_details(customer_id)

        customers_features = []
        for customer in all_customers:
            customers_features.extend(customer)

        customers_features.extend(customers_details)

        customers_features = np.array(customers_features).reshape((len(all_customers) + 1, len(customers_details)))
        movies_selectors = customers_features[:, 1]
        customers_features = customers_features[:, 2:]

        'encode features'
        le = preprocessing.LabelEncoder()
        encoded_features = le.fit_transform(list(customers_features.flat))
        features = np.array(encoded_features).reshape((len(all_customers) + 1, len(customers_details) - 2))

        encoded_customer_details = features[features.shape[0] - 1]
        'remove the last element from features array'
        features = features[:features.shape[0] - 1]
        movies_selectors = movies_selectors[:movies_selectors.shape[0] - 1]

        encoded_labels = le.fit_transform(movies_selectors)

        X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=.25)

        'Classifiers'
        MLP_classifier = MLPClassifier(alpha=0.1, hidden_layer_sizes=200, max_iter=2000, learning_rate_init=0.01)
        MLP_classifier.fit(X_train, y_train)

        SVC_classifier = SVC(kernel="rbf", C=0.095)
        SVC_classifier.fit(X_train, y_train)

        prediction = SVC_classifier.predict(encoded_customer_details.reshape((1, -1)))


        movie_selector_index = list(encoded_labels).index(prediction[0])
        movie_selector = movies_selectors[movie_selector_index]
        return movie_selector

    def get_actor_friends(self, actor_name: str):
        actor_name = actor_name.strip()
        start = time.time()

        all_customers = self.connection.get_all_customersIDs_with_actors()

        'create orders list'
        orders = []

        for (customer_id, customer_films) in all_customers:
            orders.append(customer_films)

        print(time.time() - start)
        print('Begin Mining')
        association_rules = apriori(orders, min_support=0.010, min_confidence=0.1, min_lift=2, min_length=1)
        association_results = list(association_rules)
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
