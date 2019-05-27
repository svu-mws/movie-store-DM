import mysql.connector as mysql


class DataBaseConnection:

    def __init__(self, server_name: str = "localhost", user: str = "root", password: str= "", database_name: str ="moviesdb"):
        self.Connection = mysql.connect(
            host=server_name,
            user=user,
            passwd=password,
            database=database_name)


    def get_all_customersIDs(self, cursor):

        query = 'SELECT DISTINCT CustomerID FROM Movies'
        'The output of the below statement is list with elements like (userID,) as tuple'
        cursor.execute(query)
        all_customers = cursor.fetchall()

        customers = []
        for i in range(len(all_customers)):
            customers.append(all_customers[i][0])
        return customers

    def get_theBought_films_by_customer(self, cursor, customer_ID):

        query = 'SELECT DISTINCT Movie FROM Movies WHERE CustomerID =' + str(customer_ID)
        'The output of the below statement is list with elements like (MovieName,) as tuple'
        cursor.execute(query)
        customer_films = cursor.fetchall()

        films = []
        for film in customer_films:
            films.append(film[0].strip())
        return films

    def get_the_actors_of_customer(self, cursor, customerID):

        query = 'SELECT DISTINCT Actor FROM Actors WHERE CustomerID =' + str(customerID)
        'The output of the below statement is list with elements like (ActorName,) as tuple'
        cursor.execute(query)
        actors_of_customer = cursor.fetchall()

        Actors = []
        for actor in actors_of_customer:
            Actors.append(actor[0].strip())

        return Actors

    def get_all_customers(slef, cursor):

        query = """SELECT CustomerID, `Movie Selector`, Age, `Education Level`, Gender, `Home Ownership`, `Internet Connection`, 
                `Marital Status`, `Num Bathrooms`, `Num Bedrooms`, `Num Cars`, `Num Children`, `Num TVs`, `PPV Freq`, `Buying Freq`, 
                Format, `Renting Freq`, `Viewing Freq`, `Theater Freq`, `TV Movie Freq`, `TV Signal` FROM Customers"""

        'The output of the below statement is list of tuples'
        cursor.execute(query)
        all_customers = cursor.fetchall()

        customers = []
        for i in range(len(all_customers)):
            customer = list(all_customers[i])
            customer = [value if value is not None else -1 for value in customer]
            customers.append(customer)
        return customers

    def get_customers_features(self, cursor):

        query = """SELECT CustomerID, `Movie Selector`, Age, `Education Level`, Gender, `Home Ownership`, `Internet Connection`, 
                `Marital Status`, `Num Bathrooms`, `Num Bedrooms`, `Num Cars`, `Num Children`, `Num TVs`, `PPV Freq`, `Buying Freq`, 
                Format, `Renting Freq`, `Viewing Freq`, `Theater Freq`, `TV Movie Freq`, `TV Signal` FROM Customers"""

        'The output of the below statement is list of tuples'
        cursor.execute(query)
        all_customers = cursor.fetchall()
        first_one = all_customers[0]

        all_customers_features = []
        for row in all_customers:
            customer = list(row)
            customer = [value if value is not None else -1 for value in customer]
            all_customers_features.extend(customer)
        return all_customers_features, len(all_customers), len(first_one)


    def get_customer_details(self, cursor, customer_ID):

        query = """SELECT CustomerID, `Movie Selector`, Age, `Education Level`, Gender, `Home Ownership`, `Internet Connection`, 
                `Marital Status`, `Num Bathrooms`, `Num Bedrooms`, `Num Cars`, `Num Children`, `Num TVs`, `PPV Freq`, `Buying Freq`, 
                Format, `Renting Freq`, `Viewing Freq`, `Theater Freq`, `TV Movie Freq`, `TV Signal` FROM Customers WHERE CustomerID = """ + str(customer_ID)

        'The output of the below statement is tuple'
        cursor.execute(query)
        customer = cursor.fetchone()
        customer = [value if value is not None else -1 for value in customer]
        return list(customer)


    def get_most_frequent_films_in_cluster(self, cursor, customers_list):
        customer_list_sql = "("
        for i in range(len(customers_list)):
            customer_list_sql += customers_list[i]
            if i != len(customers_list) - 1:
                customer_list_sql += ","

        customer_list_sql += ")"
        query = "SELECT Movie , COUNT(*) FROM Movies WHERE CustomerID IN " + customer_list_sql + " GROUP BY Movie"
        cursor.execute(query)
        films = cursor.fetchall()
        return films


    def get_all_customersIDs_with_films(self, cursor):

        query1 = 'SET GLOBAL  group_concat_max_len = 9999999;'
        query2 = 'SELECT CustomerID, GROUP_CONCAT( Movie SEPARATOR "|") FROM Movies GROUP BY CustomerID;'

        'The output of the below statement is list of tuples like (customerID, customer films)'
        cursor.execute(query1)
        cursor.execute(query2)
        customers_films = cursor.fetchall()
        films = []

        for customer_detail in customers_films:
            customer_id = customer_detail[0]
            customer_films = customer_detail[1].split("|")
            customer_films = [film.strip() for film in customer_films]
            films.append((customer_id, customer_films))

        return films


    def get_all_customersIDs_with_actors(self, cursor):

        query1 = 'SET GLOBAL  group_concat_max_len = 9999999;'
        query2 = 'SELECT CustomerID, GROUP_CONCAT( Actor SEPARATOR "|") FROM Actors GROUP BY CustomerID;'

        'The output of the below statement is list of tuples like (customerID, customer actors)'
        cursor.execute(query1)
        cursor.execute(query2)
        customers_actors = cursor.fetchall()
        actors = []

        for customer_detail in customers_actors:
            customer_id = customer_detail[0]
            customer_films = customer_detail[1].split("|")
            customer_films = [film.strip() for film in customer_films]
            actors.append((customer_id, customer_films))

        return actors


"""
TEST
"""
