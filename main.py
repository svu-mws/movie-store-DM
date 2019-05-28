from flask import jsonify
from flask import Flask

from DataBaseConnection import DataBaseConnection
from MiningAlgorithms import Mining

app = Flask(__name__)
db_connection = DataBaseConnection()
mining = Mining(connection=db_connection)


# http://127.0.0.1:5000/recommended_movies/Braveheart
@app.route('/recommended_movies/<movie_name>')
def get_similar_movies(movie_name):
    # Customers who show this movie also show
    result = mining.get_recommended_movies(movie_name=movie_name)
    print('Result: ', result)
    resp = jsonify(result=result)
    resp.status_code = 200
    return resp


# http://127.0.0.1:5000/user_movies/879061
@app.route('/user_movies/<user_id>')
def get_recommended_movies(user_id):
    # Get personalised recommendations
    result = mining.get_user_recommended_movies(user_id)
    resp = jsonify(result=result)
    resp.status_code = 200
    return resp


# http://127.0.0.1:5000/movie_selectors/878908
@app.route('/movie_selectors/<user_id>')
def predict_movie_selectors(user_id):
    # Who select the movie
    result = mining.get_movie_selectors(customer_id=user_id)
    resp = jsonify(result=result)
    resp.status_code = 200
    return resp

# http://127.0.0.1:5000/actor/Hopkins,%20Anthony/friends
@app.route('/actor/<actor_name>/friends')
def get_actor_friends(actor_name):
    # Who are actor’s friends
    result = mining.get_actor_friends(actor_name=actor_name)
    resp = jsonify(result=result)
    resp.status_code = 200
    return resp


if __name__ == "__main__":
    app.run()
