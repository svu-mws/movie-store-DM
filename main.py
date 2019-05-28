from flask import jsonify
from flask import Flask

from DataBaseConnection import DataBaseConnection
from MiningAlgorithms import Mining

app = Flask(__name__)
db_connection = DataBaseConnection()
mining = Mining(connection=db_connection)


@app.route('/sim_movies/<movie_id>')
def get_similar_movies(movie_id):
    # Customers who show this movie Also show
    result = mining.get_recommended_movies(movie_id=movie_id)
    resp = jsonify(result)
    resp.status_code = 200
    return resp


@app.route('/user_movies/<user_id>')
def get_recommended_movies(user_id):
    # Get personalised recommendations
    result = mining.get_user_recommended_movies(customer_id=user_id)
    resp = jsonify(result)
    resp.status_code = 200
    return resp


@app.route('/movie_selectors/<user_id>')
def predict_movie_selectors(user_id):
    # Who select the movie
    result = mining.get_movie_selectors(customer_id=user_id)
    resp = jsonify(result)
    resp.status_code = 200
    return resp


@app.route('/actor/<actor_id>/friends')
def get_actor_friends(actor_id):
    # Who are ac    tor’s friends
    result = mining.get_actor_friends(actor_id=actor_id)
    resp = jsonify(result)
    resp.status_code = 200
    return resp


if __name__ == "__main__":
    app.run()
