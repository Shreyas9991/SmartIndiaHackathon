@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        Articles = flask.request.form['Articles']
        
        input_variables = pd.DataFrame([[Articles]],
                                       columns=['Articles'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Articles':Articles},
                                     result=prediction,
                                     )