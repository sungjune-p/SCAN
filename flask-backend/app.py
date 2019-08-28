import os
from flask import Flask, flash, session, redirect, url_for, request, render_template, current_app, jsonify, send_file
from flask_cors import CORS, cross_origin

import json
import logging
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# import test


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:5000"}})
# api = Api(app=app)
# ns = api.namespace('vbs', description='design vbs web')


@app.route('/getString', methods=['POST'])
def getString():
    # string = request.data
    string = request.form['String']
    if string != None:
        current_app.logger.info("Get string succesfully from flask 5000")
        current_app.logger.info(string)
        
        a = [1, 2, 3, 4, 5, 6]
        result = dict(output = None)
        result['output'] = a
        jsonFile = json.dumps(result)
        request.post("http://localhost:5000/getScanResult", data={'Results': jsonFile})
#    test.execute(string)

    return "Good to go"


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True, port=int(os.getenv('PORT', 4444)))
