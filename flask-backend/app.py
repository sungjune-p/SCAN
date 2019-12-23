import os
from flask import Flask, flash, session, redirect, url_for, request, render_template, current_app, jsonify, send_file
from flask_cors import CORS, cross_origin

import json
import logging
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import scan_test


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:4444"}})
# api = Api(app=app)
# ns = api.namespace('vbs', description='design vbs web')
img_embs = np.load('/mnt/hard2/DB_DATA/scan_out/img_embs_1.npy')


@app.route('/getString', methods=['POST'])
def getString():
    # string = request.data
    string = request.form['String']
    if string != None:
        current_app.logger.info("Get string succesfully from flask 5000")
        current_app.logger.info(string)
        
        A = scan_test.execute(string, img_embs)
        current_app.logger.info(A)
        jsonFile = jsonify(A)
	
        return jsonFile
    else:
        return "failed"


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True, port=int(os.getenv('PORT', 4444)))
