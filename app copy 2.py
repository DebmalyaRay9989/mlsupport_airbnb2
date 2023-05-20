from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    temp_array = list()
    if request.method == 'POST':
        City = request.form('City'),
        Price = float(request.form('Price')),
        Day = request.form('Day'),
        room_type = request.form('Room Type'),
        shared_room = float(request.form('Shared Room')),
        private_room = float(request.form('Private Room')),
        person_capacity = float(request.form('Person Capacity')),
        superhost = float(request.form('Superhost')),
        multiple_rooms = float(request.form('Multiple Rooms')),
        business = float(request.form('Business')),
        cleanliness_rating = float(request.form('Cleanliness Rating')),
        guest_satisfaction = float(request.form('Guest Satisfaction')),
        bedrooms = int(request.form('Bedrooms')),
        city_center_km = float(request.form('City Center (km)')),
        metro_distance_km = float(request.form('Metro Distance (km)')),
        attraction_index = float(request.form('Attraction Index')),
        normalised_attraction_index = float(request.form('Normalised Attraction Index')),
        restraunt_index = float(request.form('Restraunt Index')),
        normalised_restraunt_index = float(request.form('Normalised Restraunt Index'))
		emp_array = temp_array + [City,Price,Day,room_type,shared_room,private_room,person_capacity,superhost,multiple_rooms,business,cleanliness_rating,guest_satisfaction,bedrooms,city_center_km,metro_distance_km,attraction_index,normalised_attraction_index,restraunt_index,normalised_restraunt_index]
		data = np.array([temp_array])
		prediction = int(model.predict(data)[0])

    #    if prediction == 0:
    #        return render_template('index.html',prediction_text="The Water Is Safe For Drinking ! {}".format(prediction))
    #    elif prediction == 1:
    #        return render_template('index.html',prediction_text="The Water Is Not Safe For Drinking! {}".format(prediction))
    #else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True)

    
