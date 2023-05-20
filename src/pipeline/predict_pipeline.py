import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 City: str,
                 Day: str,
                 room_type: str,
                 shared_room: int,
                 private_room: int,
                 person_capacity: int,
                 superhost: int,
                 multiple_rooms: int,
                 business: int,
                 cleanliness_rating: int,
                 guest_satisfaction: int,
                 bedrooms: int,
                 city_center_km: int,
                 metro_distance_km: int,
                 attraction_index: int,
                 normalised_attraction_index: int,
                 restraunt_index: int,
                 normalised_restraunt_index: int,
                 Price: int):

        self.City = City

        self.Day = Day

        self.room_type = room_type

        self.shared_room = shared_room

        self.private_room = private_room

        self.person_capacity = person_capacity

        self.superhost = superhost

        self.multiple_rooms = multiple_rooms

        self.cleanliness_rating = cleanliness_rating

        self.guest_satisfaction = guest_satisfaction

        self.bedrooms = bedrooms

        self.city_center_km = city_center_km

        self.metro_distance_km = metro_distance_km

        self.attraction_index = attraction_index

        self.normalised_attraction_index = normalised_attraction_index

        self.restraunt_index = restraunt_index

        self.normalised_restraunt_index = normalised_restraunt_index

        self.Price = Price

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "City": [self.City],
                "Day": [self.Day],
                "room_type": [self.room_type],
                "shared_room": [self.shared_room],
                "private_room": [self.private_room],
                "person_capacity": [self.person_capacity],
                "superhost": [self.superhost],
                "multiple_rooms": [self.multiple_rooms],
                "cleanliness_rating": [self.cleanliness_rating],
                "guest_satisfaction": [self.guest_satisfaction],
                "bedrooms": [self.bedrooms],
                "city_center_km": [self.city_center_km],
                "metro_distance_km": [self.metro_distance_km],
                "attraction_index": [self.attraction_index],
                "normalised_attraction_index": [self.normalised_attraction_index],
                "restraunt_index": [self.restraunt_index],
                "normalised_restraunt_index": [self.normalised_restraunt_index],
                "Price": [self.Price],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

