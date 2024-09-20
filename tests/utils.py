import os

test_record_paths = {"cs": os.path.join(os.path.dirname(__file__), "res/test_cs.wav"),
                     "en": os.path.join(os.path.dirname(__file__), "res/evacuation_en.wav")}
test_record_paths["default"] = test_record_paths["cs"]
