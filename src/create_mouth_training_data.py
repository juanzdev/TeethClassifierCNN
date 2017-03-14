#generates the mouth data from the full image
from mouth_detector_dlib import mouth_detector
classified_data_path = "../img/classified_data"
mouth_data_path = "../img/mouth_data"

detector = mouth_detector()
detector.mouth_detect_bulk(classified_data_path,mouth_data_path)
