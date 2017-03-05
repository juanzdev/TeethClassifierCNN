#generates the mouth data from the full image
from mouth_slicer import mouth_detect_bulk

classified_data_path = "../img/classified_data"
mouth_data_path = "../img/mouth_data"

mouth_detect_bulk(classified_data_path,mouth_data_path)
