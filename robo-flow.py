from roboflow import Roboflow
rf = Roboflow(api_key="RAlmNnxtyUilnVrpEtGT")
project = rf.workspace().project("rpi-cows")
model = project.version(1).model

# infer on a local image
#print(model.predict("cow.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("cow.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())