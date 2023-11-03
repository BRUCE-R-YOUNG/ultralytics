from roboflow import Roboflow
rf = Roboflow(api_key="Zw3apzuo2FVEkUK8qheP")
project = rf.workspace().project("detect-bus")
model = project.version(2).model

# infer on a local image
print(model.predict("src/24552170_m.jpg", confidence=40, overlap=30).json())

