from roboflow import Roboflow

rf = Roboflow(api_key="qVJj3gr5RajMieFy7sqh")  # free at roboflow.com

# Best elephant dataset — 2030 images
project = rf.workspace("ultimateele03").project("elephants-wz5qt")
dataset = project.version(4).download("yolov8", 
    location=r"D:\projects\ecosentinel\elephants-wz5qt")