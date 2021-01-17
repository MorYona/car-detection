import cv2
import numpy as np
import os.path
import json

#TODO: before run, change my_dir and mu_images_dir, and make sure the cfg and weights and  is in the same directory
my_dir ='D:/yolo'
my_images_dir = 'D:/yolo/images'

'''load classes'''
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


'''load the neural network'''
net = cv2.dnn.readNet(os.path.join(my_dir, 'yolov3.weights'),os.path.join(my_dir, 'yolov3.cfg'))
layers_names = net.getLayerNames()
#the output layers hold the detection of the object
output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
#the colors will be used to print the boxes in random colors
colors = np.random.uniform(0, 255, size=(len(classes), 3))



json_dic = {}
i=0
for image_file in os.listdir(my_images_dir):  # iterate over all the files in the images directory
    if not image_file.endswith('.jpg'): #if file is not an image continue
        continue
    #print(image_file)
    i+=1
    json_dic[f"image {i}"] = []
    img = cv2.imread(os.path.join(my_images_dir,image_file))
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape #later used to locate detection

    ''' Detecting objects'''
    #blob is a format that yolo can read
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers) #outs is the prediction of the network
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores) #take the class with the biggest value
            confidence = scores[class_id]
            #because the task was to locate car with 0.7 threshhold
            if confidence > 0.7:
                #print('detection!')
                #Object detected
                #add to json what the net detected and the confidence
                json_dic[f"image {i}"].append({
                   'object': classes[class_id],
                    'confidence': confidence
                })
                center_x = int(detection[0] * width) #the center of the detection
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
   # cv2.imwrite(os.path.join('D:/yolo/process_images',image_file), img)
   #  print(json_dic)
json_object = json.dumps(int(json_dic), indent=4)
with open('json_file.json', "w") as outfile:
    outfile.write(json_object)
    outfile.write('\n')

#show the last image
cv2.imshow("Image", img)

cv2.waitKey(0) #wait for the user to click X
cv2.destroyAllWindows()