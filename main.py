import os
from download import download_files
from tf import model_load , tensorflow_clear, tensorflow_on_gpu, preprocessing_image
from yolo_custom import decode_netout, correct_yolo_boxes, do_nms, draw_boxes, get_boxes
from file_manager import get_json


if __name__=='__main__':
    # urls to download yolo model
    urls = ['https://pjreddie.com/media/files/yolov3.weights', 'https://pjreddie.com/media/files/yolov3-tiny.weights']
    download_files(urls)
    # test gpu - cuda
    tensorflow_on_gpu()
    # reset memory
    tensorflow_clear()
    # paths necessary
    path = os.getcwd()
    path_weights = os.path.join(path, 'weights')
    path_images = os.path.join(path, 'images')
    # red information
    path_information = os.path.join(path, 'information.json')
    information = get_json(path_information)
    labels = information['labels']
    anchors = information['anchors']
    class_threshold = information['class_threshold']
    input = information['input']
    print(input)
    # load model
    model_name = 'yolov3.weights-tiny.h5'
    path_model = os.path.join(path_weights, model_name)
    model = model_load(path_model)
    # load image and preprocessing
    photo_filename = 'person.jpeg'
    image_path = os.path.join(path_images, photo_filename)
    image, image_w, image_h = preprocessing_image(image_path, input_size=input)    
    input_w, input_h = 416, 416
    # yolo prediction
    prediction = model.predict(image)
    print([a.shape for a in prediction])
    """
    boxes = list()
    for i in range(len(prediction)):
        # decode the output of the network
        boxes += decode_netout(prediction[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])
    # draw what we found
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
    """