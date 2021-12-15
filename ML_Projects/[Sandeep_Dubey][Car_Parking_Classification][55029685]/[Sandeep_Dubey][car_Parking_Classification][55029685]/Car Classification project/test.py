import os
import tensorflow as tf
import numpy as np
import cv2
import csv

RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

TEST_IMAGES_DIR = os.getcwd() + "/test_images"

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

def main():

    print("starting program . . .")

    if not checkIfNecessaryPathsAndFilesExist():
        return

    results = {}

    # get a list of classifications from the labels file
    classifications = []
    # for each line in the label file . . .
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(classification)
    print("classifications = " + str(classifications))

    # load the graph from file
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        graphDef = tf.GraphDef()
        graphDef.ParseFromString(retrainedGraphFile.read())
        _ = tf.import_graph_def(graphDef, name='')

    if not os.path.isdir(TEST_IMAGES_DIR):
        print("the test image directory does not seem to be a valid directory, check file / directory paths")
        return
    with tf.Session() as sess:
        for fileName in os.listdir(TEST_IMAGES_DIR):
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue
            print(fileName)

            # get the file name and full path of the current image file
            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            # attempt to open the image with OpenCV
            openCVImage = cv2.imread(imageFileWithPath)
            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                continue

            # get the final tensor from the graph
            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            # convert the OpenCV image to a TensorFlow image
            tfImage = np.array(openCVImage)[:, :, 0:3]

            # run the network to get the predictions
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            # sort predictions from most confidence to least confidence
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

            print("---------------------------------------")

            onMostLikelyPrediction = True

            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                # if the classification ends with the letter "s", remove the "s" to change from plural to singular
                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]

                # get confidence, then get confidence rounded to 2 places after the decimal
                confidence = predictions[0][prediction]

                if onMostLikelyPrediction:
                    # get the score as a %
                    if strClassification not in results:
                        results[strClassification] = 1
                    else:
                        results[strClassification] += 1

                    scoreAsAPercent = confidence * 100.0
                    # show the result to std out
                    print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(
                        scoreAsAPercent) + "% confidence")
                    # write the result on the image
                    writeResultOnImage(openCVImage,
                                       strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    cv2.imshow(fileName, openCVImage)

                    onMostLikelyPrediction = False
                    print(strClassification)
                # for any prediction, show the confidence as a ratio to five decimal places
                print(strClassification + " (" + "{0:.5f}".format(confidence) + ")")


            print(results)
            #csv_columns = ['Model', 'Count']
            with open('dict.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in results.items():
                    writer.writerow([key, value])

            cv2.waitKey()
            cv2.destroyAllWindows()

    # write the graph to file so we can view with TensorBoard
    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()


def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the test images?')
        print('')
        return False

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return False

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return False
    return True

def writeResultOnImage(openCVImage, resultText):

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    fontScale = 1.0
    fontThickness = 2

    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE,
                fontThickness)
if __name__ == "__main__":
    main()
