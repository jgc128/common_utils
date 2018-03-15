import cv2


def load_image(filename, grayscale=False):
    if not grayscale:
        img = cv2.imread(str(filename))
    else:
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)

    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
