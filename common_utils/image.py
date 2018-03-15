import cv2


def get_border_size(image, target_multiplier):
    image_shape = image.shape
    image_height = image_shape[0]
    image_width = image_shape[1]

    mul_height = (image_height // target_multiplier) * target_multiplier
    mul_width = (image_width // target_multiplier) * target_multiplier

    target_height = mul_height + target_multiplier
    target_width = mul_width + target_multiplier

    border_height = (target_height - image_height) // 2
    border_width = (target_width - image_width) // 2

    border_size = (
        border_height,
        target_height - image_height - border_height,
        border_width,
        target_width - image_width - border_width,
    )

    return border_size


def make_border(image, border_size):
    image = cv2.copyMakeBorder(
        image, border_size[0], border_size[1], border_size[2], border_size[3],
        cv2.BORDER_REFLECT
    )
    return image
