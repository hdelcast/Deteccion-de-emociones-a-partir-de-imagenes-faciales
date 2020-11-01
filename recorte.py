import itertools

import cv2
import numpy as np
import os
import sys
from PIL import Image

FIXEXP = True
MINFACE = 8 
INCREMENT = 0.06
GAMMA_THRES = 0.001
GAMMA = 0.90
FACE_RATIO = 6 
QUESTION_OVERWRITE = "Overwrite?"

CV2_FILETYPES = [
    ".pgm",
    ".png",
    ".ppm",
    ".ras",
    ".sr",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
    ".dib",
    ".jp2",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pbm",

]

PILLOW_FILETYPES = [
    ".eps",
    ".gif",
    ".icns",
    ".ico",
    ".im",
    ".msp",
    ".pcx",
    ".sgi",
    ".spi",
    ".xbm",
]

CASCFILE = "haarcascade_frontalface_default.xml"

COMBINED_FILETYPES = CV2_FILETYPES + PILLOW_FILETYPES
INPUT_FILETYPES = COMBINED_FILETYPES + [s.upper() for s in COMBINED_FILETYPES]


class ImageReadError(BaseException):
    pass


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def intersect(v1, v2):
    a1, a2 = v1
    b1, b2 = v2
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db).astype(float)
    num = np.dot(dap, dp)
    return (num / denom) * db + b1


def distance(pt1, pt2):
    distance = np.linalg.norm(pt2 - pt1)
    return distance


def bgr_to_rbg(img):
    dimensions = len(img.shape)
    if dimensions == 2:
        return img
    return img[..., ::-1]


def gamma(img, correction):
    img = cv2.pow(img / 255.0, correction)
    return np.uint8(img * 255)


def check_underexposed(image, gray):
    uexp = cv2.calcHist([gray], [0], None, [256], [0, 256])
    if sum(uexp[-26:]) < GAMMA_THRES * sum(uexp):
        image = gamma(image, GAMMA)
    return image


def check_positive_scalar(num):
    if num > 0 and not isinstance(num, str) and np.isscalar(num):
        return int(num)
    raise ValueError(">0")


def open_file(input_filename):
    extension = os.path.splitext(input_filename)[1].lower()

    if extension in CV2_FILETYPES:
        return cv2.imread(input_filename)
    if extension in PILLOW_FILETYPES:
        with Image.open(input_filename) as img_orig:
            return np.asarray(img_orig)
    return None


class Recortar:

    def __init__(
        self, width=500, height=500, face_percent=50, padding=None, fix_gamma=True,
    ):
        self.height = check_positive_scalar(height)
        self.width = check_positive_scalar(width)
        self.aspect_ratio = width / height
        self.gamma = fix_gamma

        # Face percent
        if face_percent > 100 or face_percent < 1:
            fp_error = "1-100"
            raise ValueError(fp_error)
        self.face_percent = check_positive_scalar(face_percent)

        # XML Resource
        directory = os.path.dirname(sys.modules["recorte"].__file__)
        self.casc_path = os.path.join(directory, CASCFILE)

    def cortar(self, path_or_array):

        if isinstance(path_or_array, str):
            image = open_file(path_or_array)
        else:
            image = path_or_array

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = image

        try:
            img_height, img_width = image.shape[:2]
        except AttributeError:
            raise ImageReadError
        minface = int(np.sqrt(img_height ** 2 + img_width ** 2) / MINFACE)

        face_cascade = cv2.CascadeClassifier(self.casc_path)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(minface, minface),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,
        )

        if len(faces) == 0:
            return None

        x, y, w, h = faces[-1]
        pos = self._crop_positions(img_height, img_width, x, y, w, h,)

        image = image[pos[0] : pos[1], pos[2] : pos[3]]

        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

        if self.gamma:
            image = check_underexposed(image, gray)
        return bgr_to_rbg(image)

    def _determine_safe_zoom(self, imgh, imgw, x, y, w, h):

        corners = itertools.product((x, x + w), (y, y + h))
        center = np.array([x + int(w / 2), y + int(h / 2)])
        i = np.array(
            [(0, 0), (0, imgh), (imgw, imgh), (imgw, 0), (0, 0)]
        ) 

        image_sides = [(i[n], i[n + 1]) for n in range(4)]

        corner_ratios = [self.face_percent]  
        for c in corners:
            corner_vector = np.array([center, c])
            a = distance(*corner_vector)
            intersects = list(intersect(corner_vector, side) for side in image_sides)
            for pt in intersects:
                if (pt >= 0).all() and (pt <= i[2]).all():  
                    dist_to_pt = distance(center, pt)
                    corner_ratios.append(100 * a / dist_to_pt)
        return max(corner_ratios)

    def _crop_positions(
        self, imgh, imgw, x, y, w, h,
    ):
        zoom = self._determine_safe_zoom(imgh, imgw, x, y, w, h)

        if self.height >= self.width:
            height_crop = h * 100.0 / zoom
            width_crop = self.aspect_ratio * float(height_crop)
        else:
            width_crop = w * 100.0 / zoom
            height_crop = float(width_crop) / self.aspect_ratio

        xpad = (width_crop - w) / 2
        ypad = (height_crop - h) / 2

        h1 = x - xpad
        h2 = x + w + xpad
        v1 = y - ypad
        v2 = y + h + ypad

        return [int(v1), int(v2), int(h1), int(h2)]

