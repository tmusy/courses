import os
from os.path import expanduser, join
from keras.preprocessing.image import img_to_array

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

home = expanduser("~")
datasets_root = join(home, 'datasets')


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


class Datasets(object):

    def __init__(self):
        self.datasets_root = datasets_root
        self.datasets = {}
        self._find_datasets()

    def _find_datasets(self):
        for dir_name, subdirs, files in os.walk(self.datasets_root):
            for fname in files:
                if fname == 'dl_set':
                    with open(join(self.datasets_root, dir_name, fname)) as infile:
                        line = infile.readline()
                        __, name = line.split('=')
                        name = name.strip()
                    self.datasets[name] = join(self.datasets_root, dir_name)

    def list(self):
        return self.datasets

    def load_data(self, name):
        X = []
        y = []
        pictures_dir = self.datasets.get(name)
        for dir_name, subdirs, files in os.walk(pictures_dir + '/sample'):
            label = dir_name.split('/')[-1]
            for fname in files:
                if fname.endswith('jpg'):
                    img = load_img(os.path.join(dir_name, fname))
                    arr = img_to_array(img)
                    X.append(arr)
                    y.append(label)
        return X, y

    def register_dataset(self, name, url_train, url_test=''):
        pass


if __name__ == '__main__':
    datasets = Datasets()
    ds = datasets.list()
    print(ds)
    datasets.load_data('dogscats')
