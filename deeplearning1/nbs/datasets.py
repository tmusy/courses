import urllib
import zipfile
import os
from os.path import expanduser, join
home = expanduser("~")
datasets_root = join(home, 'datasets')


def getunzipped(theurl, thedir):
  name = os.path.join(thedir, 'tmp.zip')
  try:
    name, hdrs = urllib.urlretrieve(theurl, name)
  except IOError, e:
    print "Can't retrieve %r to %r: %s" % (theurl, thedir, e)
    return
  try:
    z = zipfile.ZipFile(name)
  except zipfile.error, e:
    print "Bad zipfile (from %r): %s" % (theurl, e)
    return
  for n in z.namelist():
    dest = os.path.join(thedir, n)
    destdir = os.path.dirname(dest)
    if not os.path.isdir(destdir):
      os.makedirs(destdir)
    data = z.read(n)
    f = open(dest, 'w')
    f.write(data)
    f.close()
  z.close()
  os.unlink(name)


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
        pass

    def register_dataset(self, name, url_train, url_test=''):
        getunzipped(url_train, self.datasets_root)


if __name__ == '__main__':
    datasets = Datasets()
    ds = datasets.list()
    print(ds)
    dg_submission = 'https://www.kaggle.com/c/dogs-vs-cats/download/sampleSubmission.csv'
    dg_train = 'https://www.kaggle.com/c/dogs-vs-cats/download/train.zip'
    datasets.register_dataset('dogscats', dg_train)
