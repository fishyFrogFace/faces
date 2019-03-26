# import the necessary packages
import os


class ImagePaths(object):
    IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def list_images(self, base_path, contains=None):
        return self._list_files(base_path, valid_extensions=self.IMAGE_TYPES, contains=contains)

    def _list_files(self, base_path, valid_extensions=None, contains=None):
        for (rootDir, dirNames, filenames) in os.walk(base_path):
            for filename in filenames:
                if contains is not None and filename.find(contains) == -1:
                    continue

                extension = filename[filename.rfind("."):].lower()
                if valid_extensions is None or extension.endswith(valid_extensions):
                    image_path = os.path.join(rootDir, filename)
                    yield image_path
