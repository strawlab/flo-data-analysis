import yaml
import os

class ConfigLoader:
    def __init__(self, path: str):
        self.conf_path, _ = os.path.split(os.path.abspath(path))
        self.conf = yaml.safe_load(open(path).read())
        self.conf_filename = path
        _, filename = os.path.split(path)
        self.name, _ = os.path.splitext(filename)
        self.output_dir = self.get_filename('output_dir')

    def _rel_to_abs(self, path: str) -> str:
        """convert relative paths to absolute paths"""
        if not os.path.isabs(path):
            path = os.path.join(self.conf_path, path)
        return path
    def get_filename(self, key: str) -> str | None:
        """get a key as a filename

        return None if the key is not set
        """
        path = self.conf.get(key)
        if path is not None:
            path = self._rel_to_abs(path)
        return path
    def __getitem__(self, key):
        return self.conf.__getitem__(key)
    def get(self, key, *args):
        return self.conf.get(key, *args)
    def out_filename(self, ext: str) -> str:
        return os.path.join(self.output_dir, self.name+ext)