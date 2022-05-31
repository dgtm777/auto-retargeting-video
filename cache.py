from collections import OrderedDict, namedtuple


class Cache:
    def __init__(self, size=100000):
        self.max_size = size
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if self.cache.__len__() > self.max_size:
            self.cache.popitem(last=False)

    def get_from_cache(self, cur_id, func, *args, **kwargs):
        if cur_id in self.cache:
            return self.cache[cur_id]
        else:
            return func(cur_id, *args, **kwargs)
