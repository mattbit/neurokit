from typing import Sequence


class NamedItemsBag:
    def __init__(self, items: Sequence = None, dtype=None):
        self._items = {}
        self.dtype = dtype
        items = items or []
        for item in items:
            self.add(item)

    def __getattr__(self, name):
        return self._items[name]

    def __repr__(self):
        names = self._items.keys()
        if len(names) > 4:
            names = list(names[:4]) + ['…']
        return f"<NamedItemsBag {{{', '.join(names)}}}>"

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for item in self._items.values():
            yield item

    def add(self, item):
        if item.name in self._items:
            raise ValueError(f'Cannot add duplicate item `{item.name}`.')
        if self.dtype and not isinstance(item, self.dtype):
            raise TypeError(
                f'Item `{item.name}` is not a valid `{self.dtype}` instance.')
        self._items[item.name] = item
