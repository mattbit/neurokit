import pytest
from collections import namedtuple

from neurokit.core.common import NamedItemsBag


Item = namedtuple('Item', ['name', 'value'])
OtherItem = namedtuple('OtherItem', ['name', 'other_value'])


def test_main_properties():
    items = [Item('name1', 'val1'), Item('name2', 'val2')]
    bag = NamedItemsBag(items)

    assert len(bag) == 2
    assert bag.has('name1')
    assert bag.has('name2')
    assert bag['name1'].value == 'val1'
    assert bag.name2.value == 'val2'


def test_add_items():
    items = [Item('name1', 'val1'), Item('name2', 'val2')]
    bag = NamedItemsBag()
    for item in items:
        bag.add(item)

    assert len(bag) == len(items)
    assert bag.has('name1')
    assert bag.has('name2')
    assert bag['name1'].value == 'val1'
    assert bag.name2.value == 'val2'


def test_remove_items():
    items = [Item('name1', 'val1'), Item('name2', 'val2')]
    bag = NamedItemsBag(items)

    assert len(bag) == 2

    bag.remove('name1')

    assert len(bag) == 1
    assert not bag.has('name1')
    assert bag.has('name2')

    del bag['name2']
    assert len(bag) == 0
    assert not bag.has('name2')


def test_cannot_add_if_same_name():
    bag = NamedItemsBag()

    bag.add(Item('test', 'value1'))

    with pytest.raises(ValueError):
        bag.add(Item('test', 'value2'))


def test_cannot_add_if_no_name():
    bag = NamedItemsBag()
    item = Item(None, 'val')

    with pytest.raises(ValueError):
        bag.add(item)

    item = ('test', 'val')

    with pytest.raises(ValueError):
        bag.add(item)


def test_only_accepts_given_type():
    bag = NamedItemsBag(dtype=Item)

    with pytest.raises(TypeError):
        bag.add(OtherItem('other', 'other'))

    bag.add(Item('test', 'test'))
    assert bag.has('test')
