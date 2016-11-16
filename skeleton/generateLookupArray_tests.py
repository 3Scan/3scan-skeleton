from skeleton.generateLookupArray import generateLookupArray


def test_generateLookupArray():
    assert not generateLookupArray(2).sum(), "first two config numbers should be delete"
