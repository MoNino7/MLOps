def test_simple_math():
    """A simple test that always passes"""
    assert 1 + 1 == 2
    assert 2 + 2 == 4


def test_string_operations():
    """Test basic string operations"""
    assert "hello".upper() == "HELLO"
    assert len("test") == 4
