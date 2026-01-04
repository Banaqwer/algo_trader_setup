"""Test execution engine format specifiers."""

import pytest


def test_direction_format_specifier():
    """Test that direction format specifier works correctly with integers."""
    # Test the f-string format that was causing the issue
    trade_id = "test123"
    instrument = "EUR_USD"
    direction = 1  # Buy direction (positive integer)
    entry_price_filled = 1.12345
    
    # This should not raise a ValueError
    log_message = f"Trade {trade_id} opened: {instrument} {direction:+d} @ {entry_price_filled:.5f}"
    
    # Verify the format is correct
    assert log_message == "Trade test123 opened: EUR_USD +1 @ 1.12345"
    
    # Test with negative direction (sell)
    direction = -1
    log_message = f"Trade {trade_id} opened: {instrument} {direction:+d} @ {entry_price_filled:.5f}"
    assert log_message == "Trade test123 opened: EUR_USD -1 @ 1.12345"


def test_format_specifier_with_space_should_fail():
    """Test that format specifier with space before colon would fail."""
    trade_id = "test123"
    instrument = "EUR_USD"
    direction = 1
    entry_price_filled = 1.12345
    
    # This should raise a ValueError because of the space before the colon
    with pytest.raises(ValueError):
        # The space before : is invalid syntax and will cause an error
        log_message = f"Trade {trade_id} opened: {instrument} {direction: +d} @ {entry_price_filled:.5f}"
