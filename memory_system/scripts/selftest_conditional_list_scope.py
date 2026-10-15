"""Comma-separated conditional interests must not erase asserted history."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, config


def reason(option, text, enabled=True):
    start = option.index(text)
    claim = dict(text=text, option_span=dict(start=start, end=start+len(text)),
                 validation_errors=[], reason='no_source')
    with patch.object(config, 'CHOICE_CONDITIONAL_LIST_SCOPE', enabled):
        return answer_choice._nonasserted_scope_reason(claim, option)


def main():
    text = 'you like versatile, easy-to-wear pieces'
    option = 'D. If '+text+', consider pairing trousers with walking shoes.'
    assert reason(option, text, False) is None
    assert reason(option, text) == 'conditional_interest_not_asserted'
    for text in ('you enjoy hiking, swimming, and painting', 'you prefer bright, light clothes'):
        assert reason('If '+text+', try a local workshop.', text) == 'conditional_interest_not_asserted'
    for text in ('you like your old, comfortable shoes', 'you like hiking, as you always did',
                 'you enjoy the jackets, which you already own'):
        assert reason('If '+text+', consider a short walk.', text) is None
    option = 'If you like bright, light clothes, consider a walk. You visited Paris last year.'
    assert reason(option, 'You visited Paris last year') is None
    option = 'If you like bright colors, which you chose yesterday, consider a scarf.'
    assert reason(option, 'you chose yesterday') is None
    option = 'If you like bright colors, as I mentioned before, consider a scarf.'
    assert reason(option, 'I mentioned before') is None
    assert reason('Since '+text+', consider a short walk.', text) is None
    print('PASS: conditional lists, flag-off behavior, embedded assertions, and outside-scope claims')


if __name__ == '__main__':
    main()
