"""Template identifiers must not remove real ownership or history assertions."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, config


def classify(text, enabled=True, option=None, **overrides):
    option = option or 'A. In ' + text + ', consider asynchronous calls.'
    start = option.index(text)
    claim = dict(text=text, option_span=dict(start=start, end=start + len(text)),
                 validation_errors=[], reason='no_source')
    claim.update(overrides)
    with patch.object(config, 'CHOICE_TEMPLATE_PATH_SCOPE', enabled):
        return answer_choice._nonasserted_scope_reason(claim, option)


def main():
    for text in ('your_project_path/main.py', '<PROJECT_DIR>/src/app.py',
                 '[PROJECT_DIR]/main.py', 'path/to/your/project/index.js',
                 'your main project file (your_project_path/main.py)',
                 'the code file ([FILE_PATH])', 'your script (<CODE_PATH>)',
                 '[FILE_PATH]', '[REDACTED_PATH]', '[REDACTED_CODE_PATH]',
                 '<your_file_path>', '[PATH_TO_SCRIPT]', '[FILE_PATH_PLACEHOLDER]'):
        assert classify(text) == 'template_path_not_actual_identifier'
        assert classify(text, False) is None
        assert classify(text, reason='contradicted') is None
        assert classify(text, validation_errors=['invalid_span']) is None
        assert classify(text, option_span={'start': 0, 'end': len(text)}) is None
        assert classify(text, option='Your literal filename is ' + text + '.') is None
    for text in ('/Users/alex/project/main.py', '/home/user/app.py', 'src/main.py',
                 'my_project/main.py', 'your_project_path_backup/main.py',
                 'the project you created yesterday at your_project_path/main.py',
                 'your_project_path/main.py belongs to you',
                 '<PROJECT_DIR>/../private/file.py',
                 'your project file (/Users/alex/main.py)',
                 'your old project file ([FILE_PATH])',
                 'your project file ([FILE_PATH]) that you created yesterday',
                 'your main project file (your_project_path/main.py) contains medical records',
                 '[FILE_PATH] or your diary', '[contact email on file]'):
        assert classify(text) is None, text
    print('PASS: symbolic roots, default-off behavior, real paths, assertion and span guards')


if __name__ == '__main__':
    main()
