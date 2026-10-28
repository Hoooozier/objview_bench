"""Small, closed metadata contract. Repository references are never fetched here."""
import re


class IntakeError(Exception):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.code = code
        self.status = status


def text_field(value, maximum=80):
    if (not isinstance(value, str) or not 1 <= len(value) <= maximum
            or value != value.strip() or any(not c.isprintable() for c in value)):
        raise IntakeError('INVALID_PARAMS')
    return value


def request_key(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise IntakeError('INVALID_PARAMS')
    return value


def manifest_fields(data):
    fields = {'method_id', 'repository', 'commit_sha', 'dockerfile', 'entrypoint', 'protocol_version'}
    if not isinstance(data, dict) or set(data) != fields:
        raise IntakeError('INVALID_PARAMS')
    if any(not isinstance(value, str) for value in data.values()):
        raise IntakeError('INVALID_PARAMS')
    if not re.fullmatch(r'[0-9a-f]{32}', data['method_id']):
        raise IntakeError('INVALID_PARAMS')
    if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_-]{1,100}/[A-Za-z0-9_.-]{1,100}', data['repository']):
        raise IntakeError('INVALID_PARAMS')
    if data['repository'].rsplit('/', 1)[1] in {'.', '..'}:
        raise IntakeError('INVALID_PARAMS')
    if not re.fullmatch(r'[0-9a-f]{40}', data['commit_sha']):
        raise IntakeError('INVALID_PARAMS')
    path = data['dockerfile']
    if (not 1 <= len(path) <= 240 or not re.fullmatch(r'[A-Za-z0-9_./-]+', path)
            or any(part in {'', '.', '..'} or part.startswith('-') for part in path.split('/'))):
        raise IntakeError('INVALID_PARAMS')
    if data['entrypoint'] != '/opt/objview/run.sh' or data['protocol_version'] != 'objview-interaction/v1':
        raise IntakeError('INVALID_PARAMS')
    return dict(data)
