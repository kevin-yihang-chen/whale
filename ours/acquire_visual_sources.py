"""Bounded public source acquisition with immutable bytes and provenance.

This prepares verifier-side inputs for E1 data construction. It never creates
benchmark scores or passes source annotations to a model/proposer.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

SOURCES = {
    'clevr_metadata': {
        'url': 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip',
        'authority': 'https://cs.stanford.edu/people/jcjohns/clevr/',
        'license': 'CC-BY-4.0', 'filename': 'CLEVR_v1.0_no_images.zip', 'format': 'zip', 'max_bytes': 256*1024**2},
    'plotqa_train_annotations': {
        'url': 'https://drive.usercontent.google.com/download?id=1VzWwxBVrlep17BGZU17GpLuGpwjyWbzq&export=download&confirm=t',
        'authority': 'https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md',
        'official_file_page': 'https://drive.google.com/file/d/1VzWwxBVrlep17BGZU17GpLuGpwjyWbzq/view',
        'license': 'CC-BY-4.0', 'filename': 'plotqa_train_annotations.json', 'format': 'json',
        'max_bytes': 1280*1024**2,
        'bound_basis': 'Official download HEAD on 2026-09-10: Content-Length 1157345635 bytes.'}}


def acquire(name, output):
    spec = SOURCES[name]
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve existing source acquisition; use its receipt or a new attempt path')
    if shutil.disk_usage(output.parent).free < spec['max_bytes'] + 40*1024**3:
        raise ValueError('Acquisition would enter the reserved working-space margin')
    output.mkdir()
    record = {'source': name, 'specification': spec, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'public_source_annotations_not_model_input', 'model_calls': 0, 'bytes': 0}
    (output/'start.json').write_text(json.dumps(record, indent=2)+'\n')
    partial = output/(spec['filename']+'.partial')
    try:
        with urllib.request.urlopen(spec['url'], timeout=60) as response, partial.open('xb') as stream:
            length = response.headers.get('Content-Length')
            record['response_content_length'] = int(length) if length is not None else None
            record['response_content_type'] = response.headers.get('Content-Type')
            if length is not None and int(length) > spec['max_bytes']:
                raise ValueError('Source exceeds the declared download bound')
            digest = hashlib.sha256()
            first = True
            while True:
                chunk = response.read(1024**2)
                if not chunk:
                    break
                if first:
                    valid = chunk.startswith(b'PK\x03\x04') if spec['format'] == 'zip' else chunk.lstrip().startswith((b'[', b'{'))
                    if not valid:
                        raise ValueError('Source returned a confirmation/error page or unexpected format')
                    first = False
                record['bytes'] += len(chunk)
                if record['bytes'] > spec['max_bytes']:
                    raise ValueError('Streaming source exceeded its download bound')
                stream.write(chunk)
                digest.update(chunk)
            if first or (length is not None and record['bytes'] != int(length)):
                raise ValueError('Source download is empty or incomplete')
            record.update(content_type=response.headers.get('Content-Type'), sha256=digest.hexdigest())
        final = output/spec['filename']
        partial.rename(final)
        record.update(status='DOWNLOADED_SOURCE_BYTES', path=str(final.resolve()),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            limitations=['Original bytes downloaded; semantic/oracle and split validation are separate.',
                'No assertion that these source records are already valid visual intervention pairs.'])
        (output/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({k: record[k] for k in ('source','status','bytes','sha256')}), flush=True)
    except BaseException as error:
        record.update(status='SOURCE_ACQUISITION_FAILED', error_type=type(error).__name__, error=str(error),
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        (output/'failure.json').write_text(json.dumps(record, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=tuple(SOURCES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    acquire(args.source, args.output)
