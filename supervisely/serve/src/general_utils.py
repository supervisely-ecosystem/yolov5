import requests
import supervisely_lib as sly


def download_weights_by_url(url, local_path, logger):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        progress = sly.Progress("Downloading weights", total_size_in_bytes, ext_logger=logger, is_size=True)
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.iters_done_report(len(chunk))
    return local_path