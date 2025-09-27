import logging
import os
import re
from typing import List, Optional, Union

import requests
from tqdm.auto import tqdm

_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def available_models() -> List[str]:
    """Returns the names of available models."""
    return list(_MODELS.keys())


def get_assets_path():
    """Returns the path to the assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_logger():
    """Returns the module logger."""
    return logging.getLogger("faster_whisper")


def _default_cache_dir():
    """Default cache directory for model downloads."""
    return os.path.expanduser("~/.cache/faster_whisper_models")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _is_model_present(dest_dir: str) -> bool:
    """
    Check whether the required files exist locally:
      - config.json (required)
      - tokenizer.json (required)
      - model.bin (required)
      - vocabulary.json OR vocabulary.txt (one required)
    Optional:
      - preprocessor_config.json (optional)
    """
    if not os.path.isdir(dest_dir):
        return False

    required = {"config.json", "tokenizer.json", "model.bin"}
    has = set()
    has_vocab = False

    for root, _, files in os.walk(dest_dir):
        for f in files:
            if f in required:
                has.add(f)
            if f in ("vocabulary.json", "vocabulary.txt"):
                has_vocab = True

    return required.issubset(has) and has_vocab


def _download_file(url: str, dest_path: str, headers: dict = None, tqdm_class=None) -> bool:
    """
    Download a single file. Returns True on success, False if 404.
    Raises requests.exceptions.RequestException for other errors.
    """
    headers = headers or {}
    resp = requests.get(url, stream=True, headers=headers, timeout=30)
    if resp.status_code == 200:
        total = int(resp.headers.get("content-length", 0))
        tmp_path = dest_path + ".part"
        if tqdm_class is not None and total > 0:
            bar = tqdm_class(total=total, unit="B", unit_scale=True, leave=False)
            use_bar = True
        else:
            # dummy object with update and close methods
            class _NoBar:
                def update(self, n):  # noqa: D401
                    pass

                def close(self):  # noqa: D401
                    pass

            bar = _NoBar()
            use_bar = False

        with open(tmp_path, "wb") as f:
            try:
                for chunk in resp.iter_content(chunk_size=32 * 1024):
                    if chunk:
                        f.write(chunk)
                        if use_bar:
                            bar.update(len(chunk))
            finally:
                if use_bar:
                    bar.close()

        os.replace(tmp_path, dest_path)
        return True
    elif resp.status_code == 404:
        return False
    else:
        resp.raise_for_status()


def snapshot_download_simple(
    repo_id: str,
    local_files_only: bool = False,
    tqdm_class: Optional[type] = None,
    revision: Optional[str] = None,
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
) -> str:
    """
    Minimal snapshot downloader that only fetches:
      - config.json (required)
      - preprocessor_config.json (optional)
      - tokenizer.json (required)
      - vocabulary.json or vocabulary.txt (one required)
      - model.bin (required)
    """
    logger = get_logger()
    revision = revision or "main"

    if local_dir is not None:
        dest_root = os.path.abspath(local_dir)
    else:
        base_cache = cache_dir or _default_cache_dir()
        safe_repo = repo_id.replace("/", "_")
        dest_root = os.path.abspath(os.path.join(base_cache, safe_repo))

    # If only local files allowed, just verify presence and return or raise
    if local_files_only:
        if _is_model_present(dest_root):
            return dest_root
        raise FileNotFoundError(f"Local files only requested but model files not found in {dest_root}")

    _ensure_dir(dest_root)

    # Files we will attempt to fetch
    required_files = ["config.json", "tokenizer.json", "model.bin"]
    optional_preproc_names = ["preprocessor_config.json"]
    vocab_options = ["vocabulary.json", "vocabulary.txt"]

    # Build headers for auth if token provided as string
    headers = {}
    if token and isinstance(token, str):
        headers["Authorization"] = f"Bearer {token}"

    base_url = f"https://huggingface.co/{repo_id}/resolve/{revision}"

    # Attempt to download required files first
    downloaded = set()
    for fname in required_files:
        dest_path = os.path.join(dest_root, fname)
        if os.path.exists(dest_path):
            downloaded.add(fname)
            continue
        file_url = f"{base_url}/{fname}"
        try:
            ok = _download_file(file_url, dest_path, headers=headers, tqdm_class=tqdm_class or tqdm)
            if ok:
                downloaded.add(fname)
                logger.debug("Downloaded %s", fname)
        except requests.exceptions.RequestException as e:
            logger.warning("Failed to download %s: %s", file_url, e)

    # Try vocabulary options (one required)
    vocab_downloaded = False
    for vocab in vocab_options:
        dest_path = os.path.join(dest_root, vocab)
        if os.path.exists(dest_path):
            vocab_downloaded = True
            break
        file_url = f"{base_url}/{vocab}"
        try:
            ok = _download_file(file_url, dest_path, headers=headers, tqdm_class=tqdm_class or tqdm)
            if ok:
                vocab_downloaded = True
                logger.debug("Downloaded %s", vocab)
                break
        except requests.exceptions.RequestException as e:
            logger.warning("Failed to download %s: %s", file_url, e)

    # Try optional preprocessor file
    for pre in optional_preproc_names:
        dest_path = os.path.join(dest_root, pre)
        if os.path.exists(dest_path):
            continue
        file_url = f"{base_url}/{pre}"
        try:
            _download_file(file_url, dest_path, headers=headers, tqdm_class=tqdm_class or tqdm)
            logger.debug("Downloaded optional preprocessor %s", pre)
            break  # only need one of them
        except requests.exceptions.RequestException:
            # ignore failures for optional file
            if os.path.exists(dest_path):
                continue
            try:
                # clean up any .part left behind
                part = dest_path + ".part"
                if os.path.exists(part):
                    os.remove(part)
            except Exception:
                pass
            continue

    # Final verification: required files & one vocab must be present
    missing = []
    for fname in required_files:
        if not os.path.exists(os.path.join(dest_root, fname)):
            missing.append(fname)
    if not any(os.path.exists(os.path.join(dest_root, v)) for v in vocab_options):
        missing.append("vocabulary.json|vocabulary.txt")

    if missing:
        raise FileNotFoundError(
            f"Could not download required model files for {repo_id}. Missing: {', '.join(missing)}"
        )

    return dest_root


def download_model(
    size_or_id: str,
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[str, bool]] = None,
):
    """Downloads a CTranslate2 Whisper model from the Hugging Face Hub.

    Args:
      size_or_id: Size of the model to download from https://huggingface.co/Systran
        (tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, medium.en,
        distil-medium.en, large-v1, large-v2, large-v3, large, distil-large-v2,
        distil-large-v3), or a CTranslate2-converted model ID from the Hugging Face Hub
        (e.g. Systran/faster-whisper-large-v3).
      output_dir: Directory where the model should be saved. If not set, the model is saved in
        the cache directory.
      local_files_only:  If True, avoid downloading the file and return the path to the local
        cached file if it exists.
      cache_dir: Path to the folder where cached files are stored.
      revision: An optional Git revision id which can be a branch name, a tag, or a
            commit hash.
      use_auth_token: HuggingFace authentication token or True to use the
            token stored by the HuggingFace config folder.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """
    if re.match(r".*/.*", size_or_id):
        repo_id = size_or_id
    else:
        repo_id = _MODELS.get(size_or_id)
        if repo_id is None:
            raise ValueError(
                "Invalid model size '%s', expected one of: %s"
                % (size_or_id, ", ".join(_MODELS.keys()))
            )

    kwargs = {
        "local_files_only": local_files_only,
        "tqdm_class": disabled_tqdm,
        "revision": revision,
    }

    if output_dir is not None:
        kwargs["local_dir"] = output_dir

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    if use_auth_token is not None:
        kwargs["token"] = use_auth_token

    try:
        return snapshot_download_simple(repo_id, **kwargs)
    except (requests.exceptions.ConnectionError, FileNotFoundError) as exception:
        logger = get_logger()
        logger.warning(
            "An error occurred while synchronizing the model %s from the Hugging Face Hub:\n%s",
            repo_id,
            exception,
        )
        logger.warning("Trying to load the model directly from the local cache, if it exists.")

        kwargs["local_files_only"] = True
        return snapshot_download_simple(repo_id, **kwargs)


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class disabled_tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )
