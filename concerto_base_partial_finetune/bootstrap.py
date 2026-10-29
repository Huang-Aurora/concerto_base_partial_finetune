"""Runtime bootstrap for Kaggle: make the official Concerto code and the
pretrained checkpoint available before any `import concerto` happens.

Called at the top of train.py:
  import config as C
  import bootstrap
  bootstrap.ensure(C.CONCERTO_CKPT)

Steps:
  1. locate (or download) the official Concerto source tree -> sys.path
  2. download the pretrained checkpoint if missing (HF, mirror fallback)
  3. verify hard deps (torch, spconv, torch_scatter, timm, sklearn) with an
     actionable error message instead of a bare ImportError
"""
import os
import shutil
import sys
import zipfile

CONCERTO_ZIP_URLS = [
    "https://github.com/Pointcept/Concerto/archive/refs/heads/main.zip",
    "https://ghproxy.com/https://github.com/Pointcept/Concerto/archive/refs/heads/main.zip",
]
CONCERTO_SRC = os.environ.get("CONCERTO_SRC", "/kaggle/working/Concerto-main")


def _have_concerto_src(dest):
    return os.path.isfile(os.path.join(dest, "concerto", "__init__.py"))


def _download(url, out_file):
    import urllib.request
    import ssl
    print(f"[bootstrap] downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, out_file)
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ctx, timeout=600) as r, \
                open(out_file, "wb") as f:
            shutil.copyfileobj(r, f)


def ensure_concerto_src():
    try:
        import concerto  # noqa: F401
        return  # already importable (e.g. PYTHONPATH set by the user)
    except ImportError:
        pass
    if not _have_concerto_src(CONCERTO_SRC):
        parent = os.path.dirname(CONCERTO_SRC) or "."
        os.makedirs(parent, exist_ok=True)
        zip_path = os.path.join(parent, "concerto-main.zip")
        ok = False
        for url in CONCERTO_ZIP_URLS:
            try:
                _download(url, zip_path)
                ok = True
                break
            except Exception as e:
                print(f"[bootstrap] download failed: {e}", flush=True)
        if not ok:
            raise RuntimeError(
                "cannot download Concerto source; place Concerto-main next to "
                "this folder or set CONCERTO_SRC / PYTHONPATH manually")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(parent)
        os.remove(zip_path)
    sys.path.insert(0, CONCERTO_SRC)
    import concerto  # noqa: F401
    print(f"[bootstrap] concerto source: {CONCERTO_SRC}", flush=True)


def ensure_concerto_ckpt(ckpt_path):
    if os.path.isfile(ckpt_path):
        return
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    name = os.path.basename(ckpt_path)
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo_id="Pointcept/Concerto", filename=name,
                        repo_type="model", local_dir=os.path.dirname(ckpt_path))
    except Exception:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from importlib import reload
        reload(sys.modules["huggingface_hub"])
        hf_hub_download(repo_id="Pointcept/Concerto", filename=name,
                        repo_type="model", local_dir=os.path.dirname(ckpt_path))
    assert os.path.isfile(ckpt_path), f"ckpt download failed: {ckpt_path}"
    print(f"[bootstrap] checkpoint: {ckpt_path} "
          f"({os.path.getsize(ckpt_path) / 2**20:.0f} MiB)", flush=True)


def check_deps():
    missing = []
    for mod, hint in [
        ("torch", "preinstalled on Kaggle GPU notebooks"),
        ("spconv", "pip install spconv-cu121 (see requirements.txt)"),
        ("torch_scatter", "pip install torch-scatter (find-links in "
                          "requirements.txt must match torch/cuda)"),
        ("timm", "pip install timm"),
        ("sklearn", "pip install scikit-learn"),
        ("scipy", "pip install scipy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"  - {mod}: {hint}")
    if missing:
        raise ImportError(
            "missing dependencies:\n" + "\n".join(missing) +
            "\nRun: pip install -r requirements.txt")


def ensure(ckpt_path):
    check_deps()
    ensure_concerto_src()
    ensure_concerto_ckpt(ckpt_path)
    os.makedirs(os.environ.get("OUTPUT_ROOT", "/kaggle/working/outputs"),
                exist_ok=True)
