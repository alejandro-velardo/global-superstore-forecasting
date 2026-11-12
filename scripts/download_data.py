"""Download the Superstore dataset from Kaggle into data/raw/.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token).
Use the dataset slug from the link in the assignment PDF (topic #18), e.g.:

    python scripts/download_data.py --dataset <owner>/<dataset-slug>
"""

import argparse

from superstore.config import RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Kaggle slug '<owner>/<dataset>'")
    args = parser.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi  # imported lazily: needs credentials

    api = KaggleApi()
    api.authenticate()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(args.dataset, path=str(RAW_DIR), unzip=True)
    print("Downloaded to", RAW_DIR, "->", sorted(p.name for p in RAW_DIR.iterdir()))


if __name__ == "__main__":
    main()
