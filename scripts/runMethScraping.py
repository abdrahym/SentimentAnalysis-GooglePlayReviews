import os
import csv
import pandas as pd
from google_play_scraper import app, reviews_all, Sort


# Build method getScrapviewSaved
def getScrapviewSaved(
    Idname,
    IdScrap,
    lang="id",
    country="id",
    sort=Sort.MOST_RELEVANT,
    count=10000,
    Path=None,
):
    try:
        if Path:
            os.makedirs(Path, exist_ok=True)
            file_path = os.path.join(
                Path, f"{Idname}.csv"
            )
        else:
            file_path = (
                f"{Idname}.csv"
            )

        scrapreview = reviews_all(
            IdScrap, lang=lang, country=country, sort=sort, count=count
        )

        if not scrapreview:
            print("⚠️ Comment not found.")
            return

        df = pd.DataFrame(scrapreview)

        df.to_csv(file_path, index=False, encoding="utf-8")

        print(f"✅ Data save in : {file_path}")

    except Exception as e:
        print(f"❌ error: {e}")


getScrapviewSaved(IdName, IdScrap, count=count, Path=Path)
