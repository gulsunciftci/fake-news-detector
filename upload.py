from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id="gulsunnciftci/fake-news-data",
    repo_type="dataset",
    private=False,
    exist_ok=True
)
print("Repo hazır!")

dosyalar = [
    "../data/admin_data.csv",
    "../data/cleaned_news.csv",
    "../data/Fake.csv",
    "../data/final_cleaned_news.csv",
    "../data/True.csv",
    "../data/updated_news_dataset.csv"
]

for dosya in dosyalar:
    print(f"Yükleniyor: {dosya}")
    api.upload_file(
        path_or_fileobj=dosya,
        path_in_repo=dosya.split("/")[-1],
        repo_id="gulsunnciftci/fake-news-data",
        repo_type="dataset"
    )
    print(f"✓ Tamamlandı: {dosya}")

print("\nTüm dosyalar yüklendi!")