from fastcore.all import *
from duckduckgo_search import DDGS
from fastai.vision.all import *
from fastdownload import download_url


def search_images(term, max_images=50):
    print(f"Searching for '{term}'")
    with DDGS(headers = {"Accept-Encoding": "gzip, deflate, br"}) as ddgs:
        results = ddgs.images(keywords=term, max_results=max_images)
        return L(results).itemgot('image')
    
queries = ("good road","bad road")
path = Path("road_types")

for i in queries:
    dest = path/i
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(i))
    resize_images(dest, max_size=400, dest=dest)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"{len(failed)} images failed to download properly")

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)


test_image = search_images("random road photo", max_images=1)
test_image = download_url(test_image[0])

results,_,probs = learn.predict(PILImage.create(Image.open(test_image).to_thumb(256,256)))
print(results, f"with a probability of {probs[0]:.4f}")
Image.open(test_image).to_thumb(256,256)