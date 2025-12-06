from PIL import Image
from src.proposals import get_selective_search_proposals, filter_proposals
from src.warp_and_crop import crop_and_resize_patches

if __name__ == "__main__":
    img = Image.open("data/sample.jpg").convert("RGB")
    boxes = get_selective_search_proposals(img)
    boxes = filter_proposals(boxes, max_proposals=100)
    patches, mapped = crop_and_resize_patches(img, boxes)
    print("Patches shape:", None if patches is None else patches.shape)
    print("Mapped boxes returned:", len(mapped))
