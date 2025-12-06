from PIL import Image
from src.proposals import get_selective_search_proposals, filter_proposals

if __name__ == "__main__":
    img = Image.open("data/sample.jpg").convert("RGB")
    boxes = get_selective_search_proposals(img, mode="fast")
    print("Total raw proposals:", len(boxes))
    boxes2 = filter_proposals(boxes, min_area=1000, max_proposals=300)
    print("Filtered proposals:", len(boxes2))
    print("First 5 boxes:", boxes2[:5])
