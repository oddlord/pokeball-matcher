import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

PKMN_IMGS_DIR_NAME = "pokemons-to-match"
POKEBALL_IMGS_DIR_NAME = "pokeballs"

POKEBALL_IMG_NAMES = {
    "Poké Ball": "pokeball.png",
    "Great Ball": "greatball.png",
    "Ultra Ball": "ultraball.png",
    "Master Ball": "masterball.png",
    "Heavy Ball": "heavyball.png",
    "Lure Ball": "lureball.png",
    "Friend Ball": "friendball.png",
    "Love Ball": "loveball.png",
    "Level Ball": "levelball.png",
    "Fast Ball": "fastball.png",
    "Moon Ball": "moonball.png",
    "Premier Ball": "premierball.png",
    "Luxury Ball": "luxuryball.png",
    "Net Ball": "netball.png",
    "Dive Ball": "diveball.png",
    "Nest Ball": "nestball.png",
    "Repeat Ball": "repeatball.png",
    "Timer Ball": "timerball.png",
    "Dusk Ball": "duskball.png",
    "Heal Ball": "healball.png",
    "Quick Ball": "quickball.png",
    "Safari Ball": "safariball.png",
    "Sport Ball": "sportball.png",
    "Dream Ball": "dreamball.png",
    "Beast Ball": "beastball.png",
}

def extract_dominant_colors(img_array, k=5):
    # Flatten image to (pixels, 3)
    pixels = img_array.reshape(-1, 3)

    # Run KMeans to find k dominant colors
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    
    # Return cluster centers (dominant colors)
    return kmeans.cluster_centers_

def compare_color_palettes(img_path_1, img_path_2, k=5):
    # Load images
    img1 = cv2.imread(img_path_1)
    img2 = cv2.imread(img_path_2)

    palette1 = extract_dominant_colors(img1, k)
    palette2 = extract_dominant_colors(img2, k)

    # Compute pairwise distances between two color palettes
    distances = cdist(palette1, palette2, metric='euclidean')
    # Sum of minimum distances between each color
    distance = np.mean(np.min(distances, axis=1))
    similarity_score = 1 - min(distance / 100, 1)
    return similarity_score

def get_alpha_mask(img):
    if img.shape[2] == 4:
        # BGRA image – grab alpha channel
        alpha = img[:, :, 3]
        mask = (alpha > 0).astype(np.uint8)
        return mask
    else:
        # No alpha channel – use a full mask
        return np.ones(img.shape[:2], dtype=np.uint8)

def compare_colour_histograms(img_path_1, img_path_2):
    # Load images
    img1 = cv2.imread(img_path_1)
    img2 = cv2.imread(img_path_2)

    # Convert to HSV color space
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Get alpha channel masks
    mask1 = get_alpha_mask(img1)
    mask2 = get_alpha_mask(img2)

    # Calculate histograms (H and S channels)
    hist1 = cv2.calcHist([hsv1], [0, 1], mask1, [100, 100], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], mask2, [100, 100], [0, 180, 0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Compare histograms using correlation
    # similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    # similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return similarity


script_dir_path = os.path.dirname(os.path.abspath(__file__))
pkmn_imgs_dir_path = os.path.join(script_dir_path, PKMN_IMGS_DIR_NAME)
pokeball_imgs_dir_path = os.path.join(script_dir_path, POKEBALL_IMGS_DIR_NAME)

pkmn_imgs = os.listdir(pkmn_imgs_dir_path)
for pkmn_img in pkmn_imgs:
    if not pkmn_img.endswith(".png") and not pkmn_img.endswith(".jpg"):
        continue
    print(f"\n{pkmn_img}")
    pkmn_img_path = os.path.join(pkmn_imgs_dir_path, pkmn_img)
    pokeball_scores = []
    for pokeball in POKEBALL_IMG_NAMES:
        pokeball_img_path = os.path.join(pokeball_imgs_dir_path, POKEBALL_IMG_NAMES[pokeball])
        # score = compare_colour_histograms(pkmn_img_path, pokeball_img_path)
        score = compare_color_palettes(pkmn_img_path, pokeball_img_path)
        pokeball_scores.append({
            "pokeball": pokeball,
            "score": score
        })
    pokeball_scores.sort(key=lambda ps : ps["score"], reverse=True)
    for pokeball_score in pokeball_scores:
        print(f"\t{pokeball_score['pokeball']}: {pokeball_score['score']}")
