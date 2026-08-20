import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm

def images_to_video(image_dir, output_video_path, fps=30):
    """
    Convert a folder of images into a video.
    """

    # Get sorted image list
    images = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith(".png") or f.endswith(".jpg")
    ])

    if len(images) == 0:
        raise ValueError("No images found in directory.")

    # Read first image to get size
    first_img_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_img_path)

    if frame is None:
        raise ValueError(f"Failed to read image: {first_img_path}")

    height, width, _ = frame.shape

    # Video writer (mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # ensure size consistency
        frame = cv2.resize(frame, (width, height))

        video.write(frame)

    video.release()
    print(f"Video saved to: {output_video_path}")

def extract_mask_contour(
    mask,
    output = None,
    thickness=2,
):
    """
    Extract contours from a binary mask.

    Args:
        mask (str): Input mask path.
        output (str): Output contour image path.
        thickness (int): Contour line thickness.
    """
    if isinstance(mask, str):
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to load image: {mask}")

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    h, w = binary.shape

    contour_img = np.zeros_like(binary)

    cv2.drawContours(
        contour_img,
        contours,
        -1,
        255,
        thickness
    )

    if output is not None:
        cv2.imwrite(output, contour_img)

    return contour_img

def overlay_contour(
    img,
    contour,
    output_path=None,
    alpha=1.0,
    color=(0, 255, 0),
):
    """
    Overlay contour (white pixels) onto image with alpha blending.

    result = (1 - alpha) * image + alpha * color (only on contour pixels)
    """

    if isinstance(img, str):
        img = np.array(Image.open(img).convert("RGB")).astype(np.float32)
    elif isinstance(img, np.ndarray):
        img = img.astype(np.float32)
    else:
        raise ValueError(f"img must be a path or numpy array, got {type(img)}")

    if isinstance(contour, str):
        contour = np.array(Image.open(contour).convert("L")).astype(np.uint8)
    elif isinstance(contour, np.ndarray):
        contour = contour.astype(np.uint8)
    else:
        raise ValueError(f"contour must be a path or numpy array, got {type(contour)}")

    if contour.shape != img.shape[:2]:
        raise ValueError(
            f"Contour size {contour.shape} does not match image size {img.shape[:2]}"
        )

    # Binary mask for contour pixels
    mask = contour > 0

    # Prepare color layer
    color_layer = np.zeros_like(img, dtype=np.float32)
    color_layer[:] = color

    # Blend only on contour pixels
    img[mask] = (1 - alpha) * img[mask] + alpha * color_layer[mask]

    result = np.clip(img, 0, 255).astype(np.uint8)

    if output_path is not None:
        Image.fromarray(result).save(output_path)

    return result


def overlay_image(
    image1,
    image2,
    output_path=None,
    alpha=0.5,
    mask=None,
):
    """
    result = (1 - alpha) * image1 + alpha * image2

    If mask is provided, blending is applied only where mask > 0.
    """

    # Load images

    if isinstance(image1, str):
        img1 = np.array(Image.open(image1).convert("RGB")).astype(np.float32)
    elif isinstance(image1, np.ndarray):
        img1 = image1.astype(np.float32)
    else:
        raise ValueError(f"image1 must be a path or numpy array, got {type(image1)}")

    if isinstance(image2, str):
        img2 = np.array(Image.open(image2).convert("RGB")).astype(np.float32)
    elif isinstance(image2, np.ndarray):
        img2 = image2.astype(np.float32)
    else:
        raise ValueError(f"image2 must be a path or numpy array, got {type(image2)}")

    if img1.shape != img2.shape:
        raise ValueError(
            f"Image sizes differ: {img1.shape} vs {img2.shape}"
        )

    # Blend images
    blended = (1 - alpha) * img1 + alpha * img2

    if mask is not None:
        if isinstance(mask, str):
            mask = np.array(Image.open(mask).convert("L"))
        elif isinstance(mask, np.ndarray):
            mask = mask.astype(np.float32)
        else:
            raise ValueError(f"mask must be a path or numpy array, got {type(mask)}")

        if mask.shape != img1.shape[:2]:
            raise ValueError(
                f"Mask size {mask.shape} does not match image size {img1.shape[:2]}"
            )

        # Binary mask
        mask = (mask > 0).astype(np.float32)

        # Expand to 3 channels
        mask = mask[..., None]

        # Only modify masked area
        result = img1 * (1 - mask) + blended * mask
    else:
        result = blended

    result = np.clip(result, 0, 255).astype(np.uint8)

    if output_path is not None:
        Image.fromarray(result).save(output_path)

    return result

if __name__ == "__main__":

    cv2.setNumThreads(1)

    dataset_name = "endonerf"
    clip_name = "pulling"
    alpha = 1.0
    contour_exist = False

    gt_mask_dir = os.path.join("./data", dataset_name, clip_name, "gt_masks")
    output_dir = os.path.join("./output", dataset_name, clip_name, "video", "ours_3000")

    render_folder = "renders/"
    render_dir = os.path.join(output_dir, render_folder)

    gt_folder = "gt/"
    gt_dir = os.path.join(output_dir, gt_folder)

    contour_folder = "contours/"
    contour_dir = os.path.join(output_dir, contour_folder)
    os.makedirs(contour_dir, exist_ok=True)

    overlay_folder = "overlay/"
    overlay_dir = os.path.join(output_dir, overlay_folder, "Alpha_{}".format(int(alpha*100)))
    os.makedirs(overlay_dir, exist_ok=True)

    renders = sorted(os.listdir(render_dir))

    for render in tqdm(renders, desc="Overlay Contour Progress"):
        image_name = render.split(".")[0]
        mask_path = os.path.join(gt_mask_dir, "frame-0" + image_name + ".mask.png")
        contour_path = os.path.join(contour_dir, image_name + ".png")

        extract_mask_contour(mask_path, contour_path, thickness=2)

        render_path = os.path.join(render_dir, image_name + ".png")
        gt_path = os.path.join(gt_dir, image_name + ".png")
        output_path = os.path.join(overlay_dir, "{}.png".format(image_name))

        if alpha > 0:
            tool_overlayed = overlay_image(render_path, gt_path, mask = mask_path, alpha=alpha)
            overlay_contour(tool_overlayed, contour_path, output_path)
        else:
            overlay_contour(render_path, contour_path, output_path)

    video_path = os.path.join(os.path.join(output_dir, overlay_folder), "Alpha_{}.mp4".format(int(alpha*100)))
    images_to_video(overlay_dir, video_path, fps=30)
