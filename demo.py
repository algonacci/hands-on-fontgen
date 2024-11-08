import marimo

__generated_with = "0.9.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""### Crop Center""")
    return


@app.cell
def __(mo):
    from PIL import Image, UnidentifiedImageError
    import os

    widths = []
    heights = []

    for root, _, filenames in mo.status.progress_bar(list(os.walk("./dataset/"))):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            try:
                img = Image.open(file_path)
            except UnidentifiedImageError:
                continue
            widths.append(img.width)
            heights.append(img.height)


    max_width = max(widths)
    max_height = max(heights)
    return (
        Image,
        UnidentifiedImageError,
        file_path,
        filename,
        filenames,
        heights,
        img,
        max_height,
        max_width,
        os,
        root,
        widths,
    )


@app.cell
def __(Image):
    def center_crop(image: Image.Image, new_width: int, new_height: int):
        width, height = image.size

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        return image.crop((left, top, right, bottom))
    return (center_crop,)


@app.cell
def __(Image, center_crop, max_height, max_width):
    center_crop(Image.open("./dataset/ABeeZee-Regular/87.png"), max_width, max_height)
    return


@app.cell
def __(
    Image,
    UnidentifiedImageError,
    center_crop,
    max_height,
    max_width,
    mo,
    os,
):
    cropped_images = []

    for root2, _, filenames2 in mo.status.progress_bar(list(os.walk("./dataset/"))):
        for filename2 in filenames2:
            file_path2 = os.path.join(root2, filename2)
            try:
                image2 = Image.open(file_path2)
            except UnidentifiedImageError:
                continue
            cropped_images.append(center_crop(image2, max_width, max_height))
    return cropped_images, file_path2, filename2, filenames2, image2, root2


@app.cell
def __(cropped_images):
    cropped_images[:5]
    return


if __name__ == "__main__":
    app.run()
