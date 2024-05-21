import json
import os
import tkinter as tk
from tkinter import colorchooser, filedialog
from PIL import Image
from alive_progress import alive_bar

data_folder="./out_formatted"
color1=(0,255,255)
color2=(255,0,0)

def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate on the scale given by a to b, using t as the point on that scale.
    Examples
    --------
        50 == lerp(0, 100, 0.5)
        4.2 == lerp(1, 5, 0.8)
    """
    return (1 - t) * a + t * b
def lerp_color(c1, c2, t):
    r = lerp(c1[0], c2[0], t)
    g = lerp(c1[1], c2[1], t)
    b = lerp(c1[2], c2[2], t)
    return (int(r), int(g), int(b))


def get_data(f:str):
    with open(f, 'r') as f:
        return json.load(f)["data"]

def get_weight_count(files:list[str]):
    return len(get_data(f"{data_folder}/{files[0]}"))

def make_image():
    #Récuperation des donnés d'entrainement triés par epoch
    files = sorted(os.listdir(data_folder), key=lambda f: int(f.split("_")[-1].split(".")[0]))
    #Récuperation du nombre de poids dans le réseau
    weights_count = get_weight_count(files)

    #Nouvelle image de dimensions (x:nombre de poids dans le réseau    y:nombre de cycles d'entrainement)
    img = Image.new(mode="RGB", size=(weights_count, len(files)))

    with alive_bar(len(files) * weights_count) as bar:
        for i, f in enumerate(files):
            data = get_data(f"{data_folder}/{f}")
            for j in range(len(data)):
                img.putpixel((j, i), lerp_color(color1, color2, data[j]))
                bar()

    path = filedialog.asksaveasfilename(confirmoverwrite=True, defaultextension=".png", filetypes=[("PNG image", "*.png")], title="Save image as")
    img.save(path)


root = tk.Tk()

def ch_path():
    global data_folder
    data_folder = filedialog.askdirectory(initialdir=data_folder, title="Choose directory")
    path_label.configure(text=data_folder)
path_label = tk.Label(root, text=data_folder)
path_button = tk.Button(root, text="Change data path", command=ch_path)
path_label.grid(row=0, column=0, sticky="NSEW")
path_button.grid(row=0, column=1, sticky="NSEW", padx=2)

def ch_c1():
    global color1
    c1 = colorchooser.askcolor(title ="Choose color", initialcolor=_from_rgb(color1))
    if (isinstance(c1, tuple) and c1[0] == None and c1[1] == None):
        return
    if (isinstance(c1, str)):
        print("Color not supported")
        return
    color1=c1[0]
    c1_label.configure(background=_from_rgb(color1))
c1_label = tk.Label(root, background=_from_rgb(color1), width=10)
c1_button = tk.Button(root, text="Change color 1", command=ch_c1)
c1_label.grid(row=1, column=0, sticky="NSEW")
c1_button.grid(row=1, column=1, sticky="NSEW", padx=2)

def ch_c2():
    global color2
    c2 = colorchooser.askcolor(title ="Choose color", initialcolor=_from_rgb(color2))
    if (isinstance(c2, tuple) and c2[0] == None and c2[1] == None):
        return
    color2=c2[0]
    c2_label.configure(background=_from_rgb(color2))
c2_label = tk.Label(root, background=_from_rgb(color2), width=10)
c2_button = tk.Button(root, text="Change color 2", command=ch_c2)
c2_label.grid(row=2, column=0, sticky="NSEW")
c2_button.grid(row=2, column=1, sticky="NSEW", padx=2)

run_button = tk.Button(root, text="Make image", command=make_image)
run_button.grid(row=3, sticky="NSEW", columnspan=2, pady=10)

root.mainloop()