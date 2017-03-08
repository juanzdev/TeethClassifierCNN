from Tkinter import *
from tkMessageBox import *
import PIL.Image
from PIL import  ImageTk
import glob
import os
from shutil import copyfile

dataset_to_label = [img for img in glob.glob("../img/original_data/b/*jpg")]
output_folder = "../img/original_data/b_labeled2/"
first_image = "../img/original_data/b/i000rb-fn.jpg"
current_index = 0

def callbackNO(self,event=None):
    #save current image label to yes
    global current_index
    print(dataset_to_label[current_index])
    path = dataset_to_label[current_index]
    head, tail = os.path.split(path)
    name = tail.split(".")[0]
    extension = tail.split(".")[1]
    new_name = name + "_not."
    output_path = output_folder+"/"+new_name+extension
    copyfile(path, output_path)

    #change to next picture
    current_index = current_index + 1
    path = dataset_to_label[current_index]
    updated_picture = ImageTk.PhotoImage(PIL.Image.open(path))
    ImageLabel.configure(image = updated_picture)
    ImageLabel.image = updated_picture

def callbackYES(self,event=None):
    #save current image label to yes
    global current_index
    print(dataset_to_label[current_index])
    path = dataset_to_label[current_index]
    head, tail = os.path.split(path)
    name = tail.split(".")[0]
    extension = tail.split(".")[1]
    new_name = name + "_showingteeth."
    output_path = output_folder+"/"+new_name+extension
    copyfile(path, output_path)

    #change to next picture
    current_index = current_index + 1
    path = dataset_to_label[current_index]
    updated_picture = ImageTk.PhotoImage(PIL.Image.open(path))
    ImageLabel.configure(image = updated_picture)
    ImageLabel.image = updated_picture



root = Tk()
root.bind('y', callbackYES)
root.bind('n', callbackNO)

Image = ImageTk.PhotoImage(PIL.Image.open(first_image))
ImageLabel = Label(root, image=Image)
ImageLabel.image = Image
ImageLabel.pack()
Button(text='Showing Teeth', command=callbackYES).pack(fill=X)
Button(text='NOT Showing Teeth', command=callbackNO).pack(fill=X)
root.mainloop()

